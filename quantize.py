#!/usr/bin/env python3
"""
Post-training quantization for AutoModelForCausalLM.
Supports multiple formats: q2_kmeans (default), q4_k (GGML-style 4-bit blocks).

CPU-based, vectorised quantization.  No GPU sync per layer.
"""

import argparse
import json
import logging
import os
import zlib

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MAXQ = {2: 3, 3: 7, 4: 15, 8: 255}
DEFAULT_GROUPSIZE = 32
VRAM_LIMIT_MB = 8192

# Max compressed size: Q4_K baseline for 0.8B is ~450 MB.  We'll know the
# exact number after the first Q4_K run and update this.
MAX_COMPRESSED_MB = 500

_NEVER_QUANTIZE = frozenset([
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
])

# Q4_K block geometry (GGML-compatible)
QK_K = 256
QK_K_SUB_BLOCKS = 8
QK_K_SUB_SIZE = QK_K // QK_K_SUB_BLOCKS  # 32


def _get_layer_groupsize(name: str, default_groupsize: int) -> int:
    attn_keywords = ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head")
    if any(kw in name for kw in attn_keywords):
        return max(16, default_groupsize // 2)
    return default_groupsize


# ---------------------------------------------------------------------------
# Activation statistics collection (calibration data)
# ---------------------------------------------------------------------------

def _collect_input_stats(
    model: nn.Module,
    model_name: str,
    nsamples: int = 16,
    seqlen: int = 1024,
) -> dict:
    from data_utils import get_wikitext2
    device = next(model.parameters()).device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    samples = get_wikitext2(nsamples, seqlen, tokenizer, split="train")

    stats = {}
    handles = []

    def _hook(name):
        def fn(_module, _input, _output):
            x = _input[0].detach().float()
            sq = (x * x).sum(dim=(0, 1)).cpu()
            cnt = x.size(0) * x.size(1)
            if name in stats:
                stats[name][0] += sq
                stats[name][1] += cnt
            else:
                stats[name] = [sq, cnt]
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(_hook(name)))

    try:
        with torch.no_grad():
            for sample in tqdm(samples, desc="Calibrating"):
                model(sample.to(device))
    finally:
        for h in handles:
            h.remove()

    return {name: s[0] / max(s[1], 1) for name, s in stats.items()}


# ---------------------------------------------------------------------------
# Q4_K quantization (GGML-style 4-bit block quantization)
# Superblock: 256 weights, 8 sub-blocks of 32.
# Per superblock: fp16 d + fp16 dmin (4 bytes).
# Per sub-block: 6-bit scale + 6-bit min (12 bytes / 8 sub-blocks).
# Per weight: 4-bit quantized value (128 bytes / 256 weights).
# Total: 144 bytes per 256 weights → 4.5 bits/weight.
# ---------------------------------------------------------------------------

def _quantize_one_layer_q4k(layer: nn.Linear, n_sub_blocks: int = QK_K_SUB_BLOCKS) -> dict:
    W = layer.weight.data.float()
    out_features, in_features = W.shape
    sub_size = QK_K // n_sub_blocks

    assert in_features % QK_K == 0, (
        f"in_features ({in_features}) must be divisible by QK_K ({QK_K})")
    assert QK_K % n_sub_blocks == 0, (
        f"QK_K ({QK_K}) must be divisible by n_sub_blocks ({n_sub_blocks})")
    n_blocks = in_features // QK_K

    W_r = W.reshape(out_features, n_blocks, n_sub_blocks, sub_size)
    # shape: (out, n_blocks, n_sub_blocks, sub_size)

    # Per-sub-block min/max → per-block (superblock) min/max
    w_sub_min = W_r.amin(dim=-1)      # (out, n_blocks, 8)
    w_sub_max = W_r.amax(dim=-1)
    w_blk_min = w_sub_min.amin(dim=-1)  # (out, n_blocks)
    w_blk_max = w_sub_max.amax(dim=-1)

    # Global scale d and min-offset base dmin (positive magnitude)
    d = (w_blk_max - w_blk_min) / 15.0
    d[d < 1e-8] = 1e-8
    dmin = w_blk_min.abs().clamp(min=1e-8)

    # Per-sub-block effective scale and min
    d_sub = (w_sub_max - w_sub_min) / 15.0
    d_sub[d_sub < 1e-8] = 1e-8

    sc_ratio = d_sub / d.unsqueeze(-1).clamp(min=1e-8)
    sc_6bit = torch.clamp(torch.round(sc_ratio * 63.0), 1, 63).to(torch.uint8)
    sc_norm = sc_6bit.float() / 63.0

    min_ratio = w_sub_min.abs() / dmin.unsqueeze(-1).clamp(min=1e-8)
    m_6bit = torch.clamp(torch.round(min_ratio * 63.0), 0, 63).to(torch.uint8)
    m_norm = m_6bit.float() / 63.0

    eff_scale = d.unsqueeze(-1) * sc_norm         # (out, n_blocks, 8)
    eff_scale[eff_scale < 1e-8] = 1e-8
    eff_offset = dmin.unsqueeze(-1) * m_norm       # (out, n_blocks, 8)

    # Quantize:  w ≈ eff_scale * q - eff_offset,   q ∈ {0..15}
    q = torch.round(
        (W_r + eff_offset.unsqueeze(-1)) / eff_scale.unsqueeze(-1)
    )
    q = torch.clamp(q, 0, 15).to(torch.uint8)

    # Dequantize back into the weight tensor
    W_q_r = eff_scale.unsqueeze(-1) * q.float() - eff_offset.unsqueeze(-1)
    W_q = W_q_r.reshape(out_features, in_features)
    layer.weight.data = W_q.to(layer.weight.dtype)

    quants_flat = q.reshape(out_features, in_features)

    fmt_tag = f"q4_k" if n_sub_blocks == QK_K_SUB_BLOCKS else f"q4_k_sb{n_sub_blocks}"
    return {
        "d":              d.numpy().astype(np.float16),
        "dmin":           dmin.numpy().astype(np.float16),
        "scales_6bit":    sc_6bit.numpy(),          # (out, n_blocks, n_sub_blocks)
        "mins_6bit":      m_6bit.numpy(),           # (out, n_blocks, n_sub_blocks)
        "quants":         quants_flat.numpy(),      # (out, in_features) raw 0-15
        "format":         fmt_tag,
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
        "n_sub_blocks":   n_sub_blocks,
    }


# ---------------------------------------------------------------------------
# Per-layer 2-bit K-means quantization (vectorised, CPU)
# ---------------------------------------------------------------------------

def _quantize_one_layer(
    layer: nn.Linear,
    bits: int,
    groupsize: int,
    act_stats: torch.Tensor | None = None,
    diffusion: float = 0.5,
) -> dict:
    W = layer.weight.data.float()                  # CPU float32
    out_features, in_features = W.shape
    g = groupsize if groupsize != -1 else in_features
    n_groups = in_features // g
    maxq = MAXQ[bits]
    num_entries = maxq + 1
    zero_pt = (maxq + 1) / 2

    did_sort = act_stats is not None
    sort_idx = None
    unsort_idx = None
    if did_sort:
        sort_idx = act_stats.argsort()
        unsort_idx = sort_idx.argsort()
        W = W[:, sort_idx]
        act_stats = act_stats[sort_idx]

    W_orig = W.clone()

    codes = torch.zeros(out_features, in_features, dtype=torch.uint8)
    codebooks = torch.zeros(out_features, n_groups, num_entries, dtype=torch.bfloat16)
    W_q_full = torch.zeros(out_features, in_features)

    for i in range(n_groups):
        start = i * g
        end = start + g
        W_g = W[:, start:end]                      # view

        xmax = torch.maximum(W_g.amin(dim=-1).abs(),
                             W_g.amax(dim=-1))
        scale = xmax / (maxq / 2)
        scale[scale == 0] = 1.0

        for _ in range(3):
            q = torch.clamp(torch.round(W_g / scale.unsqueeze(-1)) + zero_pt,
                            0, maxq)
            q_centered = q - zero_pt
            if act_stats is not None:
                h_g = act_stats[start:end]
                num = (W_g * q_centered * h_g).sum(dim=-1)
                den = (q_centered * q_centered * h_g).sum(dim=-1)
            else:
                num = (W_g * q_centered).sum(dim=-1)
                den = (q_centered * q_centered).sum(dim=-1)
            den[den == 0] = 1.0
            scale = num / den
            scale[scale <= 0] = 1.0

        W_g_sorted = W_g.sort(dim=-1).values
        quantile_idxs = [int(g * j / (num_entries - 1)) for j in range(num_entries)]
        quantile_idxs[-1] = g - 1
        cb_init = W_g_sorted[:, quantile_idxs]

        h_g = act_stats[start:end] if act_stats is not None else None

        best_err = None
        best_cb = None

        for trial in range(3):
            if trial == 0:
                cb = cb_init.clone()
            else:
                cb_std = cb_init.std(dim=-1, keepdim=True).clamp(min=1e-8)
                cb_noise = torch.randn_like(cb_init) * cb_std * 0.05
                cb = cb_init + cb_noise

            for _ in range(20):
                dists = (W_g.unsqueeze(-1) - cb.unsqueeze(1)).pow(2)
                if h_g is not None:
                    dists = (dists * h_g.unsqueeze(0).unsqueeze(-1)).sqrt()
                assign = dists.argmin(dim=-1)
                for j in range(num_entries):
                    mask_j = (assign == j).float()
                    if h_g is not None:
                        weighted_W = W_g * h_g.unsqueeze(0)
                        summed = (weighted_W * mask_j).sum(dim=-1)
                        count = (h_g.unsqueeze(0) * mask_j).sum(dim=-1).clamp(min=1e-8)
                    else:
                        summed = (W_g * mask_j).sum(dim=-1)
                        count = mask_j.sum(dim=-1).clamp(min=1)
                    cb[:, j] = summed / count

            final_dists = (W_g.unsqueeze(-1) - cb.unsqueeze(1)).pow(2)
            if h_g is not None:
                final_dists = final_dists * h_g.unsqueeze(0).unsqueeze(-1)
            trial_err = final_dists.min(dim=-1).values.sum(dim=-1)

            if best_err is None:
                best_err = trial_err
                best_cb = cb.clone()
            else:
                better = trial_err < best_err
                if better.any():
                    best_err[better] = trial_err[better]
                    best_cb[better] = cb[better]

        dists = (W_g.unsqueeze(-1) - best_cb.unsqueeze(1)).pow(2)
        if h_g is not None:
            dists = dists * h_g.unsqueeze(0).unsqueeze(-1)
        q_idx = dists.argmin(dim=-1)
        W_q = torch.gather(best_cb, 1, q_idx)

        error = W_g - W_q
        codes[:, start:end] = q_idx.to(torch.uint8)
        codebooks[:, i, :] = best_cb.to(torch.bfloat16)
        W_q_full[:, start:end] = W_q

        if i + 1 < n_groups:
            ns = (i + 1) * g
            ne = ns + g
            if act_stats is not None:
                h_next = act_stats[ns:ne]
                eps_w = (1.0 / h_next.clamp(min=1e-8))
                eps_w = eps_w / eps_w.mean()
                eps_w = eps_w.clamp(max=3.0)
                W[:, ns:ne] += diffusion * error * eps_w
            else:
                W[:, ns:ne] += diffusion * error

    cb_flat = codebooks.float().reshape(-1)
    cb_min = cb_flat.min()
    cb_max = cb_flat.max()
    range_val = cb_max - cb_min
    if range_val < 1e-8:
        cb_q = torch.zeros(cb_flat.shape, dtype=torch.uint8)
        cb_deq_flat = cb_flat
    else:
        cb_q = torch.clamp(
            torch.round((cb_flat - cb_min) / range_val * 255.0),
            0, 255,
        ).to(torch.uint8)
        cb_deq_flat = cb_min + cb_q.float() * (range_val / 255.0)
    cb_deq = cb_deq_flat.reshape(out_features, n_groups, num_entries).to(torch.bfloat16)

    for i in range(n_groups):
        start = i * g
        end = start + g
        cb_i = cb_deq[:, i, :]
        W_g_orig = W_orig[:, start:end]
        dists = (W_g_orig.unsqueeze(-1) - cb_i.unsqueeze(1)).pow(2)
        if act_stats is not None:
            h_g = act_stats[start:end]
            dists = dists * h_g.unsqueeze(0).unsqueeze(-1)
        new_codes = dists.argmin(dim=-1)
        codes[:, start:end] = new_codes.to(torch.uint8)
        W_q_full[:, start:end] = torch.gather(cb_i, 1, new_codes.long())

    bias = (W_orig - W_q_full).mean(dim=-1)
    W_q_full += bias.unsqueeze(-1)

    if did_sort:
        W_q_full = W_q_full[:, unsort_idx]

    layer.weight.data = W_q_full.to(layer.weight.dtype)

    return {
        "codes":     codes.numpy(),
        "codebook_q": cb_q.numpy().astype(np.uint8),
        "cb_min":    float(cb_min),
        "cb_max":    float(cb_max),
        "codebook_dtype": "q8_codebook",
        "shape":     [out_features, in_features],
        "n_groups":  n_groups,
    }


# ---------------------------------------------------------------------------
# Compressed storage
# ---------------------------------------------------------------------------

def _pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 4 × 2-bit values into one uint8."""
    out, inp = codes.shape
    assert inp % 4 == 0
    codes = codes.reshape(out, inp // 4, 4)
    packed = (codes[..., 0]
              | (codes[..., 1] << 2)
              | (codes[..., 2] << 4)
              | (codes[..., 3] << 6))
    return packed.astype(np.uint8)


def _pack_4bit(codes: np.ndarray) -> np.ndarray:
    """Pack 2 × 4-bit values into one uint8."""
    out, inp = codes.shape
    assert inp % 2 == 0
    codes = codes.reshape(out, inp // 2, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4))
    return packed.astype(np.uint8)


def _pack_3bit(codes: np.ndarray) -> np.ndarray:
    """Pack 8 × 3-bit values into 3 uint8 bytes."""
    out, inp = codes.shape
    assert inp % 8 == 0
    codes = codes.reshape(out, inp // 8, 8).astype(np.uint32)
    packed_24 = (codes[..., 0]
                 | (codes[..., 1] << 3)
                 | (codes[..., 2] << 6)
                 | (codes[..., 3] << 9)
                 | (codes[..., 4] << 12)
                 | (codes[..., 5] << 15)
                 | (codes[..., 6] << 18)
                 | (codes[..., 7] << 21))
    b = np.zeros((out, inp // 8, 3), dtype=np.uint8)
    b[..., 0] = packed_24 & 0xFF
    b[..., 1] = (packed_24 >> 8) & 0xFF
    b[..., 2] = (packed_24 >> 16) & 0xFF
    return b.reshape(out, -1)


def _save_compressed(meta: dict, save_dir: str, bits: int, fmt: str = "q2_kmeans") -> int:
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    for name, data in tqdm(meta.items(), desc="Saving compressed"):
        fname = os.path.join(save_dir, name + ".npz")
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if data.get("format", "").startswith("q4_k"):
            # Q4_K format: pack 4-bit quants into uint8 pairs
            quants = data["quants"]
            q_out = _pack_4bit(quants)
            if fmt == "q4_k_zlib":
                q_raw = q_out.tobytes()
                q_compressed = zlib.compress(q_raw, level=9)
                q_compressed_arr = np.frombuffer(q_compressed, dtype=np.uint8)
                np.savez(fname,
                         quants_zlib=q_compressed_arr,
                         quants_shape=q_out.shape,
                         d=data["d"],
                         dmin=data["dmin"],
                         scales_6bit=data["scales_6bit"],
                         mins_6bit=data["mins_6bit"],
                         format="q4_k_zlib")
            else:
                np.savez(fname,
                         quants=q_out,
                         d=data["d"],
                         dmin=data["dmin"],
                         scales_6bit=data["scales_6bit"],
                         mins_6bit=data["mins_6bit"],
                         format=data["format"])
        elif bits == 3 and data["shape"][1] % 8 == 0:
            codes_out = _pack_3bit(data["codes"])
            if "codebook_q" in data:
                np.savez(fname,
                         codes=codes_out,
                         codebook_q=data["codebook_q"],
                         cb_min=data["cb_min"],
                         cb_max=data["cb_max"],
                         codebook_dtype=data["codebook_dtype"])
            else:
                np.savez(fname,
                         codes=codes_out,
                         scales=data["scales"],
                         scale_dtype=data["scale_dtype"],
                         zero_pt=data["zero_pt"])
        elif bits == 4 and data["shape"][1] % 2 == 0:
            codes_out = _pack_4bit(data["codes"])
            np.savez(fname, codes=codes_out,
                     codebook_q=data.get("codebook_q"),
                     cb_min=data.get("cb_min"),
                     cb_max=data.get("cb_max"),
                     codebook_dtype=data.get("codebook_dtype", "q4"))
        elif bits == 2 and data["shape"][1] % 4 == 0:
            codes_out = _pack_2bit(data["codes"])
            if "codebook_q" in data:
                np.savez(fname,
                         codes=codes_out,
                         codebook_q=data["codebook_q"],
                         cb_min=data["cb_min"],
                         cb_max=data["cb_max"],
                         codebook_dtype=data["codebook_dtype"])
            elif "codebook" in data:
                np.savez(fname,
                         codes=codes_out,
                         codebook=data["codebook"],
                         codebook_dtype=data["codebook_dtype"])
            else:
                np.savez(fname,
                         codes=codes_out,
                         scales=data["scales"],
                         scale_dtype=data["scale_dtype"],
                         zero_pt=data["zero_pt"])
        else:
            codes_out = data["codes"]
            if "codebook_q" in data:
                np.savez(fname,
                         codes=codes_out,
                         codebook_q=data["codebook_q"],
                         cb_min=data["cb_min"],
                         cb_max=data["cb_max"],
                         codebook_dtype=data["codebook_dtype"])
            else:
                np.savez(fname,
                         codes=codes_out,
                         scales=data["scales"],
                         scale_dtype=data["scale_dtype"],
                         zero_pt=data["zero_pt"])
        total_bytes += os.path.getsize(fname)
        layer_info[name] = data["shape"]

    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({
            "bits": bits,
            "format": fmt,
            "layers": layer_info,
            "total_compressed_mb": round(total_bytes / 1e6, 2),
        }, f, indent=2)

    logger.info("Compressed storage: %.1f MB (%d layers)",
                total_bytes / 1e6, len(meta))
    return total_bytes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    bits: int = 2,
    groupsize: int = DEFAULT_GROUPSIZE,
    save_compressed_dir: str | None = None,
    act_stats: dict | None = None,
    fmt: str = "q2_kmeans",
) -> dict:
    model.eval()
    model.cpu()
    meta: dict = {}

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear)]

    skipped_small = 0
    for name, layer in tqdm(layers, desc="Quantizing"):
        if any(pattern in name for pattern in _NEVER_QUANTIZE):
            skipped_small += 1
            continue

        if fmt == "q4_k" or fmt == "q4_k_zlib":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k(layer)
        elif fmt == "q4_k_lb":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k(layer, n_sub_blocks=4)
        else:
            layer_gs = _get_layer_groupsize(name, groupsize)
            if groupsize != -1 and layer.weight.shape[1] % layer_gs != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], layer_gs)
                continue
            layer_act = act_stats.get(name) if act_stats is not None else None
            if "lm_head" in name:
                layer_diffusion = 0.1
            elif any(kw in name for kw in ("q_proj", "k_proj", "v_proj", "o_proj")):
                layer_diffusion = 0.7
            elif any(kw in name for kw in ("gate_proj", "up_proj", "down_proj")):
                layer_diffusion = 0.3
            else:
                layer_diffusion = 0.5
            meta[name] = _quantize_one_layer(layer, bits, layer_gs,
                                             act_stats=layer_act,
                                             diffusion=layer_diffusion)

    if save_compressed_dir:
        _save_compressed(meta, save_compressed_dir, bits, fmt)

    if skipped_small:
        logger.info("Skipped %d small/SSM layers (never quantize)", skipped_small)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training quantization for causal LMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--groupsize", type=int, default=DEFAULT_GROUPSIZE,
                        help="Group size (min 16, must divide in_features)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Precision for loading the base model")
    parser.add_argument("--save", default=None,
                        help="Directory to save quantized model")
    parser.add_argument("--calibration-cache", default=None,
                        help="Path to cache/load calibration stats (.npz file)")
    parser.add_argument("--format", default="q2_kmeans",
                        choices=["q2_kmeans", "q3_kmeans", "q4_k", "q4_k_zlib", "q4_k_lb"],
                        help="Quantization format (q2_kmeans=K-means 2-bit, q3_kmeans=K-means 3-bit, q4_k=GGML-style 4-bit blocks, q4_k_zlib=Q4_K+zlib, q4_k_lb=Q4_K large sub-blocks)")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    logger.info("Loading model: %s (dtype=%s)", args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    act_stats = None
    if args.format in ("q2_kmeans", "q3_kmeans"):
        if args.calibration_cache and os.path.exists(args.calibration_cache):
            logger.info("Loading cached calibration stats: %s", args.calibration_cache)
            cached = np.load(args.calibration_cache, allow_pickle=True)
            act_stats = {k: torch.from_numpy(cached[k]) for k in cached.files}
            model.to("cuda")
        elif torch.cuda.is_available():
            logger.info("Collecting calibration activation stats on GPU")
            torch.cuda.reset_peak_memory_stats()
            model.to("cuda")
            act_stats = _collect_input_stats(model, args.model,
                                             nsamples=16, seqlen=1024)
            logger.info("Collected stats for %d layers", len(act_stats))
            if args.calibration_cache and act_stats:
                os.makedirs(os.path.dirname(args.calibration_cache) or ".", exist_ok=True)
                np.savez(args.calibration_cache,
                         **{k: v.numpy() for k, v in act_stats.items()})
                logger.info("Cached calibration stats to %s", args.calibration_cache)

    compressed_dir = os.path.join(args.save, "compressed") if args.save else None

    effective_bits = 4 if args.format in ("q4_k", "q4_k_zlib", "q4_k_lb") else (3 if args.format == "q3_kmeans" else args.bits)
    logger.info("Quantizing  format=%s  bits=%d  groupsize=%d  symmetric=True",
                args.format, effective_bits, args.groupsize)
    meta = quantize_model(
        model,
        bits=effective_bits,
        groupsize=args.groupsize,
        save_compressed_dir=compressed_dir,
        act_stats=act_stats,
        fmt=args.format,
    )
    logger.info("Quantized %d linear layers.", len(meta))

    # ---- resource limit enforcement ----
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    if peak_vram > VRAM_LIMIT_MB:
        logger.error(
            "Peak VRAM %.1f MB exceeds limit of %d MB.",
            peak_vram, VRAM_LIMIT_MB)
        raise SystemExit(1)
    logger.info("Peak VRAM: %.1f MB (limit %d MB)", peak_vram, VRAM_LIMIT_MB)

    if compressed_dir and meta:
        total_mb = round(
            sum(os.path.getsize(os.path.join(compressed_dir, f))
                for f in os.listdir(compressed_dir) if f.endswith(".npz"))
            / 1e6, 2)
        if total_mb > MAX_COMPRESSED_MB:
            logger.error(
                "Compressed size %.1f MB exceeds limit of %d MB. "
                "Increase --groupsize to reduce overhead.",
                total_mb, MAX_COMPRESSED_MB)
            raise SystemExit(1)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)
        logger.info("Saved dequantised model to '%s'", args.save)


if __name__ == "__main__":
    main()
