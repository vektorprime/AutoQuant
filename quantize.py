#!/usr/bin/env python3
"""
q2_k post-training quantization for AutoModelForCausalLM.

CPU-based, vectorised per-group min/max symmetric quantization.
No GPU sync per layer — the entire model is quantised on CPU in one pass.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Quantization is always symmetric — hardcoded.
MAXQ = {2: 3, 3: 7, 4: 15, 8: 255}
DEFAULT_GROUPSIZE = 32
MAX_COMPRESSED_MB = 1575  # 1500 + 5% tolerance
VRAM_LIMIT_MB = 8192

# Modules that must NEVER be quantized (SSM projections, norms, small params).
# Norms and Conv1d are already excluded by the nn.Linear filter — this list
# catches nn.Linear sub-modules that should stay at native precision.
_NEVER_QUANTIZE = frozenset([
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
])


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
# Per-layer 2-bit symmetric quantization (vectorised, CPU)
# ---------------------------------------------------------------------------

def _quantize_one_layer(
    layer: nn.Linear,
    bits: int,
    groupsize: int,
    act_stats: torch.Tensor | None = None,
) -> dict:
    W = layer.weight.data.float()
    out_features, in_features = W.shape
    g = groupsize if groupsize != -1 else in_features
    n_groups = in_features // g
    maxq = MAXQ[bits]
    zero_pt = (maxq + 1) / 2                       # 2.0 for 2-bit
    diffusion = 0.5

    codes = torch.zeros(out_features, in_features, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(out_features, n_groups, device=W.device)
    W_q_full = torch.zeros(out_features, in_features, device=W.device)

    if act_stats is not None:
        act_stats = act_stats.to(W.device)

    for i in range(n_groups):
        start = i * g
        end = start + g
        W_g = W[:, start:end]                      # view

        xmax = torch.maximum(W_g.amin(dim=-1).abs(),
                             W_g.amax(dim=-1))
        xmax = torch.clamp(xmax, min=1e-8)
        scale = xmax / (maxq / 2)

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
            den = torch.clamp(den, min=1e-8)
            scale = num / den
            scale = torch.clamp(scale, min=1e-8)

        q = torch.clamp(torch.round(W_g / scale.unsqueeze(-1)) + zero_pt,
                        0, maxq)
        W_q = scale.unsqueeze(-1) * (q - zero_pt)

        error = W_g - W_q
        codes[:, start:end] = q.to(torch.uint8)
        scales[:, i] = scale
        W_q_full[:, start:end] = W_q

        if i + 1 < n_groups:
            ns = (i + 1) * g
            ne = ns + g
            W[:, ns:ne] += diffusion * error

    layer.weight.data = W_q_full.to(layer.weight.dtype)

    return {
        "codes":  codes.cpu().numpy(),
        "scales": scales.to(torch.bfloat16).cpu().view(torch.int16).numpy(),
        "scale_dtype": "bfloat16",
        "zero_pt": zero_pt,
        "shape":  [out_features, in_features],
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


def _save_compressed(meta: dict, save_dir: str, bits: int) -> int:
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    for name, data in tqdm(meta.items(), desc="Saving compressed"):
        fname = os.path.join(save_dir, name + ".npz")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if bits == 4 and data["shape"][1] % 2 == 0:
            codes_out = _pack_4bit(data["codes"])
        elif bits == 2 and data["shape"][1] % 4 == 0:
            codes_out = _pack_2bit(data["codes"])
        else:
            codes_out = data["codes"]
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
) -> dict:
    model.eval()                                    # keep on GPU if already there
    meta: dict = {}

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear)]

    skipped_small = 0
    for name, layer in tqdm(layers, desc="Quantizing"):
        layer_gs = _get_layer_groupsize(name, groupsize)
        if groupsize != -1 and layer.weight.shape[1] % layer_gs != 0:
            logger.warning("Skipping %s: in_features %d not divisible by %d",
                           name, layer.weight.shape[1], layer_gs)
            continue
        if any(pattern in name for pattern in _NEVER_QUANTIZE):
            skipped_small += 1
            continue
        layer_act = act_stats.get(name) if act_stats is not None else None
        meta[name] = _quantize_one_layer(layer, bits, layer_gs,
                                         act_stats=layer_act)

    if save_compressed_dir:
        _save_compressed(meta, save_compressed_dir, bits)

    if skipped_small:
        logger.info("Skipped %d small/SSM layers (never quantize)", skipped_small)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="q2_k quantization for causal LMs (symmetric, always on)",
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

    logger.info("Quantizing  bits=%d  groupsize=%d  symmetric=True",
                args.bits, args.groupsize)
    meta = quantize_model(
        model,
        bits=args.bits,
        groupsize=args.groupsize,
        save_compressed_dir=compressed_dir,
        act_stats=act_stats,
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
