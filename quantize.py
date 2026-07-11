#!/usr/bin/env python3
"""
Post-training quantization for AutoModelForCausalLM.
Supports q2_kmeans and a custom Q4_K-like affine packed format.

CPU-based, vectorised quantization.  No GPU sync per layer.
"""

import argparse
import json
import logging
import os

import numpy as np
import safetensors.torch as st
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
MAX_COMPRESSED_MB = 6000

_NEVER_QUANTIZE = frozenset([
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
])

# Q4_K-like block geometry (custom storage; not byte-compatible with GGML)
QK_K = 256
QK_K_SUB_BLOCKS = 8
QK_K_SUB_SIZE = QK_K // QK_K_SUB_BLOCKS  # 32

# Packed format constants. Version 2 stores one d/dmin pair per output row
# and superblock.  The scale dtype is recorded in the manifest.
Q4K_FORMAT_NAME = "q4k_affine_v2"
Q4K_PACKED_KEYS = ["quants_packed", "d", "dmin", "scales_mins_packed"]


def _normalise_shape(shape) -> tuple[int, int]:
    if isinstance(shape, torch.Tensor):
        shape = shape.detach().cpu().tolist()
    return int(shape[0]), int(shape[1])


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

    return {name: value[0] / max(value[1], 1) for name, value in stats.items()}


# ---------------------------------------------------------------------------
# Q4_K-like affine block quantization
#
# Superblock: 256 weights, 8 sub-blocks of 32.
# Per output row and superblock: d + dmin (fp32 by default).
# Per sub-block: 6-bit scale + 6-bit min (packed into 12 bytes).
# Per weight: 4-bit code.
# ---------------------------------------------------------------------------

def decode_q4k(packed: dict, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Canonical CPU decoder used by tests and optional fake-quant checks."""
    quants_packed = torch.as_tensor(packed["quants_packed"], dtype=torch.uint8)
    # Backward-compatible aliases for checkpoints produced by the previous code.
    d_value = packed["d"] if "d" in packed else packed["d8"]
    dmin_value = packed["dmin"] if "dmin" in packed else packed["dm8"]
    d = torch.as_tensor(d_value).float()
    dmin = torch.as_tensor(dmin_value).float()
    sm = torch.as_tensor(packed["scales_mins_packed"], dtype=torch.uint8)
    out_features, in_features = _normalise_shape(packed["shape"])
    n_blocks = in_features // QK_K

    q_lo = quants_packed & 0x0F
    q_hi = quants_packed >> 4
    q_flat = torch.empty(out_features, in_features, dtype=torch.uint8)
    q_flat[:, 0::2] = q_lo
    q_flat[:, 1::2] = q_hi

    s = sm.to(dtype=torch.int32)
    packed_sm = torch.empty(out_features, n_blocks, 8, dtype=torch.int32)
    for pair in range(4):
        i = pair * 2
        b = pair * 3
        packed_sm[:, :, i] = s[:, :, b] | ((s[:, :, b + 1] & 0x0F) << 8)
        packed_sm[:, :, i + 1] = (s[:, :, b + 1] >> 4) | (s[:, :, b + 2] << 4)

    sc_norm = ((packed_sm >> 6) & 0x3F).float() / 63.0
    min_norm = (packed_sm & 0x3F).float() / 63.0
    q_blocks = q_flat.reshape(
        out_features, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE
    ).float()
    weights = (
        (d.unsqueeze(-1) * sc_norm).unsqueeze(-1) * q_blocks
        - (dmin.unsqueeze(-1) * min_norm).unsqueeze(-1)
    )
    return weights.reshape(out_features, in_features).to(dtype=dtype)


def _encode_q4k(
    layer: nn.Linear,
    scale_dtype: torch.dtype = torch.float32,
    refine_mode: str = "legacy_exact",
    refine_iters: int = 3,
    ddmin_store_dtype: torch.dtype | None = None,
) -> dict:
    """Encode one Linear weight without mutating the source model.

    refine_mode:
      * legacy_exact: reproduce the historical fake-quant BF16 path.  Codes
        are assigned once, then d/dmin are solved with those codes held fixed.
      * alternating: alternate code assignment and least-squares d/dmin solves.
      * none: min/max initialization only.

    ddmin_store_dtype: if set and lower-precision than scale_dtype, d/dmin are
      rounded to this dtype and codes + LS are re-optimized to absorb the
      rounding error.  This recovers quality lost from reduced-precision
      metadata storage.
    """
    if scale_dtype not in (torch.float16, torch.float32):
        raise ValueError("Q4_K scale dtype must be float16 or float32")
    if (ddmin_store_dtype is not None
            and ddmin_store_dtype not in (torch.float16, torch.float32)):
        raise ValueError(
            "ddmin_store_dtype must be float16 or float32"
        )
    if refine_mode not in {"legacy_exact", "alternating", "none"}:
        raise ValueError(
            "refine_mode must be one of: legacy_exact, alternating, none"
        )

    weights = layer.weight.detach().float().cpu()
    out_features, in_features = weights.shape
    if in_features % QK_K != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by {QK_K}"
        )
    n_blocks = in_features // QK_K
    blocks = weights.reshape(
        out_features, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE
    )

    sub_min = blocks.amin(dim=-1)
    sub_max = blocks.amax(dim=-1)
    block_min = sub_min.amin(dim=-1)
    block_max = sub_max.amax(dim=-1)

    d = ((block_max - block_min) / 15.0).clamp_min(1e-8)
    d_sub = ((sub_max - sub_min) / 15.0).clamp_min(1e-8)

    if refine_mode == "legacy_exact":
        # This intentionally matches d46e089, including abs() semantics.
        dmin = block_min.abs().clamp_min(1e-8)
        sub_min_magnitude = sub_min.abs()
    else:
        # Mathematically correct affine interpretation for w = d*s*q - dmin*m.
        dmin = (-block_min).clamp_min(0.0)
        sub_min_magnitude = (-sub_min).clamp_min(0.0)

    scales_6bit = torch.clamp(
        torch.round((d_sub / d.unsqueeze(-1).clamp_min(1e-8)) * 63.0),
        1,
        63,
    ).to(torch.uint8)
    mins_6bit = torch.clamp(
        torch.round(
            (sub_min_magnitude / dmin.unsqueeze(-1).clamp_min(1e-8)) * 63.0
        ),
        0,
        63,
    ).to(torch.uint8)
    scale_norm = scales_6bit.float() / 63.0
    min_norm = mins_6bit.float() / 63.0

    def assign_codes(current_d: torch.Tensor, current_dmin: torch.Tensor) -> torch.Tensor:
        effective_scale = (
            current_d.unsqueeze(-1) * scale_norm
        ).clamp_min(1e-8)
        effective_offset = current_dmin.unsqueeze(-1) * min_norm
        return torch.clamp(
            torch.round(
                (blocks + effective_offset.unsqueeze(-1))
                / effective_scale.unsqueeze(-1)
            ),
            0,
            15,
        )

    codes = assign_codes(d, dmin)
    flat_weights = weights.reshape(out_features, n_blocks, QK_K)
    expanded_min = min_norm.unsqueeze(-1).expand(
        -1, -1, -1, QK_K_SUB_SIZE
    ).flatten(start_dim=2)

    def solve_scales(
        fixed_codes: torch.Tensor,
        current_d: torch.Tensor,
        current_dmin: torch.Tensor,
        legacy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_codes = (scale_norm.unsqueeze(-1) * fixed_codes).flatten(start_dim=2)
        scale_sq = (scale_codes * scale_codes).sum(dim=-1)
        min_sq = (expanded_min * expanded_min).sum(dim=-1)
        cross = (scale_codes * expanded_min).sum(dim=-1)
        weight_scale = (flat_weights * scale_codes).sum(dim=-1)
        weight_min = (flat_weights * expanded_min).sum(dim=-1)
        determinant = scale_sq * min_sq - cross * cross

        if legacy:
            # Historical code replaced every tiny determinant with +1e-12.
            determinant = determinant.clone()
            determinant[determinant.abs() < 1e-12] = 1e-12
            solved_d = (
                weight_scale * min_sq - weight_min * cross
            ) / determinant
            solved_dmin = (
                weight_scale * cross - weight_min * scale_sq
            ) / determinant
            return (
                solved_d.clamp_min(1e-8),
                solved_dmin.abs().clamp_min(1e-8),
            )

        valid = determinant.abs() > 1e-12
        solved_d = torch.where(
            valid,
            (weight_scale * min_sq - weight_min * cross) / determinant,
            current_d,
        ).clamp_min(1e-8)
        solved_dmin = torch.where(
            valid,
            (weight_scale * cross - weight_min * scale_sq) / determinant,
            current_dmin,
        ).clamp_min(0.0)
        return solved_d, solved_dmin

    if refine_mode == "legacy_exact" and refine_iters > 0:
        # Repeating the old solve was algebraically redundant because codes did
        # not change.  One solve reproduces the final historical values.
        d, dmin = solve_scales(codes, d, dmin, legacy=True)
    elif refine_mode == "alternating":
        for _ in range(max(0, refine_iters)):
            d, dmin = solve_scales(codes, d, dmin, legacy=False)
            codes = assign_codes(d, dmin)

    effective_store_dtype = (
        ddmin_store_dtype if ddmin_store_dtype is not None else scale_dtype
    )
    if effective_store_dtype != scale_dtype and effective_store_dtype == torch.float16:
        d_rounded = d.to(torch.float16).to(torch.float32)
        dmin_rounded = dmin.to(torch.float16).to(torch.float32)
        codes = assign_codes(d_rounded, dmin_rounded)
        if refine_mode != "none":
            d, dmin = solve_scales(codes, d_rounded, dmin_rounded,
                                   legacy=(refine_mode == "legacy_exact"))
        else:
            d, dmin = d_rounded, dmin_rounded
        d_stored = d.to(torch.float16).contiguous()
        dmin_stored = dmin.to(torch.float16).contiguous()
    else:
        d_stored = d.to(scale_dtype).contiguous()
        dmin_stored = dmin.to(scale_dtype).contiguous()

    sm_values = (scales_6bit.to(torch.int32) << 6) | mins_6bit.to(torch.int32)
    sm_packed = torch.empty(out_features, n_blocks, 12, dtype=torch.uint8)
    for pair in range(4):
        i = pair * 2
        b = pair * 3
        first = sm_values[:, :, i]
        second = sm_values[:, :, i + 1]
        sm_packed[:, :, b] = first & 0xFF
        sm_packed[:, :, b + 1] = (first >> 8) | ((second & 0x0F) << 4)
        sm_packed[:, :, b + 2] = second >> 4

    flat_codes = codes.to(torch.uint8).reshape(out_features, in_features)
    quants_packed = flat_codes[:, 0::2] | (flat_codes[:, 1::2] << 4)

    return {
        "quants_packed": quants_packed.numpy(),
        "d": d_stored.numpy(),
        "dmin": dmin_stored.numpy(),
        "scales_mins_packed": sm_packed.numpy(),
        "shape": [out_features, in_features],
        "scale_dtype": str(effective_store_dtype).removeprefix("torch."),
        "refine_mode": refine_mode,
    }


def _quantize_one_layer_q4k(
    layer: nn.Linear,
    scale_dtype: torch.dtype = torch.float32,
    refine_mode: str = "legacy_exact",
    ddmin_store_dtype: torch.dtype | None = None,
) -> dict:
    """Return packed bytes without replacing or mutating the source weight."""
    return _encode_q4k(
        layer,
        scale_dtype=scale_dtype,
        refine_mode=refine_mode,
        ddmin_store_dtype=ddmin_store_dtype,
    )


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
    zero_pt = (maxq + 1) / 2                       # 2.0 for 2-bit

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
    codebooks = torch.zeros(out_features, n_groups, 4, dtype=torch.bfloat16)
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
        quantile_idxs = [0, g // 3, 2 * g // 3, g - 1]
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
                for j in range(4):
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
    cb_deq = cb_deq_flat.reshape(out_features, n_groups, 4).to(torch.bfloat16)

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
# Compressed storage (safetensors + manifest + residual)
# ---------------------------------------------------------------------------

def _save_packed_q4k(meta: dict, save_dir: str, model: nn.Module) -> int:
    """Save packed Q4_K weights, untouched residual tensors, and a manifest."""
    os.makedirs(save_dir, exist_ok=True)
    module_map = dict(model.named_modules())

    packed_tensors: dict[str, torch.Tensor] = {}
    layer_manifest: dict[str, dict] = {}
    total_packed_bytes = 0
    scale_dtypes = set()
    refine_modes = set()

    for name, data in tqdm(meta.items(), desc="Saving packed"):
        if name not in module_map or not isinstance(module_map[name], nn.Linear):
            raise RuntimeError(f"Quantized layer '{name}' is not an nn.Linear")
        prefix = f"{name}.{Q4K_FORMAT_NAME}"
        for key in Q4K_PACKED_KEYS:
            if key not in data:
                raise RuntimeError(f"Layer '{name}' is missing packed field '{key}'")
            tensor = torch.from_numpy(np.ascontiguousarray(data[key]))
            packed_tensors[f"{prefix}.{key}"] = tensor
            total_packed_bytes += tensor.numel() * tensor.element_size()

        module = module_map[name]
        scale_dtypes.add(data["scale_dtype"])
        refine_modes.add(data["refine_mode"])
        layer_manifest[name] = {
            "shape": [int(data["shape"][0]), int(data["shape"][1])],
            "has_bias": module.bias is not None,
            "scale_dtype": data["scale_dtype"],
            "refine_mode": data["refine_mode"],
        }

    if len(scale_dtypes) > 1:
        raise RuntimeError(f"Mixed Q4_K scale dtypes are not supported: {scale_dtypes}")
    if len(refine_modes) > 1:
        raise RuntimeError(f"Mixed Q4_K refine modes are not supported: {refine_modes}")

    shard_name = f"{Q4K_FORMAT_NAME}-00001-of-00001.safetensors"
    if packed_tensors:
        st.save_file(packed_tensors, os.path.join(save_dir, shard_name))
        logger.info(
            "Packed weights: %.1f MB (%d layers, %d tensors)",
            total_packed_bytes / 1e6,
            len(layer_manifest),
            len(packed_tensors),
        )

    quantized_weight_keys = {f"{name}.weight" for name in layer_manifest}
    residual_state: dict[str, torch.Tensor] = {}
    residual_bytes = 0
    for key, tensor in model.state_dict().items():
        if key in quantized_weight_keys:
            continue
        stored = tensor.detach().cpu().contiguous().clone()
        if "embed_tokens" in key or "lm_head" in key:
            stored = stored.to(torch.bfloat16)
        residual_state[key] = stored
        residual_bytes += stored.numel() * stored.element_size()

    max_residual_bytes = 6 * 1024**3
    if residual_bytes > max_residual_bytes:
        raise RuntimeError(
            f"Residual too large: {residual_bytes / 1e9:.2f} GB. "
            "Large projections were probably skipped and would defeat packed-only loading."
        )

    residual_name = "residual.safetensors"
    if residual_state:
        st.save_file(residual_state, os.path.join(save_dir, residual_name))
        logger.info(
            "Residual: %.1f MB (%d tensors)",
            residual_bytes / 1e6,
            len(residual_state),
        )

    manifest = {
        "format": Q4K_FORMAT_NAME,
        "format_version": 2,
        "block_size": QK_K,
        "sub_block_size": QK_K_SUB_SIZE,
        "output_scale_group": 1,
        "scale_dtype": next(iter(scale_dtypes), "float32"),
        "refine_mode": next(iter(refine_modes), "legacy_exact"),
        "weight_files": [shard_name],
        "residual_file": residual_name,
        "quantized_layers": layer_manifest,
        "residual_parameters": sorted(residual_state),
    }
    with open(os.path.join(save_dir, "q4k_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    total_bytes = total_packed_bytes + residual_bytes
    logger.info(
        "Total: %.1f MB (packed %.1f + residual %.1f)",
        total_bytes / 1e6,
        total_packed_bytes / 1e6,
        residual_bytes / 1e6,
    )
    return total_bytes


# ---------------------------------------------------------------------------
# Quantization driver
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    bits: int = 2,
    groupsize: int = DEFAULT_GROUPSIZE,
    save_compressed_dir: str | None = None,
    act_stats: dict | None = None,
    fmt: str = "q2_kmeans",
    q4k_scale_dtype: torch.dtype = torch.float32,
    q4k_refine_mode: str = "legacy_exact",
    q4k_ddmin_store_dtype: torch.dtype | None = None,
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

        if fmt == "q4_k":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k(
                layer,
                scale_dtype=q4k_scale_dtype,
                refine_mode=q4k_refine_mode,
                ddmin_store_dtype=q4k_ddmin_store_dtype,
            )
        else:
            layer_gs = _get_layer_groupsize(name, groupsize)
            if groupsize != -1 and layer.weight.shape[1] % layer_gs != 0:
                logger.warning("Skipping %s", name)
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
        if fmt != "q4_k":
            raise NotImplementedError(
                "The packed-only safetensors writer currently supports only --format q4_k"
            )
        _save_packed_q4k(meta, save_compressed_dir, model)

    if skipped_small:
        logger.info("Skipped %d small/SSM layers", skipped_small)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training quantization for causal LMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--groupsize", type=int, default=DEFAULT_GROUPSIZE)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save", default=None,
                        help="Directory to save quantized model")
    parser.add_argument("--calibration-cache", default=None)
    parser.add_argument("--format", default="q2_kmeans",
                        choices=["q2_kmeans", "q4_k"])
    parser.add_argument(
        "--q4k-scale-dtype",
        default="float32",
        choices=["float16", "float32"],
        help=(
            "Storage dtype for per-superblock d/dmin. float32 costs about "
            "0.125 extra bits/weight and avoids scale-rounding loss."
        ),
    )
    parser.add_argument(
        "--q4k-refine-mode",
        default="legacy_exact",
        choices=["legacy_exact", "alternating", "none"],
        help=(
            "legacy_exact reproduces the historical dequantized-BF16 Q4_K "
            "quantizer; alternating reassigns codes after each LS solve."
        ),
    )
    parser.add_argument(
        "--q4k-ddmin-fp16",
        action="store_true",
        default=False,
        help=(
            "Store per-superblock d/dmin at fp16 precision, using LS "
            "re-optimization to absorb the rounding error.  Saves ~0.125 GB."
        ),
    )
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
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True)

    act_stats = None
    if args.format == "q2_kmeans":
        if args.calibration_cache and os.path.exists(args.calibration_cache):
            logger.info("Loading cached calibration: %s", args.calibration_cache)
            cached = np.load(args.calibration_cache, allow_pickle=True)
            act_stats = {k: torch.from_numpy(cached[k]) for k in cached.files}
            model.to("cuda")
        elif torch.cuda.is_available():
            logger.info("Collecting calibration activation stats on GPU")
            torch.cuda.reset_peak_memory_stats()
            model.to("cuda")
            act_stats = _collect_input_stats(model, args.model,
                                             nsamples=16, seqlen=1024)
            if args.calibration_cache and act_stats:
                os.makedirs(os.path.dirname(args.calibration_cache) or ".", exist_ok=True)
                np.savez(args.calibration_cache,
                         **{k: v.numpy() for k, v in act_stats.items()})

    logger.info("Quantizing  format=%s  bits=%d", args.format,
                4 if args.format == "q4_k" else args.bits)
    q4k_scale_dtype = getattr(torch, args.q4k_scale_dtype)
    q4k_ddmin_store_dtype = torch.float16 if args.q4k_ddmin_fp16 else None
    meta = quantize_model(
        model,
        bits=args.bits,
        groupsize=args.groupsize,
        save_compressed_dir=args.save,
        act_stats=act_stats,
        fmt=args.format,
        q4k_scale_dtype=q4k_scale_dtype,
        q4k_refine_mode=args.q4k_refine_mode,
        q4k_ddmin_store_dtype=q4k_ddmin_store_dtype,
    )
    logger.info("Quantized %d linear layers.", len(meta))

    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    if peak_vram > VRAM_LIMIT_MB:
        logger.error("Peak VRAM %.1f MB exceeds limit %d MB.", peak_vram, VRAM_LIMIT_MB)
        raise SystemExit(1)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.config.save_pretrained(args.save)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)
        logger.info("Saved config + tokenizer to '%s'", args.save)


if __name__ == "__main__":
    main()
