#!/usr/bin/env python3
"""
Post-training quantization for AutoModelForCausalLM.
Supports multiple formats: q2_kmeans (default), q4_k (GGML-style 4-bit blocks).

CPU-based, vectorised quantization.  No GPU sync per layer.
"""

import argparse
import hashlib
import json
import logging
import os
import struct

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

def _quantize_one_layer_q4k(layer: nn.Linear, quants_delta_K: int = 256, sm_share_K: int = 128, d_share_K: int = 8, act_stats: torch.Tensor | None = None, quants_delta_bits: int = 2, skip_delta_sm: bool = False, subblock_delta_bits: int = 0, ref_bits: int = 4, sparse_delta: bool = False, sparse_threshold: float = 0.0, quants_no_delta: bool = False) -> dict:
    W = layer.weight.data.float()
    out_features, in_features = W.shape

    assert in_features % QK_K == 0, (
        f"in_features ({in_features}) must be divisible by QK_K ({QK_K})")
    n_blocks = in_features // QK_K

    W_r = W.reshape(out_features, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE)
    # shape: (out, n_blocks, 8, 32)

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

    W_flat = W.reshape(out_features, n_blocks, QK_K)

    act_w = None
    if act_stats is not None:
        act_w = act_stats.to(W.device).view(1, n_blocks, QK_K)
        act_w = act_w / act_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    for _ in range(3):
        s = sc_norm.unsqueeze(-1) * q.float()
        s_flat = s.flatten(start_dim=2)
        m = m_norm.unsqueeze(-1).expand(-1, -1, -1, QK_K_SUB_SIZE)
        m_flat = m.flatten(start_dim=2)

        s_sq = (s_flat * s_flat * act_w).sum(dim=-1) if act_w is not None else (s_flat * s_flat).sum(dim=-1)
        m_sq = (m_flat * m_flat * act_w).sum(dim=-1) if act_w is not None else (m_flat * m_flat).sum(dim=-1)
        sm = (s_flat * m_flat * act_w).sum(dim=-1) if act_w is not None else (s_flat * m_flat).sum(dim=-1)
        ws = (W_flat * s_flat * act_w).sum(dim=-1) if act_w is not None else (W_flat * s_flat).sum(dim=-1)
        wm = (W_flat * m_flat * act_w).sum(dim=-1) if act_w is not None else (W_flat * m_flat).sum(dim=-1)

        det = s_sq * m_sq - sm * sm
        det[det.abs() < 1e-12] = 1e-12
        d_new = (ws * m_sq - wm * sm) / det
        dmin_new = (ws * sm - wm * s_sq) / det

        d = d_new.clamp(min=1e-8)
        dmin = dmin_new.abs().clamp(min=1e-8)

    # Share d/dmin across K=4 output channels with 4-bit per-channel scale factors
    K = d_share_K
    out_pad = ((out_features + K - 1) // K) * K
    n_groups = out_pad // K

    d_storage = d.clone()
    dmin_storage = dmin.clone()

    if out_features < out_pad:
        d_storage = torch.nn.functional.pad(d_storage, (0, 0, 0, out_pad - out_features))
        dmin_storage = torch.nn.functional.pad(dmin_storage, (0, 0, 0, out_pad - out_features))

    d_grp = d_storage.reshape(n_groups, K, n_blocks)
    dmin_grp = dmin_storage.reshape(n_groups, K, n_blocks)

    shared_d = d_grp.amax(dim=1)
    shared_dmin = dmin_grp.amax(dim=1)

    sf_d = (d_grp / shared_d.unsqueeze(1).clamp(min=1e-8)).clamp(0.0, 1.0)
    sf_d_4bit = torch.clamp(torch.round(sf_d * 15.0), 1, 15).to(torch.uint8)
    sf_dm = (dmin_grp / shared_dmin.unsqueeze(1).clamp(min=1e-8)).clamp(0.0, 1.0)
    sf_dm_4bit = torch.clamp(torch.round(sf_dm * 15.0), 1, 15).to(torch.uint8)

    # Delta-encode shared_d/dmin across superblocks (storage-only compression)
    d_ref = shared_d.numpy().astype(np.float32)
    d_ref = np.maximum(d_ref, 1e-8)
    d_log6 = np.clip(np.round(np.log2(d_ref) * 6.0 + 32.0), 0, 63).astype(np.uint8)

    dm_ref = shared_dmin.numpy().astype(np.float32)
    dm_ref = np.maximum(dm_ref, 1e-8)
    dm_log6 = np.clip(np.round(np.log2(dm_ref) * 6.0 + 32.0), 0, 63).astype(np.uint8)

    base_d = d_log6[:, 0:1]
    base_dm = dm_log6[:, 0:1]
    delta_d = (d_log6[:, 1:].astype(np.int16) - d_log6[:, :-1].astype(np.int16)).clip(-7, 7).astype(np.int8)
    delta_dm = (dm_log6[:, 1:].astype(np.int16) - dm_log6[:, :-1].astype(np.int16)).clip(-7, 7).astype(np.int8)

    # Pack base: 2 × 6-bit → 2 bytes
    base_packed = (base_d.astype(np.uint8) & 0x3F) | ((base_dm.astype(np.uint8) & 0x3F) << 6)

    # Pack deltas: 2 × 4-bit signed → 1 byte each (store as nibbles)
    delta_d_u4 = (delta_d + 8).clip(0, 15).astype(np.uint8)
    delta_dm_u4 = (delta_dm + 8).clip(0, 15).astype(np.uint8)
    delta_packed = (delta_d_u4 & 0x0F) | ((delta_dm_u4 & 0x0F) << 4)

    eff_scale = d.unsqueeze(-1) * sc_norm
    eff_scale[eff_scale < 1e-8] = 1e-8
    eff_offset = dmin.unsqueeze(-1) * m_norm

    W_q_r = eff_scale.unsqueeze(-1) * q.float() - eff_offset.unsqueeze(-1)
    W_q = W_q_r.reshape(out_features, in_features)
    layer.weight.data = W_q.to(layer.weight.dtype)

    quants_flat = q.reshape(out_features, in_features)

    # Share scales/mins across K_sm output channels with 2-bit multiplicative deltas
    K_sm = min(sm_share_K, max(out_features, 2))
    out_pad_sm = ((out_features + K_sm - 1) // K_sm) * K_sm
    n_groups_sm = out_pad_sm // K_sm

    sc_np = sc_6bit.numpy()
    m_np = m_6bit.numpy()

    if out_features < out_pad_sm:
        sc_pad = np.pad(sc_np, ((0, out_pad_sm - out_features), (0, 0), (0, 0)), mode='constant', constant_values=0)
        m_pad = np.pad(m_np, ((0, out_pad_sm - out_features), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        sc_pad = sc_np
        m_pad = m_np

    sc_grp = sc_pad.reshape(n_groups_sm, K_sm, n_blocks, 8)
    m_grp = m_pad.reshape(n_groups_sm, K_sm, n_blocks, 8)

    shared_sc = sc_grp.max(axis=1)
    shared_m = m_grp.max(axis=1)

    delta_sc = np.clip(np.round(
        sc_grp.astype(np.float32) / np.maximum(shared_sc.astype(np.float32), 1.0)[:, np.newaxis, :, :] * 3.0
    ), 0, 3).astype(np.uint8)
    delta_m = np.clip(np.round(
        m_grp.astype(np.float32) / np.maximum(shared_m.astype(np.float32), 1.0)[:, np.newaxis, :, :] * 3.0
    ), 0, 3).astype(np.uint8)

    delta_sc = delta_sc.reshape(out_pad_sm, n_blocks, 8)[:out_features]
    delta_m = delta_m.reshape(out_pad_sm, n_blocks, 8)[:out_features]

    delta_sm_packed = _pack_q4k_sm_deltas_block_delta(delta_sc, delta_m)

    # Inter-channel quants delta compression: store 1 ref + (Kq-1) delta channels
    Kq = min(quants_delta_K, out_features)
    quants_np = quants_flat.numpy()

    if Kq >= 2:
        out_pad_q = ((out_features + Kq - 1) // Kq) * Kq
        if out_features < out_pad_q:
            q_pad = np.pad(quants_np, ((0, out_pad_q - out_features), (0, 0)), mode='constant', constant_values=0)
        else:
            q_pad = quants_np
        n_groups_q = out_pad_q // Kq
        q_grp = q_pad.reshape(n_groups_q, Kq, -1)
        ref_indices = _select_best_reference(q_grp, n_blocks, Kq, out_features)
        for g in range(n_groups_q):
            ri = int(ref_indices[g])
            if ri != 0:
                tmp = q_grp[g, 0].copy()
                q_grp[g, 0] = q_grp[g, ri]
                q_grp[g, ri] = tmp
        q_ref = q_grp[:, 0, :]
        if ref_bits == 2:
            quants_ref_packed = _pack_2bit(q_ref)
        elif ref_bits == 1:
            quants_ref_packed = _pack_1bit(q_ref)
        else:
            quants_ref_packed = _pack_4bit(q_ref)
        if quants_no_delta:
            quants_kw = dict(quants_ref=quants_ref_packed, Kq=Kq, ref_bits=ref_bits, ref_indices=ref_indices, quants_no_delta=True)
        else:
            q_tgt = q_grp[:, 1:, :].reshape(-1, q_ref.shape[1])
            quants_delta_packed = _pack_quants_delta(q_ref, q_tgt, Kq - 1, quants_delta_bits, subblock_delta_bits=subblock_delta_bits, sparse_delta=sparse_delta, sparse_threshold=sparse_threshold)
            quants_kw = dict(quants_ref=quants_ref_packed, Kq=Kq, delta_bits=quants_delta_bits, subblock_delta_bits=subblock_delta_bits, ref_bits=ref_bits, ref_indices=ref_indices)
            if isinstance(quants_delta_packed, dict):
                quants_kw["quants_delta"] = quants_delta_packed["data"]
                quants_kw["delta_format"] = quants_delta_packed["format"]
                if quants_delta_packed["format"] == "sparse":
                    quants_kw["delta_mask"] = quants_delta_packed["delta_mask"]
                    quants_kw["delta_data"] = quants_delta_packed["delta_data"]
                    quants_kw["delta_nz_counts"] = quants_delta_packed["nz_counts"]
                    if quants_delta_packed.get("mask_packed_2ch"):
                        quants_kw["mask_packed_2ch"] = True
                    if quants_delta_packed.get("mask_format") == "fullmask_bitpacked":
                        quants_kw["delta_fullmask"] = quants_delta_packed["delta_fullmask"]
                        quants_kw["mask_format"] = "fullmask_bitpacked"
                quants_kw["sparse_delta"] = True
            else:
                quants_kw["quants_delta"] = quants_delta_packed
                quants_kw["sparse_delta"] = False
    else:
        quants_kw = dict(quants=quants_np)

    result = {
        **quants_kw,
        "base_packed":    base_packed,
        "delta_packed":   delta_packed,
        "scales_mins_shared": shared_sc,
        "mins_shared":    shared_m,
        "format":         "q4_k",
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
        "K":              K,
        "K_sm":           K_sm,
    }
    if not skip_delta_sm:
        result["delta_sm"] = delta_sm_packed
        result["delta_sm_encoded"] = True
    return result


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


def _pack_1bit(codes: np.ndarray) -> np.ndarray:
    """Pack 8 × 1-bit values into one uint8."""
    out, inp = codes.shape
    assert inp % 8 == 0
    codes = codes.reshape(out, inp // 8, 8)
    packed = (codes[..., 0]
              | (codes[..., 1] << 1)
              | (codes[..., 2] << 2)
              | (codes[..., 3] << 3)
              | (codes[..., 4] << 4)
              | (codes[..., 5] << 5)
              | (codes[..., 6] << 6)
              | (codes[..., 7] << 7))
    return packed.astype(np.uint8)


def _pack_4bit(codes: np.ndarray) -> np.ndarray:
    """Pack 2 × 4-bit values into one uint8."""
    out, inp = codes.shape
    assert inp % 2 == 0
    codes = codes.reshape(out, inp // 2, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4))
    return packed.astype(np.uint8)


def _pack_q4k_sm(scales_6bit: np.ndarray, mins_6bit: np.ndarray) -> np.ndarray:
    """Pack 8 × (6-bit scale + 6-bit min) = 96 bits → 12 bytes per superblock.
    Saves 4 bytes/superblock vs separate uint8 arrays."""
    out, nb, nsub = scales_6bit.shape
    assert nsub == 8
    sm = (scales_6bit.astype(np.uint16) << 6) | mins_6bit.astype(np.uint16)
    packed = np.zeros((out, nb, 12), dtype=np.uint8)
    for p in range(4):
        i = p * 2
        s0 = sm[:, :, i]
        s1 = sm[:, :, i + 1]
        b = p * 3
        packed[:, :, b] = s0 & 0xFF
        packed[:, :, b + 1] = (s0 >> 8) | ((s1 & 0x0F) << 4)
        packed[:, :, b + 2] = s1 >> 4
    return packed


def _pack_quants_delta(ref_quants: np.ndarray, tgt_quants: np.ndarray, n_tgt: int, delta_bits: int = 2, subblock_delta_bits: int = 0, sparse_delta: bool = False, sparse_threshold: float = 0.0):
    """Pack delta between ref and target quants.
    delta_bits=1: 32 bytes/sb per-weight (0,+1).
    delta_bits=2: 64 bytes/sb per-weight (-1,0,1,2).
    subblock_delta_bits>0: per-sub-block encoding, subblock_delta_bits bytes/sb per 8 sub-blocks.
    sparse_delta=True: skip all-zero superblocks, storing mask + non-zero data only.
    sparse_threshold: fraction of max sub_q to treat as zero (increases sparsity)."""
    out, inp = ref_quants.shape
    n_blocks = inp // QK_K
    ref_sb = ref_quants.reshape(out, n_blocks, QK_K).astype(np.int16)
    tgt_sb = tgt_quants.reshape(n_tgt, out, n_blocks, QK_K).astype(np.int16)

    if subblock_delta_bits > 0:
        packed = _pack_quants_delta_subblock(ref_sb, tgt_sb, n_tgt, out, n_blocks, subblock_delta_bits, delta_bits, sparse_threshold)
        if sparse_delta:
            result = _pack_quants_delta_subblock_sparse(packed, n_tgt, out, n_blocks)
            return result
        return packed

    bytes_per_sb = 32 if delta_bits == 1 else 64
    packed = np.zeros((n_tgt, out, n_blocks, bytes_per_sb), dtype=np.uint8)
    for t in range(n_tgt):
        diff = tgt_sb[t] - ref_sb
        if delta_bits == 1:
            delta = (diff > 0).astype(np.uint8)
            for o in range(out):
                for b in range(n_blocks):
                    packed[t, o, b] = _pack_1bit(delta[o, b].reshape(1, QK_K)).reshape(bytes_per_sb)
        else:
            delta = np.clip(diff + 1, 0, 3).astype(np.uint8)
            for o in range(out):
                for b in range(n_blocks):
                    packed[t, o, b] = _pack_2bit(delta[o, b].reshape(1, QK_K)).reshape(bytes_per_sb)
    return packed


def _pack_quants_delta_subblock_sparse(dense_packed: np.ndarray, n_tgt: int, out: int, n_blocks: int) -> dict:
    """Convert dense sub-block delta to sparse: skip all-zero superblocks.
    Returns dict with 'format'='dense' or 'sparse', plus the relevant arrays.
    Bitpacks mask: 1-bit fullness bitmap + per-channel masks only for partial channels."""
    bytes_per_sb = dense_packed.shape[-1]
    n_channels = n_tgt * out
    mask_bytes = (n_blocks + 7) // 8
    pack2 = (n_blocks <= 4)

    dense_flat = dense_packed.reshape(n_channels, n_blocks, bytes_per_sb)
    nz_sb_per_ch = np.zeros(n_channels, dtype=np.int32)
    zero_rows_all = np.zeros(n_channels, dtype=np.bool_)
    nz_list = []

    for c in range(n_channels):
        ch_data = dense_flat[c]
        zero_rows = (ch_data == 0).all(axis=1)
        nz = np.where(~zero_rows)[0]
        if len(nz) > 0:
            nz_sb_per_ch[c] = len(nz)
            zero_rows_all[c] = False
            for sb_idx in nz:
                nz_list.append(ch_data[sb_idx].copy())
        else:
            zero_rows_all[c] = True

    total_nz = int(nz_sb_per_ch.sum())
    full_channels = (nz_sb_per_ch == n_blocks) & (~zero_rows_all)
    n_full = int(full_channels.sum())

    fullmask_bytes = (n_channels + 7) // 8
    fullmask = np.zeros(fullmask_bytes, dtype=np.uint8)
    for c in range(n_channels):
        if full_channels[c]:
            fullmask[c // 8] |= (1 << (c % 8))

    is_partial = (~full_channels) & (~zero_rows_all)
    partial_idxs = np.where(is_partial)[0]
    n_partial = len(partial_idxs)
    partial_idx_map = {int(pidx): i for i, pidx in enumerate(partial_idxs)}

    if pack2:
        partial_packed_channels = (n_partial + 1) // 2
        partial_masks = np.zeros(partial_packed_channels * mask_bytes, dtype=np.uint8)
        for pi, c in enumerate(partial_idxs):
            ch_data = dense_flat[c]
            zero_rows = (ch_data == 0).all(axis=1)
            sb_bits = 0
            for sb_idx in range(n_blocks):
                if not zero_rows[sb_idx]:
                    sb_bits |= (1 << sb_idx)
            byte_idx = pi // 2
            if pi % 2 == 0:
                partial_masks[byte_idx] |= (sb_bits & 0x0F)
            else:
                partial_masks[byte_idx] |= ((sb_bits & 0x0F) << 4)
        masks_out = partial_masks.reshape(partial_packed_channels, mask_bytes)
    else:
        partial_masks = np.zeros((n_partial, mask_bytes), dtype=np.uint8)
        for pi, c in enumerate(partial_idxs):
            ch_data = dense_flat[c]
            zero_rows = (ch_data == 0).all(axis=1)
            for sb_idx in range(n_blocks):
                if not zero_rows[sb_idx]:
                    byte_idx = sb_idx // 8
                    bit_idx = sb_idx % 8
                    partial_masks[pi, byte_idx] |= (1 << bit_idx)
        masks_out = partial_masks

    fullmask_size = fullmask_bytes
    sparse_data_size = total_nz * bytes_per_sb
    dense_size = dense_packed.nbytes

    old_mask_size = ((n_channels + 1) // 2) * mask_bytes if pack2 else n_channels * mask_bytes
    partial_mask_size = masks_out.nbytes if n_partial > 0 else 0
    new_mask_size = fullmask_size + partial_mask_size

    use_fullmask = (new_mask_size < old_mask_size)
    if not use_fullmask:
        if sparse_data_size + old_mask_size >= dense_size:
            return dict(format='dense', data=dense_packed)
        if pack2:
            masks = np.zeros(((n_channels + 1) // 2) * mask_bytes, dtype=np.uint8)
            for c in range(n_channels):
                if zero_rows_all[c]:
                    continue
                ch_data = dense_flat[c]
                zero_rows = (ch_data == 0).all(axis=1)
                sb_bits = 0
                for sb_idx in range(n_blocks):
                    if not zero_rows[sb_idx]:
                        sb_bits |= (1 << sb_idx)
                byte_idx = c // 2
                if c % 2 == 0:
                    masks[byte_idx] |= (sb_bits & 0x0F)
                else:
                    masks[byte_idx] |= ((sb_bits & 0x0F) << 4)
            masks_out = masks.reshape(((n_channels + 1) // 2), mask_bytes)
        else:
            masks = np.zeros((n_tgt, out, mask_bytes), dtype=np.uint8)
            masks_flat = masks.reshape(n_channels, mask_bytes)
            for c in range(n_channels):
                if zero_rows_all[c]:
                    continue
                ch_data = dense_flat[c]
                zero_rows = (ch_data == 0).all(axis=1)
                for sb_idx in range(n_blocks):
                    if not zero_rows[sb_idx]:
                        byte_idx = sb_idx // 8
                        bit_idx = sb_idx % 8
                        masks_flat[c, byte_idx] |= (1 << bit_idx)
            masks_out = masks

    total_sparse = sparse_data_size + masks_out.nbytes + (fullmask_size if use_fullmask else 0)
    if total_sparse >= dense_size:
        return dict(format='dense', data=dense_packed)

    if nz_list:
        nz_data = np.concatenate([b.reshape(1, -1) for b in nz_list], axis=0).reshape(-1)
    else:
        nz_data = np.zeros(0, dtype=np.uint8)

    result = dict(format='sparse', data=dense_packed, delta_mask=masks_out,
                  delta_data=nz_data,
                  nz_counts=nz_sb_per_ch.reshape(n_tgt, out))
    if use_fullmask:
        result["delta_fullmask"] = fullmask
        result["delta_fullmask_n"] = n_blocks
        result["mask_format"] = "fullmask_bitpacked"
    elif pack2:
        result["mask_packed_2ch"] = True
    return result


def _select_best_reference(q_grp: np.ndarray, n_blocks: int, Kq: int, out_features: int) -> np.ndarray:
    """Select the most representative channel as reference per group.
    Uses sub-block mean quants to find channel closest to group centroid.
    Returns ref_indices of shape (n_groups_q,) as uint16."""
    n_groups_q = q_grp.shape[0]
    in_features = q_grp.shape[2]
    q_sb = q_grp.reshape(n_groups_q, Kq, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE)
    q_sb_means = q_sb.mean(axis=-1).reshape(n_groups_q, Kq, -1)
    centroid = q_sb_means.mean(axis=1)
    dists = ((q_sb_means - centroid[:, np.newaxis, :]) ** 2).sum(axis=2)
    ref_indices = np.zeros(n_groups_q, dtype=np.uint16)
    for g in range(n_groups_q):
        n_real = min(Kq, out_features - g * Kq)
        if n_real <= 0:
            n_real = Kq
        dists_g = dists[g, :n_real]
        ref_indices[g] = int(dists_g.argmin())
    return ref_indices


def _pack_quants_delta_subblock(ref_sb, tgt_sb, n_tgt, out, n_blocks, subblock_delta_bits, delta_bits, sparse_threshold=0.0):
    """Encode quants delta at sub-block (32-weight) granularity.
    subblock_delta_bits: bits per sub-block (1, 2, 3, or 4). Returns (n_tgt, out, n_blocks, bytes_per_sb).
    sparse_threshold: fraction of max_val below which sub_q is zeroed (e.g. 0.25 = skip weak signals)."""
    max_val = (1 << subblock_delta_bits) - 1
    min_val_incl = max_val * sparse_threshold
    bytes_per_sb = subblock_delta_bits
    packed = np.zeros((n_tgt, out, n_blocks, bytes_per_sb), dtype=np.uint8)

    for t in range(n_tgt):
        diff = tgt_sb[t].astype(np.int16) - ref_sb.astype(np.int16)
        diff_sb = diff.reshape(out, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE)

        if delta_bits == 1:
            delta = (diff_sb > 0).astype(np.float32)
        else:
            delta = np.clip(diff_sb.astype(np.float32) + 1, 0, 3) / 3.0

        mean_delta = delta.mean(axis=-1)
        sub_q = np.clip(np.round(mean_delta * max_val), 0, max_val).astype(np.uint8)
        if sparse_threshold > 0:
            sub_q[sub_q.astype(np.float32) <= min_val_incl] = 0

        for o in range(out):
            for b in range(n_blocks):
                vals = sub_q[o, b]  # (8,) uint8
                if subblock_delta_bits == 1:
                    packed[t, o, b, 0] = (vals[0] | (vals[1] << 1) | (vals[2] << 2) | (vals[3] << 3)
                                          | (vals[4] << 4) | (vals[5] << 5) | (vals[6] << 6) | (vals[7] << 7))
                elif subblock_delta_bits == 2:
                    packed[t, o, b, 0] = (vals[0] | (vals[1] << 2) | (vals[2] << 4) | (vals[3] << 6))
                    packed[t, o, b, 1] = (vals[4] | (vals[5] << 2) | (vals[6] << 4) | (vals[7] << 6))
                elif subblock_delta_bits == 3:
                    packed[t, o, b, 0] = (vals[0] | (vals[1] << 3))
                    packed[t, o, b, 0] |= ((vals[2] & 0x3) << 6)
                    packed[t, o, b, 1] = ((vals[2] >> 2) | (vals[3] << 1) | (vals[4] << 4) | ((vals[5] & 0x1) << 7))
                    packed[t, o, b, 2] = ((vals[5] >> 1) | (vals[6] << 2) | (vals[7] << 5))
                elif subblock_delta_bits == 4:
                    packed[t, o, b, 0] = (vals[0] | (vals[1] << 4))
                    packed[t, o, b, 1] = (vals[2] | (vals[3] << 4))
                    packed[t, o, b, 2] = (vals[4] | (vals[5] << 4))
                    packed[t, o, b, 3] = (vals[6] | (vals[7] << 4))
    return packed


def _pack_q4k_sm_joint(shared_sc: np.ndarray, shared_m: np.ndarray, bits: int = 5) -> np.ndarray:
    """Pack 8 x (bits+bits joint scale+min) → bytes per superblock.
    bits=5: 10 bytes/sb, bits=4: 8 bytes/sb."""
    ng, nb, nsub = shared_sc.shape
    assert nsub == 8
    maxv = (1 << bits) - 1
    sc_q = np.clip(np.round(shared_sc.astype(np.float32) / (2 ** (6 - bits))), 0, maxv).astype(np.uint16)
    m_q = np.clip(np.round(shared_m.astype(np.float32) / (2 ** (6 - bits))), 0, maxv).astype(np.uint16)
    joint = (sc_q << bits) | m_q
    bpw = bits * 2
    bytes_per_sb = bpw
    packed = np.zeros((ng, nb, bytes_per_sb), dtype=np.uint8)
    if bits == 5:
        for p in range(2):
            i = p * 4
            v0 = joint[:, :, i]; v1 = joint[:, :, i + 1]
            v2 = joint[:, :, i + 2]; v3 = joint[:, :, i + 3]
            b = p * 5
            packed[:, :, b] = v0 & 0xFF
            packed[:, :, b + 1] = ((v0 >> 8) & 0x3) | ((v1 & 0x3F) << 2)
            packed[:, :, b + 2] = ((v1 >> 6) & 0xF) | ((v2 & 0xF) << 4)
            packed[:, :, b + 3] = ((v2 >> 4) & 0x3F) | ((v3 & 0x3) << 6)
            packed[:, :, b + 4] = (v3 >> 2) & 0xFF
    else:
        packed = joint.astype(np.uint8)
    return packed


def _pack_q4k_sm_deltas(delta_sc: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    """Pack 8 sub-blocks × per-sub-block (2-bit sc delta + 2-bit m delta) → 4 bytes/superblock."""
    out, nb, nsub = delta_sc.shape
    assert nsub == 8
    delta_sc = delta_sc.astype(np.uint8) & 0x3
    delta_m = delta_m.astype(np.uint8) & 0x3
    packed = np.zeros((out, nb, 4), dtype=np.uint8)
    for p in range(4):
        i = p * 2
        lo = (delta_sc[:, :, i] & 0x3) | ((delta_m[:, :, i] & 0x3) << 2)
        hi = (delta_sc[:, :, i + 1] & 0x3) | ((delta_m[:, :, i + 1] & 0x3) << 2)
        packed[:, :, p] = lo | (hi << 4)
    return packed


def _pack_q4k_sm_deltas_block_delta(delta_sc: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    """Delta-encode packed delta_sm across superblocks: 4+2*(nb-1) bytes/channel vs 4*nb."""
    out, nb, nsub = delta_sc.shape
    assert nsub == 8
    if nb <= 1:
        return _pack_q4k_sm_deltas(delta_sc, delta_m)

    packed = _pack_q4k_sm_deltas(delta_sc, delta_m)  # (out, nb, 4)
    ref = packed[:, 0:1, :].copy()  # (out, 1, 4) — first block reference
    result = np.zeros((out, nb, 2), dtype=np.uint8)
    result[:, 0, :2] = ref[:, 0, :2]

    for b in range(1, nb):
        for p in range(4):
            cur = packed[:, b, p].astype(np.int16)
            prv = packed[:, b - 1, p].astype(np.int16)
            delta = np.clip(cur - prv, -8, 7).astype(np.int8)
            nibble = (delta + 8).clip(0, 15).astype(np.uint8)
            if p < 2:
                result[:, b, 0] |= nibble << (p * 4)
            else:
                result[:, b, 1] |= nibble << ((p - 2) * 4)
    return result


def _pack_fp16_to_log8(arr: np.ndarray) -> np.ndarray:
    """Pack fp16 d/dmin into uint8 log-scale. ~6 MB savings per model."""
    arr = arr.astype(np.float32)
    arr = np.maximum(arr, 1e-8)
    log2 = np.log2(arr)
    encoded = np.clip(np.round(log2 * 12.0 + 128.0), 0, 255).astype(np.uint8)
    return encoded


def _pack_layer_data(data: dict, bits: int, fmt: str) -> dict:
    """Pack a single layer's data into a dict ready for np.savez."""
    if data.get("format") == "q4_k":
        if "quants_ref" in data:
            q_out = data["quants_ref"]
            q_delta = data.get("quants_delta")
        else:
            quants = data["quants"]
            q_out = _pack_4bit(quants)
            q_delta = None
        if "scales_mins_shared" in data:
            sm_packed = _pack_q4k_sm_joint(data["scales_mins_shared"], data["mins_shared"], bits=4)
        else:
            sm_packed = _pack_q4k_sm(data["scales_6bit"], data["mins_6bit"])
        if "base_packed" in data:
            save_kw = dict(
                quants=q_out,
                base_packed=data["base_packed"],
                delta_packed=data["delta_packed"],
                K=data.get("K", 4),
                scales_mins_packed=sm_packed,
                format=data["format"])
            if q_delta is not None:
                if data.get("sparse_delta") and data.get("delta_format") == "sparse":
                    save_kw["delta_mask"] = data["delta_mask"]
                    save_kw["delta_data"] = data["delta_data"]
                    save_kw["sparse_delta"] = True
                    save_kw["delta_format"] = "sparse"
                    if data.get("mask_packed_2ch"):
                        save_kw["mask_packed_2ch"] = True
                    if data.get("mask_format") == "fullmask_bitpacked":
                        save_kw["delta_fullmask"] = data["delta_fullmask"]
                        save_kw["mask_format"] = "fullmask_bitpacked"
                else:
                    save_kw["quants_delta"] = q_delta
                save_kw["Kq"] = data.get("Kq", 2)
                save_kw["delta_bits"] = data.get("delta_bits", 2)
                save_kw["ref_bits"] = data.get("ref_bits", 4)
                if data.get("subblock_delta_bits", 0) > 0:
                    save_kw["subblock_delta_bits"] = data["subblock_delta_bits"]
                if "ref_indices" in data:
                    save_kw["ref_indices"] = data["ref_indices"]
            elif data.get("quants_no_delta"):
                save_kw["Kq"] = data.get("Kq", 2)
                save_kw["ref_bits"] = data.get("ref_bits", 4)
                save_kw["quants_no_delta"] = True
                if "ref_indices" in data:
                    save_kw["ref_indices"] = data["ref_indices"]
            if "delta_sm" in data:
                save_kw["delta_sm"] = data["delta_sm"]
                save_kw["K_sm"] = data.get("K_sm", 4)
                if data.get("delta_sm_encoded"):
                    save_kw["delta_sm_encoded"] = True
            return save_kw
        else:
            d8 = _pack_fp16_to_log8(data["d"])
            dm8 = _pack_fp16_to_log8(data["dmin"])
            return dict(quants=q_out, d8=d8, dm8=dm8, scales_mins_packed=sm_packed, format=data["format"])
    elif bits == 4 and data["shape"][1] % 2 == 0:
        codes_out = _pack_4bit(data["codes"])
        return dict(codes=codes_out, codebook_q=data.get("codebook_q"),
                    cb_min=data.get("cb_min"), cb_max=data.get("cb_max"),
                    codebook_dtype=data.get("codebook_dtype", "q4"))
    elif bits == 2 and data["shape"][1] % 4 == 0:
        codes_out = _pack_2bit(data["codes"])
        if "codebook_q" in data:
            return dict(codes=codes_out, codebook_q=data["codebook_q"],
                        cb_min=data["cb_min"], cb_max=data["cb_max"],
                        codebook_dtype=data["codebook_dtype"])
        elif "codebook" in data:
            return dict(codes=codes_out, codebook=data["codebook"],
                        codebook_dtype=data["codebook_dtype"])
        else:
            return dict(codes=codes_out, scales=data["scales"],
                        scale_dtype=data["scale_dtype"], zero_pt=data["zero_pt"])
    else:
        codes_out = data["codes"]
        if "codebook_q" in data:
            return dict(codes=codes_out, codebook_q=data["codebook_q"],
                        cb_min=data["cb_min"], cb_max=data["cb_max"],
                        codebook_dtype=data["codebook_dtype"])
        else:
            return dict(codes=codes_out, scales=data["scales"],
                        scale_dtype=data["scale_dtype"], zero_pt=data["zero_pt"])


def _serialize_compact(meta: dict, packed_layers: list) -> bytes:
    """Compact binary format V6: schema-based + all-array dedup + compact layer encoding."""
    B = bytearray()
    B.extend(b'AQ06')
    B.extend(struct.pack('<I', 0))

    import re
    _LAYER_TYPES = {'q_proj': 0, 'k_proj': 1, 'v_proj': 2, 'o_proj': 3,
                    'gate_proj': 4, 'up_proj': 5, 'down_proj': 6}
    _LAYER_RE = re.compile(r'model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)')
    _EMBED_RE = re.compile(r'model\.embed_tokens')
    _LMHEAD_RE = re.compile(r'^lm_head$')

    def _encode_name(name):
        m = _LAYER_RE.match(name)
        if m:
            return 0, int(m.group(1)), _LAYER_TYPES.get(m.group(2), 255)
        if _EMBED_RE.match(name):
            return 1, 0, 7
        if _LMHEAD_RE.match(name):
            return 2, 0, 8
        return 3, 0, 255

    dedup_tables = {}
    table_data = {}
    def _dedup(dtype, raw_bytes):
        h = raw_bytes
        if dtype not in dedup_tables:
            dedup_tables[dtype] = []
            table_data[dtype] = {}
        if h not in table_data[dtype]:
            table_data[dtype][h] = len(dedup_tables[dtype])
            dedup_tables[dtype].append(raw_bytes)
        return table_data[dtype][h]

    quants_table = []
    quants_to_idx = {}
    layer_quants_idx = {}
    layer_dedup = {}

    for name, shape, arrays in packed_layers:
        layer_dedup[name] = {}
        if 'quants' in arrays:
            qb = arrays['quants'].tobytes()
            qhash = hashlib.md5(qb).hexdigest()
            if qhash not in quants_to_idx:
                quants_to_idx[qhash] = len(quants_table)
                quants_table.append((shape[1], qb))
            layer_quants_idx[name] = quants_to_idx[qhash]
        for k, v in arrays.items():
            if k != 'quants' and isinstance(v, np.ndarray):
                raw = v.tobytes()
                layer_dedup[name][k] = _dedup(k, raw)

    B.extend(struct.pack('<H', len(quants_table)))
    for in_feat, qb in quants_table:
        B.extend(struct.pack('<I', in_feat))
        B.extend(struct.pack('<I', len(qb)))
        B.extend(qb)

    schema = ['base_packed', 'delta_packed', 'scales_mins_packed', 'ref_indices']
    dedup_schema = [s for s in schema if s in dedup_tables]

    B.extend(struct.pack('<B', len(dedup_schema)))
    for dt in dedup_schema:
        entries = dedup_tables[dt]
        key_b = dt.encode('utf-8')
        B.extend(struct.pack('<B', len(key_b)))
        B.extend(key_b)
        B.extend(struct.pack('<H', len(entries)))
        for entry in entries:
            B.extend(struct.pack('<H', len(entry)))
            B.extend(entry)

    defaults = {}
    for k in ['format', 'quants_no_delta', 'K', 'ref_bits', 'Kq']:
        vals = [a.get(k) for _, _, a in packed_layers if k in a]
        if vals and all(v == vals[0] for v in vals):
            defaults[k] = vals[0]

    B.extend(struct.pack('<B', len(defaults)))
    for k, v in defaults.items():
        kb = k.encode('utf-8')
        B.extend(struct.pack('<B', len(kb)))
        B.extend(kb)
        B.extend(_serialize_leaf(v))

    B.extend(struct.pack('<H', len(packed_layers)))
    for name, shape, arrays in packed_layers:
        enc_type, block_idx, layer_type = _encode_name(name)
        B.extend(struct.pack('<B', enc_type))
        if enc_type == 0:
            B.extend(struct.pack('<B', block_idx))
            B.extend(struct.pack('<B', layer_type))
        elif enc_type <= 2:
            B.extend(struct.pack('<B', layer_type))
        else:
            name_b = name.encode('utf-8')
            B.extend(struct.pack('<B', len(name_b)))
            B.extend(name_b)
        B.extend(struct.pack('<I', shape[0]))
        B.extend(struct.pack('<I', shape[1]))
        qidx = layer_quants_idx.get(name, 0xFFFF)
        B.extend(struct.pack('<H', qidx))

        for s in schema:
            v = arrays.get(s)
            if s in dedup_tables and isinstance(v, np.ndarray):
                didx = layer_dedup[name][s]
                B.extend(struct.pack('<H', didx))
            else:
                B.extend(_serialize_leaf(v))

    total_sz = len(B)
    B[4:8] = struct.pack('<I', total_sz)
    return bytes(B)

    total_sz = len(B)
    B[4:8] = struct.pack('<I', total_sz)
    return bytes(B)


def _serialize_leaf(v):
    if v is None:
        return struct.pack('<B', 0xFF)
    if isinstance(v, np.ndarray):
        raw = v.tobytes()
        L = len(raw)
        if L < 256:
            return struct.pack('<B', 0x00) + struct.pack('<B', L) + raw
        else:
            return struct.pack('<B', 0x00) + struct.pack('<H', L) + raw
    if isinstance(v, bool):
        return struct.pack('<B', 0x01) + (b'\x01' if v else b'\x00')
    if isinstance(v, int):
        if -128 <= v <= 127:
            return struct.pack('<B', 0x02) + struct.pack('<b', v)
        elif -32768 <= v <= 32767:
            return struct.pack('<B', 0x03) + struct.pack('<h', v)
        else:
            return struct.pack('<B', 0x04) + struct.pack('<i', v)
    if isinstance(v, str):
        b = v.encode('utf-8')
        return struct.pack('<B', 0x05) + struct.pack('<B', len(b)) + b
    return struct.pack('<B', 0xFF)


def _save_compressed(meta: dict, save_dir: str, bits: int, fmt: str = "q2_kmeans", consolidated: bool = False) -> int:
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    packed_layers = []
    for name, data in tqdm(meta.items(), desc="Packing layers"):
        packed = _pack_layer_data(data, bits, fmt)
        layer_info[name] = data["shape"]
        packed_layers.append((name, data["shape"], packed))

    raw = _serialize_compact(meta, packed_layers)
    fname = os.path.join(save_dir, "model.bin")
    with open(fname, 'wb') as f:
        f.write(raw)
    total_bytes = os.path.getsize(fname)

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
    quants_delta_K: int = 256,
    sm_share_K: int = 128,
    d_share_K: int = 8,
    quants_delta_bits: int = 2,
    skip_delta_sm: bool = False,
    subblock_delta_bits: int = 0,
    ref_bits: int = 4,
    sparse_delta: bool = False,
    sparse_threshold: float = 0.0,
    quants_no_delta: bool = False,
    consolidated: bool = False,
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
            layer_act = act_stats.get(name) if act_stats is not None else None
            meta[name] = _quantize_one_layer_q4k(layer, quants_delta_K, sm_share_K, d_share_K, act_stats=layer_act, quants_delta_bits=quants_delta_bits, skip_delta_sm=skip_delta_sm, subblock_delta_bits=subblock_delta_bits, ref_bits=ref_bits, sparse_delta=sparse_delta, sparse_threshold=sparse_threshold, quants_no_delta=quants_no_delta)
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
        _save_compressed(meta, save_compressed_dir, bits, fmt, consolidated=consolidated)

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
                        choices=["q2_kmeans", "q4_k"],
                        help="Quantization format (q2_kmeans=K-means 2-bit, q4_k=GGML-style 4-bit blocks)")
    parser.add_argument("--quants-delta-K", type=int, default=256,
                        help="Inter-channel quants delta group size (Kq). 1=full quants, >=2=1 ref + (Kq-1) 2-bit deltas.")
    parser.add_argument("--sm-share-K", type=int, default=128,
                        help="Scale/min sharing group size across output channels.")
    parser.add_argument("--d-share-K", type=int, default=8,
                        help="D/dmin sharing group size across output channels.")
    parser.add_argument("--quants-delta-bits", type=int, default=2, choices=[1, 2],
                        help="Bits per quants delta: 2=lossless(-1,0,1,2) 1=aggressive(0,+1).")
    parser.add_argument("--skip-delta-sm", action="store_true",
                        help="Skip per-channel delta_sm storage (saves ~6 MB).")
    parser.add_argument("--quants-delta-subblock", type=int, default=0,
                        help="Sub-block delta bits (0=per-weight deltas, 1/2/3/4=sub-block granularity).")
    parser.add_argument("--quants-ref-bits", type=int, default=4, choices=[1, 2, 4],
                        help="Reference quants packing: 4=pack_4bit (2/byte), 2=pack_2bit (4/byte), 1=pack_1bit (8/byte).")
    parser.add_argument("--sparse-delta", action="store_true",
                        help="Sparse sub-block delta encoding: skip all-zero superblocks.")
    parser.add_argument("--quants-no-delta", action="store_true",
                        help="Skip quants delta entirely: all channels share reference quants (compressed only).")
    parser.add_argument("--consolidated", action="store_true",
                        help="Save all layers into a single consolidated .npz file (reduces overhead).")
    parser.add_argument("--sparse-threshold", type=float, default=0.0,
                        help="Sub-block delta threshold: fraction of max sub_q to treat as zero (0.0..1.0)."
                             "Higher = more sparsity, lower quality.")
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
    if args.format == "q2_kmeans":
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
    elif args.format == "q4_k":
        # Load cached calibration for activation-weighted LS
        cache_path = args.calibration_cache or "cache/calib_stats.npz"
        if os.path.exists(cache_path):
            logger.info("Loading cached calibration stats for Q4_K LS: %s", cache_path)
            cached = np.load(cache_path, allow_pickle=True)
            act_stats = {k: torch.from_numpy(cached[k]) for k in cached.files}

    compressed_dir = os.path.join(args.save, "compressed") if args.save else None

    effective_bits = 4 if args.format == "q4_k" else args.bits
    logger.info("Quantizing  format=%s  bits=%d  groupsize=%d  symmetric=True",
                args.format, effective_bits, args.groupsize)
    meta = quantize_model(
        model,
        bits=args.bits,
        groupsize=args.groupsize,
        save_compressed_dir=compressed_dir,
        act_stats=act_stats,
        fmt=args.format,
        quants_delta_K=args.quants_delta_K,
        sm_share_K=args.sm_share_K,
        d_share_K=args.d_share_K,
        quants_delta_bits=args.quants_delta_bits,
        skip_delta_sm=args.skip_delta_sm,
        subblock_delta_bits=args.quants_delta_subblock,
        ref_bits=args.quants_ref_bits,
        sparse_delta=args.sparse_delta,
        sparse_threshold=args.sparse_threshold,
        quants_no_delta=args.quants_no_delta,
        consolidated=args.consolidated,
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
