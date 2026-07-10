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

def _quantize_one_layer_q4k(layer: nn.Linear, quants_delta_K: int = 64) -> dict:
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

    for _ in range(3):
        s = sc_norm.unsqueeze(-1) * q.float()
        s_flat = s.flatten(start_dim=2)
        m = m_norm.unsqueeze(-1).expand(-1, -1, -1, QK_K_SUB_SIZE)
        m_flat = m.flatten(start_dim=2)

        s_sq = (s_flat * s_flat).sum(dim=-1)
        m_sq = (m_flat * m_flat).sum(dim=-1)
        sm = (s_flat * m_flat).sum(dim=-1)
        ws = (W_flat * s_flat).sum(dim=-1)
        wm = (W_flat * m_flat).sum(dim=-1)

        det = s_sq * m_sq - sm * sm
        det[det.abs() < 1e-12] = 1e-12
        d_new = (ws * m_sq - wm * sm) / det
        dmin_new = (ws * sm - wm * s_sq) / det

        d = d_new.clamp(min=1e-8)
        dmin = dmin_new.abs().clamp(min=1e-8)

    # Share d/dmin across K=4 output channels with 4-bit per-channel scale factors
    K = 4
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

    # Share scales/mins across K_sm=4 output channels with 2-bit multiplicative deltas
    K_sm = 4
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

    delta_sm_packed = _pack_q4k_sm_deltas(delta_sc, delta_m)

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
        q_ref = q_grp[:, 0, :]
        q_tgt = q_grp[:, 1:, :].reshape(-1, q_ref.shape[1])
        quants_ref_packed = _pack_4bit(q_ref)
        quants_delta_packed = _pack_quants_delta(q_ref, q_tgt, Kq - 1)
        quants_kw = dict(quants_ref=quants_ref_packed, quants_delta=quants_delta_packed, Kq=Kq)
    else:
        quants_kw = dict(quants=quants_np)

    return {
        **quants_kw,
        "base_packed":    base_packed,
        "delta_packed":   delta_packed,
        "scales_mins_shared": shared_sc,
        "mins_shared":    shared_m,
        "delta_sm":       delta_sm_packed,
        "format":         "q4_k",
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
        "K":              K,
        "K_sm":           K_sm,
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


def _pack_quants_delta(ref_quants: np.ndarray, tgt_quants: np.ndarray, n_tgt: int) -> np.ndarray:
    """Pack 256 × 2-bit signed deltas (-1,0,1,2) into 64 bytes per target channel."""
    out, inp = ref_quants.shape
    n_blocks = inp // QK_K
    ref_sb = ref_quants.reshape(out, n_blocks, QK_K).astype(np.int16)
    tgt_sb = tgt_quants.reshape(n_tgt, out, n_blocks, QK_K).astype(np.int16)
    packed = np.zeros((n_tgt, out, n_blocks, 64), dtype=np.uint8)
    for t in range(n_tgt):
        diff = tgt_sb[t] - ref_sb
        delta = np.clip(diff + 1, 0, 3).astype(np.uint8)
        for o in range(out):
            for b in range(n_blocks):
                packed[t, o, b] = _pack_2bit(delta[o, b].reshape(1, QK_K)).reshape(64)
    return packed


def _pack_q4k_sm_joint(shared_sc: np.ndarray, shared_m: np.ndarray) -> np.ndarray:
    """Pack 8 × (5+5 bit joint scale+min) = 80 bits → 10 bytes per superblock."""
    ng, nb, nsub = shared_sc.shape
    assert nsub == 8
    sc5 = np.clip(np.round(shared_sc.astype(np.float32) / 2.0), 0, 31).astype(np.uint16)
    m5 = np.clip(np.round(shared_m.astype(np.float32) / 2.0), 0, 31).astype(np.uint16)
    joint = (sc5 << 5) | m5
    packed = np.zeros((ng, nb, 10), dtype=np.uint8)
    for p in range(2):
        i = p * 4
        v0 = joint[:, :, i]
        v1 = joint[:, :, i + 1]
        v2 = joint[:, :, i + 2]
        v3 = joint[:, :, i + 3]
        b = p * 5
        packed[:, :, b] = v0 & 0xFF
        packed[:, :, b + 1] = ((v0 >> 8) & 0x3) | ((v1 & 0x3F) << 2)
        packed[:, :, b + 2] = ((v1 >> 6) & 0xF) | ((v2 & 0xF) << 4)
        packed[:, :, b + 3] = ((v2 >> 4) & 0x3F) | ((v3 & 0x3) << 6)
        packed[:, :, b + 4] = (v3 >> 2) & 0xFF
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


def _pack_fp16_to_log8(arr: np.ndarray) -> np.ndarray:
    """Pack fp16 d/dmin into uint8 log-scale. ~6 MB savings per model."""
    arr = arr.astype(np.float32)
    arr = np.maximum(arr, 1e-8)
    log2 = np.log2(arr)
    encoded = np.clip(np.round(log2 * 12.0 + 128.0), 0, 255).astype(np.uint8)
    return encoded


def _save_compressed(meta: dict, save_dir: str, bits: int, fmt: str = "q2_kmeans") -> int:
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    for name, data in tqdm(meta.items(), desc="Saving compressed"):
        fname = os.path.join(save_dir, name + ".npz")
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if data.get("format") == "q4_k":
            if "quants_ref" in data:
                q_out = data["quants_ref"]
                q_delta = data["quants_delta"]
            else:
                quants = data["quants"]
                q_out = _pack_4bit(quants)
                q_delta = None
            if "scales_mins_shared" in data:
                sm_packed = _pack_q4k_sm_joint(data["scales_mins_shared"], data["mins_shared"])
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
                    save_kw["quants_delta"] = q_delta
                    save_kw["Kq"] = data.get("Kq", 2)
                if "delta_sm" in data:
                    save_kw["delta_sm"] = data["delta_sm"]
                    save_kw["K_sm"] = data.get("K_sm", 4)
                np.savez(fname, **save_kw)
            else:
                d8 = _pack_fp16_to_log8(data["d"])
                dm8 = _pack_fp16_to_log8(data["dmin"])
                np.savez(fname,
                         quants=q_out,
                         d8=d8,
                         dm8=dm8,
                         scales_mins_packed=sm_packed,
                         format=data["format"])
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
    quants_delta_K: int = 64,
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
            meta[name] = _quantize_one_layer_q4k(layer, quants_delta_K)
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
                        choices=["q2_kmeans", "q4_k"],
                        help="Quantization format (q2_kmeans=K-means 2-bit, q4_k=GGML-style 4-bit blocks)")
    parser.add_argument("--quants-delta-K", type=int, default=64,
                        help="Inter-channel quants delta group size (Kq). 1=full quants, >=2=1 ref + (Kq-1) 2-bit deltas.")
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
