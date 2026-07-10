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

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def _get_hadamard(n: int) -> torch.Tensor:
    """Return normalized n×n Hadamard matrix (n must be a power of 2)."""
    if n in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[n]
    import math
    H = torch.ones(1, 1)
    k = 1
    while k < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
        k *= 2
    H = H.float() / math.sqrt(n)
    _HADAMARD_CACHE[n] = H
    return H


def _get_layer_groupsize(name: str, default_groupsize: int) -> int:
    attn_keywords = ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head")
    if any(kw in name for kw in attn_keywords):
        return max(16, default_groupsize // 2)
    return default_groupsize


NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Vector quantization: groups of 4 adjacent weights, 256-entry codebook
# ~2.03 bits/weight. Codebook shared across all output rows per block.
# ---------------------------------------------------------------------------

VQ_BLOCK = 256
VQ_VEC_SIZE = 4
VQ_CODEBOOK_SIZE = 256

_GLOBAL_VQ_CODEBOOK: torch.Tensor | None = None

def _quantize_one_layer_vq(layer: nn.Linear) -> dict:
    """Vector quantization: 256-entry codebook of 4-vectors, trained once on first
    layer's data and reused globally. ~2.0 bits/weight."""
    global _GLOBAL_VQ_CODEBOOK

    W = layer.weight.data.float()
    out_features, in_features = W.shape

    if in_features % VQ_BLOCK != 0:
        raise ValueError(f"in_features ({in_features}) not divisible by {VQ_BLOCK}")

    n_blocks = in_features // VQ_BLOCK
    n_vecs_per_row = VQ_BLOCK // VQ_VEC_SIZE  # 64
    K = VQ_CODEBOOK_SIZE
    ASSIGN_BATCH = 4096

    # Train global codebook on first call (subsampled)
    if _GLOBAL_VQ_CODEBOOK is None:
        sample_vecs = W.reshape(-1, VQ_VEC_SIZE)
        N_all = sample_vecs.shape[0]
        train_N = min(N_all, 65536)
        train_vecs = sample_vecs[torch.randperm(N_all)[:train_N]]
        centroids = train_vecs[torch.randperm(train_N)[:K]].clone()
        for _ in range(5):
            new_c = torch.zeros(K, VQ_VEC_SIZE)
            counts = torch.zeros(K)
            for s in range(0, train_N, ASSIGN_BATCH):
                batch = train_vecs[s:s + ASSIGN_BATCH]
                c_n = (centroids ** 2).sum(dim=1).unsqueeze(0)
                v_n = (batch ** 2).sum(dim=1, keepdim=True)
                dots = batch @ centroids.T
                dists = v_n + c_n - 2 * dots
                a = dists.argmin(dim=1)
                for d in range(VQ_VEC_SIZE):
                    new_c[:, d] = new_c[:, d].index_add(0, a, batch[:, d])
                counts = counts.index_add(0, a, torch.ones(a.numel()))
            e = counts == 0
            counts[e] = 1
            centroids = new_c / counts.unsqueeze(1)
            centroids[e] = train_vecs[torch.randint(0, train_N, (e.sum().item(),))]
        _GLOBAL_VQ_CODEBOOK = centroids

    centroids = _GLOBAL_VQ_CODEBOOK

    W_r = W.reshape(out_features, n_blocks, n_vecs_per_row, VQ_VEC_SIZE)

    indices = torch.zeros(out_features, n_blocks, n_vecs_per_row, dtype=torch.uint8)
    codebooks_q = torch.zeros(n_blocks, K, VQ_VEC_SIZE, dtype=torch.uint8)
    cb_scales = torch.zeros(n_blocks, VQ_VEC_SIZE, dtype=torch.float32)
    cb_offsets = torch.zeros(n_blocks, VQ_VEC_SIZE, dtype=torch.float32)

    for b in range(n_blocks):
        vectors = W_r[:, b, :, :].reshape(-1, VQ_VEC_SIZE)
        N_local = vectors.shape[0]

        # Minibatched assignment with global codebook
        all_assign = torch.empty(N_local, dtype=torch.long)
        c_norms = (centroids ** 2).sum(dim=1).unsqueeze(0)
        for start in range(0, N_local, ASSIGN_BATCH):
            batch = vectors[start:start + ASSIGN_BATCH]
            v_norms = (batch ** 2).sum(dim=1, keepdim=True)
            dots = batch @ centroids.T
            dists = v_norms + c_norms - 2 * dots
            all_assign[start:start + batch.size(0)] = dists.argmin(dim=1)
        indices[:, b, :] = all_assign.reshape(out_features, n_vecs_per_row).to(torch.uint8)

        for d in range(VQ_VEC_SIZE):
            cb_d = centroids[:, d]
            min_d = cb_d.min()
            max_d = cb_d.max()
            scale_d = (max_d - min_d) / 255.0
            if scale_d < 1e-8:
                scale_d = 1.0
            cb_scales[b, d] = scale_d
            cb_offsets[b, d] = min_d
            codebooks_q[b, :, d] = torch.clamp(
                torch.round((cb_d - min_d) / scale_d), 0, 255
            ).to(torch.uint8)

    W_q = torch.zeros_like(W)
    for b in range(n_blocks):
        cb_deq = cb_offsets[b:b+1, :].float() + codebooks_q[b].float() * cb_scales[b:b+1, :].float()
        idx_block = indices[:, b, :]
        cb_lookup = cb_deq[idx_block.long()]
        start = b * VQ_BLOCK
        end = start + VQ_BLOCK
        W_q[:, start:end] = cb_lookup.reshape(out_features, VQ_BLOCK)

    layer.weight.data = W_q.to(layer.weight.dtype)

    indices_flat = indices.reshape(out_features, in_features // VQ_VEC_SIZE).numpy()

    return {
        "codebook_q":  codebooks_q.numpy(),
        "cb_scales":   cb_scales.numpy().astype(np.float16),
        "cb_offsets":  cb_offsets.numpy().astype(np.float16),
        "indices":     indices_flat,
        "format":      "vq",
        "shape":       [out_features, in_features],
        "n_blocks":    n_blocks,
    }


def _quantize_one_layer_nf4(layer: nn.Linear, groupsize: int = 64) -> dict:
    """NF4: NormalFloat 4-bit. Non-uniform levels from QLoRA paper.
    Block size groupsize, one fp16 scale per block. At groupsize=32: 4.5 bpw.
    At groupsize=64: 4.25 bpw.  At groupsize=128: 4.125 bpw."""
    W = layer.weight.data.float()
    out_features, in_features = W.shape
    assert in_features % groupsize == 0, (
        f"in_features ({in_features}) must be divisible by groupsize ({groupsize})")
    n_blocks = in_features // groupsize

    levels = NF4_LEVELS.clone()
    W_r = W.reshape(out_features, n_blocks, groupsize)

    w_max_abs = torch.maximum(
        W_r.amax(dim=-1).abs(), W_r.amin(dim=-1).abs()
    )  # (out, n_blocks)
    w_max_abs[w_max_abs < 1e-8] = 1e-8

    W_scaled = W_r / w_max_abs.unsqueeze(-1)  # in [-1, 1]

    levels_dev = levels.to(W.device)
    dists = (W_scaled.unsqueeze(-1) - levels_dev).abs()  # (out, n_blocks, groupsize, 16)
    q_idx = dists.argmin(dim=-1).to(torch.uint8)  # (out, n_blocks, groupsize)

    W_q_r = w_max_abs.unsqueeze(-1) * levels_dev[q_idx.long()]
    W_q = W_q_r.reshape(out_features, in_features)
    layer.weight.data = W_q.to(layer.weight.dtype)

    quants_flat = q_idx.reshape(out_features, in_features)

    return {
        "scales":         w_max_abs.numpy().astype(np.float16),
        "quants":         quants_flat.numpy(),
        "format":         "nf4",
        "groupsize":      groupsize,
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
    }


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

def _quantize_one_layer_q4k(layer: nn.Linear, hadamard: bool = False, clip_ratio: float = 0.0) -> dict:
    W = layer.weight.data.float()
    out_features, in_features = W.shape

    assert in_features % QK_K == 0, (
        f"in_features ({in_features}) must be divisible by QK_K ({QK_K})")
    n_blocks = in_features // QK_K

    if hadamard:
        H = _get_hadamard(QK_K).to(W.device)
        W_flat = W.reshape(-1, QK_K)
        W = (W_flat @ H.T).reshape(out_features, in_features)

    W_r = W.reshape(out_features, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE)
    # shape: (out, n_blocks, 8, 32)

    if clip_ratio > 0:
        W_r_sorted = W_r.sort(dim=-1).values
        lo_idx = max(0, int(QK_K_SUB_SIZE * clip_ratio))
        hi_idx = min(QK_K_SUB_SIZE - 1, int(QK_K_SUB_SIZE * (1 - clip_ratio)))
        w_sub_min = W_r_sorted[..., lo_idx]
        w_sub_max = W_r_sorted[..., hi_idx]
    else:
        w_sub_min = W_r.amin(dim=-1)
        w_sub_max = W_r.amax(dim=-1)

    # Per-sub-block min/max → per-block (superblock) min/max
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

    if hadamard:
        W_q_flat = W_q.reshape(-1, QK_K)
        W_q = (W_q_flat @ H).reshape(out_features, in_features)

    layer.weight.data = W_q.to(layer.weight.dtype)

    quants_flat = q.reshape(out_features, in_features)

    if clip_ratio > 0:
        fmt = "q4_k_clip"
    elif hadamard:
        fmt = "q4_k_hadamard"
    else:
        fmt = "q4_k"

    return {
        "d":              d.numpy().astype(np.float16),
        "dmin":           dmin.numpy().astype(np.float16),
        "scales_6bit":    sc_6bit.numpy(),          # (out, n_blocks, 8)
        "mins_6bit":      m_6bit.numpy(),           # (out, n_blocks, 8)
        "quants":         quants_flat.numpy(),      # (out, in_features) raw 0-15
        "format":         fmt,
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
    }


# Q4_K sub4: 4 sub-blocks of 64 (instead of 8 of 32), saving 6 bytes/superblock
QK_K_SUB4_BLOCKS = 4
QK_K_SUB4_SIZE = QK_K // QK_K_SUB4_BLOCKS  # 64


def _quantize_one_layer_q4k_sub4(layer: nn.Linear) -> dict:
    """Q4_K with 4 sub-blocks of 64 + LS refinement of d/dmin.
    Metadata: fp16 d + fp16 dmin (4B) + 4*(6+6)bit scales (6B) = 10B per 256 weights.
    Total: 138 bytes per superblock → 4.3125 bpw."""
    W = layer.weight.data.float()
    W_orig = W.clone()
    out_features, in_features = W.shape

    assert in_features % QK_K == 0
    n_blocks = in_features // QK_K

    W_r = W.reshape(out_features, n_blocks, QK_K_SUB4_BLOCKS, QK_K_SUB4_SIZE)
    # shape: (out, n_blocks, 4, 64)

    for refine_iter in range(3):
        w_sub_min = W_r.amin(dim=-1)      # (out, n_blocks, 4)
        w_sub_max = W_r.amax(dim=-1)
        w_blk_min = w_sub_min.amin(dim=-1)
        w_blk_max = w_sub_max.amax(dim=-1)

        d = (w_blk_max - w_blk_min) / 15.0
        d[d < 1e-8] = 1e-8
        dmin = w_blk_min.abs().clamp(min=1e-8)

        d_sub = (w_sub_max - w_sub_min) / 15.0
        d_sub[d_sub < 1e-8] = 1e-8

        sc_ratio = d_sub / d.unsqueeze(-1).clamp(min=1e-8)
        sc_6bit = torch.clamp(torch.round(sc_ratio * 63.0), 1, 63).to(torch.uint8)
        sc_norm = sc_6bit.float() / 63.0

        min_ratio = w_sub_min.abs() / dmin.unsqueeze(-1).clamp(min=1e-8)
        m_6bit = torch.clamp(torch.round(min_ratio * 63.0), 0, 63).to(torch.uint8)
        m_norm = m_6bit.float() / 63.0

        eff_scale = d.unsqueeze(-1) * sc_norm         # (out, n_blocks, 4)
        eff_scale[eff_scale < 1e-8] = 1e-8
        eff_offset = dmin.unsqueeze(-1) * m_norm       # (out, n_blocks, 4)

        q = torch.round(
            (W_r + eff_offset.unsqueeze(-1)) / eff_scale.unsqueeze(-1)
        )
        q = torch.clamp(q, 0, 15).to(torch.uint8)

        # LS refinement of d and dmin per (out_channel, block)
        # w ≈ d * s - dmin * m  where s = sc_norm * q, m = m_norm
        s = sc_norm.unsqueeze(-1) * q.float()  # (out, n_blocks, 4, 64)
        m = m_norm.unsqueeze(-1).expand(-1, -1, -1, QK_K_SUB4_SIZE)  # (out, n_blocks, 4, 64)

        s_flat = s.flatten(start_dim=2)  # (out, n_blocks, 256)
        m_flat = m.flatten(start_dim=2)
        w_flat = W_orig.reshape(out_features, n_blocks, QK_K)  # (out, n_blocks, 256)

        # Solve [d, dmin] via least squares: min ||w_flat - d*s_flat + dmin*m_flat||^2
        s_sq = (s_flat * s_flat).sum(dim=-1)
        m_sq = (m_flat * m_flat).sum(dim=-1)
        sm = (s_flat * m_flat).sum(dim=-1)
        ws = (w_flat * s_flat).sum(dim=-1)
        wm = (w_flat * m_flat).sum(dim=-1)

        det = s_sq * m_sq - sm * sm
        det[det.abs() < 1e-8] = 1e-8
        d_new = (ws * m_sq - wm * sm) / det
        dmin_new = (ws * sm - wm * s_sq) / det

        d = d_new.clamp(min=1e-8)
        dmin = dmin_new.clamp(min=1e-8)

        sc_ratio = d_sub / d.unsqueeze(-1).clamp(min=1e-8)
        sc_6bit = torch.clamp(torch.round(sc_ratio * 63.0), 1, 63).to(torch.uint8)
        sc_norm = sc_6bit.float() / 63.0

        min_ratio = w_sub_min.abs() / dmin.unsqueeze(-1).clamp(min=1e-8)
        m_6bit = torch.clamp(torch.round(min_ratio * 63.0), 0, 63).to(torch.uint8)
        m_norm = m_6bit.float() / 63.0

        eff_scale = d.unsqueeze(-1) * sc_norm
        eff_scale[eff_scale < 1e-8] = 1e-8
        eff_offset = dmin.unsqueeze(-1) * m_norm

        q = torch.round(
            (W_r + eff_offset.unsqueeze(-1)) / eff_scale.unsqueeze(-1)
        )
        q = torch.clamp(q, 0, 15).to(torch.uint8)

    W_q_r = eff_scale.unsqueeze(-1) * q.float() - eff_offset.unsqueeze(-1)
    W_q = W_q_r.reshape(out_features, in_features)
    layer.weight.data = W_q.to(layer.weight.dtype)

    quants_flat = q.reshape(out_features, in_features)

    return {
        "d":              d.numpy().astype(np.float16),
        "dmin":           dmin.numpy().astype(np.float16),
        "scales_6bit":    sc_6bit.numpy(),          # (out, n_blocks, 4)
        "mins_6bit":      m_6bit.numpy(),           # (out, n_blocks, 4)
        "quants":         quants_flat.numpy(),
        "format":         "q4_k_sub4",
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
    }


def _quantize_one_layer_q4k_sym(layer: nn.Linear) -> dict:
    """Symmetric Q4_K: no offset/dmin, saves 8 B/superblock (4.25 bpw vs 4.5)."""
    W = layer.weight.data.float()
    out_features, in_features = W.shape

    assert in_features % QK_K == 0, (
        f"in_features ({in_features}) must be divisible by QK_K ({QK_K})")
    n_blocks = in_features // QK_K

    W_r = W.reshape(out_features, n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE)
    # shape: (out, n_blocks, 8, 32)

    w_sub_max_abs = torch.maximum(
        W_r.amax(dim=-1).abs(), W_r.amin(dim=-1).abs()
    )  # (out, n_blocks, 8)
    w_blk_max_abs = w_sub_max_abs.amax(dim=-1)  # (out, n_blocks)

    d = w_blk_max_abs / 7.0
    d[d < 1e-8] = 1e-8

    d_sub = w_sub_max_abs / 7.0
    d_sub[d_sub < 1e-8] = 1e-8

    sc_ratio = d_sub / d.unsqueeze(-1).clamp(min=1e-8)
    sc_6bit = torch.clamp(torch.round(sc_ratio * 63.0), 1, 63).to(torch.uint8)
    sc_norm = sc_6bit.float() / 63.0

    eff_scale = d.unsqueeze(-1) * sc_norm  # (out, n_blocks, 8)
    eff_scale[eff_scale < 1e-8] = 1e-8

    zero_pt = 8.0
    q = torch.round(W_r / eff_scale.unsqueeze(-1) + zero_pt)
    q = torch.clamp(q, 0, 15).to(torch.uint8)

    W_q_r = eff_scale.unsqueeze(-1) * (q.float() - zero_pt)
    W_q = W_q_r.reshape(out_features, in_features)
    layer.weight.data = W_q.to(layer.weight.dtype)

    quants_flat = q.reshape(out_features, in_features)

    return {
        "d":              d.numpy().astype(np.float16),
        "scales_6bit":    sc_6bit.numpy(),
        "quants":         quants_flat.numpy(),
        "format":         "q4_k_sym",
        "shape":          [out_features, in_features],
        "n_blocks":       n_blocks,
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


def _save_compressed(meta: dict, save_dir: str, bits: int, fmt: str = "q2_kmeans") -> int:
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    for name, data in tqdm(meta.items(), desc="Saving compressed"):
        fname = os.path.join(save_dir, name + ".npz")
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if data.get("format") == "vq":
            np.savez(fname,
                     codebook_q=data["codebook_q"],
                     cb_scales=data["cb_scales"],
                     cb_offsets=data["cb_offsets"],
                     indices=data["indices"],
                     format=data["format"])
        elif data.get("format") in ("q4_k", "q4_k_hadamard", "q4_k_clip", "q4_k_sub4"):
            # Q4_K format: pack 4-bit quants into uint8 pairs
            quants = data["quants"]
            q_out = _pack_4bit(quants)
            np.savez(fname,
                     quants=q_out,
                     d=data["d"],
                     dmin=data["dmin"],
                     scales_6bit=data["scales_6bit"],
                     mins_6bit=data["mins_6bit"],
                     format=data["format"])
        elif data.get("format") == "q4_k_sym":
            quants = data["quants"]
            q_out = _pack_4bit(quants)
            np.savez(fname,
                     quants=q_out,
                     d=data["d"],
                     scales_6bit=data["scales_6bit"],
                     format=data["format"])
        elif data.get("format") == "nf4":
            quants = data["quants"]
            q_out = _pack_4bit(quants)
            np.savez(fname,
                     quants=q_out,
                     scales=data["scales"],
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

        if fmt == "vq":
            if layer.weight.shape[1] % VQ_BLOCK != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], VQ_BLOCK)
                continue
            meta[name] = _quantize_one_layer_vq(layer)
        elif fmt == "q4_k":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k(layer)
        elif fmt == "q4_k_hadamard":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k(layer, hadamard=True)
        elif fmt == "q4_k_sym":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k_sym(layer)
        elif fmt == "q4_k_clip":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k(layer, clip_ratio=0.05)
        elif fmt == "q4_k_sub4":
            if layer.weight.shape[1] % QK_K != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], QK_K)
                continue
            meta[name] = _quantize_one_layer_q4k_sub4(layer)
        elif fmt == "nf4":
            layer_gs = _get_layer_groupsize(name, groupsize)
            if groupsize != -1 and layer.weight.shape[1] % layer_gs != 0:
                logger.warning("Skipping %s: in_features %d not divisible by %d",
                               name, layer.weight.shape[1], layer_gs)
                continue
            meta[name] = _quantize_one_layer_nf4(layer, groupsize=layer_gs)
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
                        choices=["q2_kmeans", "vq", "q4_k", "q4_k_hadamard", "q4_k_sym", "nf4", "q4_k_clip", "q4_k_sub4"],
                        help="Quantization format")
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

    effective_bits = 2.03 if args.format == "vq" else (4 if args.format in ("q4_k", "q4_k_hadamard", "q4_k_sym", "nf4", "q4_k_clip", "q4_k_sub4") else args.bits)
    logger.info("Quantizing  format=%s  bits=%d  groupsize=%d  symmetric=True",
                args.format, effective_bits, args.groupsize)
    meta = quantize_model(
        model,
        bits=args.bits,
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
