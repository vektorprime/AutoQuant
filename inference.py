#!/usr/bin/env python3
"""Packed-only GPU inference for the custom Q4_K-like affine format."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

import safetensors.torch as st
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

QK_K = 256
QK_K_SUB_BLOCKS = 8
QK_K_SUB_SIZE = QK_K // QK_K_SUB_BLOCKS
SUPPORTED_FORMATS = {"q4k_affine_v2", "q4k_shared_log8"}

import numpy as np

_HADAMARD_CACHE: dict[tuple[int, int, int], list[torch.Tensor]] = {}


def _random_hadamard_np(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]).astype(np.float32)
    row_signs = (rng.randint(0, 2, size=n) * 2 - 1).astype(np.float32)
    col_signs = (rng.randint(0, 2, size=n) * 2 - 1).astype(np.float32)
    H = row_signs[:, None] * H * col_signs[None, :]
    return H * (1.0 / np.sqrt(n))


def _get_hadamard_blocks(
    in_features: int, block_size: int, seed: int, device: torch.device
) -> list[torch.Tensor]:
    n_blocks = in_features // block_size
    cache_key = (in_features, block_size, seed)
    if cache_key not in _HADAMARD_CACHE:
        blocks = []
        for i in range(n_blocks):
            H_np = _random_hadamard_np(block_size, seed + i)
            H_t = torch.from_numpy(H_np.copy()).to(dtype=torch.float32)
            blocks.append(H_t)
        _HADAMARD_CACHE[cache_key] = blocks
    cpu_blocks = _HADAMARD_CACHE[cache_key]
    return [b.to(device=device, dtype=torch.float32) for b in cpu_blocks]


@contextmanager
def _fallback_init_empty_weights():
    """Create meta parameters while leaving constructor-created buffers real.

    Accelerate's init_empty_weights is preferred.  This fallback covers
    standard nn.Parameter instances and avoids allocating a full model.
    """
    original_register_parameter = nn.Module.register_parameter

    def register_meta_parameter(module, name, parameter):
        original_register_parameter(module, name, parameter)
        stored = module._parameters.get(name)
        if stored is not None and stored.device.type != "meta":
            module._parameters[name] = nn.Parameter(
                stored.detach().to(device="meta"),
                requires_grad=stored.requires_grad,
            )

    nn.Module.register_parameter = register_meta_parameter
    try:
        yield
    finally:
        nn.Module.register_parameter = original_register_parameter


def _empty_model_context():
    try:
        from accelerate import init_empty_weights

        return init_empty_weights(include_buffers=False)
    except ImportError:
        return _fallback_init_empty_weights()


def _set_child_module(model: nn.Module, qualified_name: str, module: nn.Module) -> None:
    parent_name, separator, child_name = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_name) if separator else model
    setattr(parent, child_name, module)


def _move_real_buffers(module: nn.Module, device: torch.device) -> None:
    """Move constructor-created non-meta buffers without touching parameters."""
    for submodule in module.modules():
        for name, buffer in list(submodule._buffers.items()):
            if buffer is None or buffer.device.type == "meta":
                continue
            if buffer.device != device:
                submodule._buffers[name] = buffer.to(device=device)


class QuantizedEmbedding(nn.Module):
    """Embedding layer backed by packed 4-bit weights with sparse row lookup."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        packed_data: dict[str, torch.Tensor],
        dtype: torch.dtype,
        symmetric: bool = False,
        sm_bits_scale: int = 6,
        sm_bits_min: int = 6,
        quant_bits: int = 4,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.compute_dtype = dtype
        self._symmetric = symmetric
        self.sm_bits_scale = sm_bits_scale
        self.sm_bits_min = sm_bits_min
        self.quant_bits = quant_bits

        self.register_buffer(
            "quants_packed",
            packed_data["quants_packed"].to(dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "scales_mins_packed",
            packed_data["scales_mins_packed"].to(dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer("d", packed_data["d"], persistent=False)
        if symmetric:
            self.register_buffer("dmin", torch.empty(0), persistent=False)
            if "symmetric_bias" in packed_data:
                self.register_buffer(
                    "symmetric_bias", packed_data["symmetric_bias"], persistent=False
                )
            else:
                self.register_buffer("symmetric_bias", torch.empty(0), persistent=False)
        else:
            self.register_buffer("dmin", packed_data.get("dmin", packed_data.get("dm8")), persistent=False)

        self.n_blocks = self.embedding_dim // QK_K

    def _unpack_quants_rows(self, begin: int, end: int) -> torch.Tensor:
        packed = self.quants_packed[begin:end]
        if self.quant_bits == 6:
            rows = end - begin
            n_groups = packed.shape[1] // 6
            reshaped = packed.reshape(rows, n_groups, 6)
            device = packed.device
            b = reshaped
            b0 = b[:, :, 0]; b1 = b[:, :, 1]; b2 = b[:, :, 2]
            b3 = b[:, :, 3]; b4 = b[:, :, 4]; b5 = b[:, :, 5]
            values = torch.empty(rows, n_groups, 8, dtype=torch.uint8, device=device)
            values[:, :, 0] = (b0 & 0x3F)
            values[:, :, 1] = ((b0 >> 6) | ((b1 & 0x0F) << 2))
            values[:, :, 2] = ((b1 >> 4) | ((b2 & 0x03) << 4))
            values[:, :, 3] = (b2 >> 2) & 0x3F
            values[:, :, 4] = (b3 & 0x3F)
            values[:, :, 5] = ((b3 >> 6) | ((b4 & 0x0F) << 2))
            values[:, :, 6] = ((b4 >> 4) | ((b5 & 0x03) << 4))
            values[:, :, 7] = (b5 >> 2) & 0x3F
            return values.reshape(rows, n_groups * 8)
        if self.quant_bits == 5:
            rows = end - begin
            n_groups = packed.shape[1] // 5
            reshaped = packed.reshape(rows, n_groups, 5).to(dtype=torch.int64)
            packed_40 = reshaped[:, :, 0].long()
            packed_40 |= reshaped[:, :, 1].long() << 8
            packed_40 |= reshaped[:, :, 2].long() << 16
            packed_40 |= reshaped[:, :, 3].long() << 24
            packed_40 |= reshaped[:, :, 4].long() << 32
            values = torch.empty(rows, n_groups, 8, dtype=torch.uint8, device=packed.device)
            for i in range(8):
                values[:, :, i] = ((packed_40 >> (i * 5)) & 0x1F).to(torch.uint8)
            return values.reshape(rows, n_groups * 8)
        low = packed & 0x0F
        high = packed >> 4
        rows, half = packed.shape
        values = torch.empty(rows, half * 2, dtype=torch.uint8, device=packed.device)
        values[:, 0::2] = low
        values[:, 1::2] = high
        return values

    def _unpack_scale_min_rows(self, begin: int, end: int):
        rows = end - begin
        sm = self.scales_mins_packed[begin:end]
        sm_last = sm.shape[-1]

        if self._symmetric:
            s = sm.to(dtype=torch.int64)
            device = s.device
            packed_40 = s[:, :, 0].long()
            packed_40 |= s[:, :, 1].long() << 8
            packed_40 |= s[:, :, 2].long() << 16
            packed_40 |= s[:, :, 3].long() << 24
            packed_40 |= s[:, :, 4].long() << 32
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                sc[:, :, i] = ((packed_40 >> (i * 5)) & 0x1F).to(torch.int32)
            scales = sc.float() / 31.0
            minima = torch.zeros(rows, self.n_blocks, 8, dtype=torch.float32, device=device)
            return scales, minima

        if sm_last == 12:
            packed = sm.to(dtype=torch.int32)
            device = packed.device
            values = torch.zeros(rows, self.n_blocks, QK_K_SUB_BLOCKS, dtype=torch.int32, device=device)
            for pair in range(4):
                i = pair * 2
                byte = pair * 3
                values[:, :, i] = packed[:, :, byte] | ((packed[:, :, byte + 1] & 0x0F) << 8)
                values[:, :, i + 1] = (packed[:, :, byte + 1] >> 4) | (packed[:, :, byte + 2] << 4)
            scales = ((values >> 6) & 0x3F).float() / 63.0
            minima = (values & 0x3F).float() / 63.0
            return scales, minima

        sc_byte_count = 4 if self.sm_bits_scale == 4 else (5 if self.sm_bits_scale == 5 else 6)
        scale_bytes = sm[:, :, :sc_byte_count].to(dtype=torch.int32)
        min_bytes = sm[:, :, sc_byte_count:].to(dtype=torch.int32)
        device = scale_bytes.device

        if self.sm_bits_scale == 4:
            b_s = scale_bytes.to(torch.int64)
            packed_32 = b_s[:, :, 0].long()
            packed_32 |= b_s[:, :, 1].long() << 8
            packed_32 |= b_s[:, :, 2].long() << 16
            packed_32 |= b_s[:, :, 3].long() << 24
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                sc[:, :, i] = ((packed_32 >> (i * 4)) & 0x0F).to(torch.int32)
            scales = sc.float() / 15.0
        elif self.sm_bits_scale == 5:
            b_s = scale_bytes.to(torch.int64)
            packed_40_sc = b_s[:, :, 0].long()
            packed_40_sc |= b_s[:, :, 1].long() << 8
            packed_40_sc |= b_s[:, :, 2].long() << 16
            packed_40_sc |= b_s[:, :, 3].long() << 24
            packed_40_sc |= b_s[:, :, 4].long() << 32
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                sc[:, :, i] = ((packed_40_sc >> (i * 5)) & 0x1F).to(torch.int32)
            scales = sc.float() / 31.0
        else:
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            b = scale_bytes
            sc[:, :, 0] = (b[:, :, 0] & 0x3F)
            sc[:, :, 1] = ((b[:, :, 0] >> 6) | ((b[:, :, 1] & 0x0F) << 2))
            sc[:, :, 2] = ((b[:, :, 1] >> 4) | ((b[:, :, 2] & 0x03) << 4))
            sc[:, :, 3] = (b[:, :, 2] >> 2) & 0x3F
            sc[:, :, 4] = (b[:, :, 3] & 0x3F)
            sc[:, :, 5] = ((b[:, :, 3] >> 6) | ((b[:, :, 4] & 0x0F) << 2))
            sc[:, :, 6] = ((b[:, :, 4] >> 4) | ((b[:, :, 5] & 0x03) << 4))
            sc[:, :, 7] = (b[:, :, 5] >> 2) & 0x3F
            scales = sc.float() / 63.0

        min_bytes_last = min_bytes.shape[-1]
        if min_bytes_last == 5:
            bm = min_bytes.to(torch.int64)
            packed_40 = bm[:, :, 0]
            packed_40 |= bm[:, :, 1] << 8
            packed_40 |= bm[:, :, 2] << 16
            packed_40 |= bm[:, :, 3] << 24
            packed_40 |= bm[:, :, 4] << 32
            mn = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                mn[:, :, i] = ((packed_40 >> (i * 5)) & 0x1F).to(torch.int32)
            minima = mn.float() / 31.0
        elif min_bytes_last == 3:
            bm = min_bytes.to(torch.int64)
            packed_24 = bm[:, :, 0]
            packed_24 |= bm[:, :, 1] << 8
            packed_24 |= bm[:, :, 2] << 16
            mn = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                mn[:, :, i] = ((packed_24 >> (i * 3)) & 0x7).to(torch.int32)
            minima = mn.float() / 7.0
        else:
            bm = min_bytes.to(torch.int64)
            packed_32 = bm[:, :, 0]
            packed_32 |= bm[:, :, 1] << 8
            packed_32 |= bm[:, :, 2] << 16
            packed_32 |= bm[:, :, 3] << 24
            mn = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                mn[:, :, i] = ((packed_32 >> (i * 4)) & 0x0F).to(torch.int32)
            minima = mn.float() / 15.0

        return scales, minima

    def _dequantize_rows(self, begin: int, end: int, dtype: torch.dtype) -> torch.Tensor:
        codes = self._unpack_quants_rows(begin, end).float()
        scale_norm, min_norm = self._unpack_scale_min_rows(begin, end)
        d = self.d[begin:end].float()
        rows = end - begin
        code_blocks = codes.reshape(rows, self.n_blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE)
        if self._symmetric:
            weights = (
                (d.unsqueeze(-1) * scale_norm).unsqueeze(-1) * (code_blocks - 7.5)
            )
            if self.symmetric_bias.numel() > 0:
                b = self.symmetric_bias[begin:end].float()
                weights = weights + b.unsqueeze(-1).unsqueeze(-1)
        else:
            dmin = self.dmin[begin:end].float()
            weights = (
                (d.unsqueeze(-1) * scale_norm).unsqueeze(-1) * code_blocks
                - (dmin.unsqueeze(-1) * min_norm).unsqueeze(-1)
            )
        return weights.reshape(rows, self.embedding_dim).to(dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        unique_ids, inverse = flat.unique(sorted=True, return_inverse=True)

        n_unique = len(unique_ids)
        result = torch.zeros(
            n_unique, self.embedding_dim,
            device=input_ids.device, dtype=self.compute_dtype,
        )

        BATCH = 256
        MAX_DENSE = 2048
        for si in range(0, n_unique, BATCH):
            ei = min(si + BATCH, n_unique)
            batch_uids = unique_ids[si:ei]
            init_dense = batch_uids[0].item()
            end_dense = batch_uids[-1].item() + 1

            if end_dense - init_dense <= MAX_DENSE:
                full = self._dequantize_rows(init_dense, end_dense, self.compute_dtype)
                result[si:ei] = full[batch_uids - init_dense]
            else:
                for bi, rid in enumerate(batch_uids.tolist()):
                    result[si + bi] = self._dequantize_rows(
                        rid, rid + 1, self.compute_dtype
                    ).squeeze(0)

        output = result[inverse]
        return output.reshape(*input_ids.shape, self.embedding_dim)


class QuantizedLinear(nn.Module):
    """Linear layer backed only by packed 4-bit weights.

    The current fallback kernel dequantizes output-row tiles.  It bounds peak
    memory but is slower than a fused CUDA/Triton kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        packed_data: dict[str, torch.Tensor],
        dtype: torch.dtype,
        tile_rows: int = 512,
        sm_bits_scale: int = 6,
        sm_bits_min: int = 6,
        hadamard: bool = False,
        hadamard_block_size: int = 256,
        hadamard_seed: int = 42,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = dtype
        self.tile_rows = int(tile_rows)
        self.sm_bits_scale = sm_bits_scale
        self.sm_bits_min = sm_bits_min
        self._hadamard = hadamard
        self._hadamard_block_size = hadamard_block_size
        self._hadamard_seed = hadamard_seed
        self._h_blocks: list[torch.Tensor] | None = None
        self._symmetric = symmetric

        self.register_buffer(
            "quants_packed",
            packed_data["quants_packed"].to(dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "scales_mins_packed",
            packed_data["scales_mins_packed"].to(dtype=torch.uint8),
            persistent=False,
        )

        d_key = "d" if "d" in packed_data else "d8"
        dmin_key = "dmin" if "dmin" in packed_data else "dm8"
        if d_key not in packed_data:
            raise RuntimeError("Packed layer is missing d scale tensor")
        self.register_buffer("d", packed_data[d_key], persistent=False)
        if symmetric:
            self.register_buffer("dmin", torch.empty(0), persistent=False)
            if "symmetric_bias" in packed_data:
                self.register_buffer(
                    "symmetric_bias",
                    packed_data["symmetric_bias"],
                    persistent=False,
                )
            else:
                self.register_buffer("symmetric_bias", torch.empty(0), persistent=False)
        else:
            if dmin_key not in packed_data:
                raise RuntimeError("Packed layer is missing dmin scale tensor")
            self.register_buffer("dmin", packed_data[dmin_key], persistent=False)
        if not self.d.is_floating_point():
            raise RuntimeError(
                "d/dmin must be linear floating-point scales. This checkpoint "
                "looks like an incompatible log8 layout."
            )
        if not symmetric and self.d.dtype != self.dmin.dtype:
            raise RuntimeError(
                f"d and dmin dtype mismatch: {self.d.dtype} versus {self.dmin.dtype}"
            )

        if self.in_features % QK_K != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by {QK_K}"
            )
        self.n_blocks = self.in_features // QK_K

        expected_quants = (self.out_features, self.in_features // 2)
        sm_shape = tuple(self.scales_mins_packed.shape)
        if sm_shape[-1] not in (5, 8, 9, 10, 11, 12):
            raise RuntimeError(
                f"Bad scale/min packed shape {sm_shape}; expected last dim 5 (symmetric), 8, 9, 10, 11, or 12"
            )
        if sm_shape[-1] == 5 and not self._symmetric:
            self._symmetric = True
        self._sm_delta = (
            sm_shape[-1] == 9 and sm_bits_scale == 6 and sm_bits_min == 5 and not self._symmetric
        )
        self._sm_asymmetric = sm_shape[-1] in (5, 8, 9, 10, 11)
        expected_sm = (self.out_features, self.n_blocks, sm_shape[-1])
        expected_scale = (self.out_features, self.n_blocks)
        if tuple(self.quants_packed.shape) != expected_quants:
            raise RuntimeError(
                f"Bad quants shape {tuple(self.quants_packed.shape)}; "
                f"expected {expected_quants}"
            )
        if sm_shape != expected_sm:
            raise RuntimeError(
                f"Bad scale/min shape {sm_shape}; "
                f"expected {expected_sm}"
            )
        if self._symmetric:
            if tuple(self.d.shape) != expected_scale:
                raise RuntimeError(
                    f"Bad d shape: d={tuple(self.d.shape)}, expected={expected_scale}"
                )
        else:
            if tuple(self.d.shape) != expected_scale or tuple(self.dmin.shape) != expected_scale:
                raise RuntimeError(
                    f"Bad d/dmin shape: d={tuple(self.d.shape)}, "
                    f"dmin={tuple(self.dmin.shape)}, expected={expected_scale}"
                )

        if has_bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device="meta", dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def _unpack_quants_rows(self, begin: int, end: int) -> torch.Tensor:
        packed = self.quants_packed[begin:end]
        low = packed & 0x0F
        high = packed >> 4
        rows, half = packed.shape
        values = torch.empty(
            rows,
            half * 2,
            dtype=torch.uint8,
            device=packed.device,
        )
        values[:, 0::2] = low
        values[:, 1::2] = high
        return values

    def _unpack_scale_min_rows(
        self, begin: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = end - begin
        if self._symmetric:
            sm = self.scales_mins_packed[begin:end].to(dtype=torch.int64)
            device = sm.device
            packed_40 = sm[:, :, 0].long()
            packed_40 |= sm[:, :, 1].long() << 8
            packed_40 |= sm[:, :, 2].long() << 16
            packed_40 |= sm[:, :, 3].long() << 24
            packed_40 |= sm[:, :, 4].long() << 32
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                sc[:, :, i] = ((packed_40 >> (i * 5)) & 0x1F).to(torch.int32)
            scales = sc.float() / 31.0
            minima = torch.zeros(rows, self.n_blocks, 8, dtype=torch.float32, device=device)
            return scales, minima

        if self._sm_delta:
            sm = self.scales_mins_packed[begin:end].to(dtype=torch.int64)
            device = sm.device

            scale_ref = sm[:, :, 0] & 0x3F
            min_ref = ((sm[:, :, 0] >> 6) & 0x03) << 3
            min_ref = min_ref | (sm[:, :, 1] & 0x07)

            scales = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            minima = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            scales[:, :, 0] = scale_ref.to(torch.int32)
            minima[:, :, 0] = min_ref.to(torch.int32)

            for i in range(7):
                bi = i + 1
                sd = ((sm[:, :, bi] >> 3) & 0x0F) - 8
                if i < 6:
                    md = ((sm[:, :, bi] >> 7) & 0x01) << 3
                    md = md | (sm[:, :, bi + 1] & 0x07)
                else:
                    md = ((sm[:, :, 7] >> 7) & 0x01) << 3
                    md = md | (sm[:, :, 8] & 0x07)
                md = md - 8
                scales[:, :, i + 1] = torch.clamp(
                    scale_ref + sd, 1, 63
                ).to(torch.int32)
                minima[:, :, i + 1] = torch.clamp(
                    min_ref + md, 0, 31
                ).to(torch.int32)

            return scales.float() / 63.0, minima.float() / 31.0

        if not self._sm_asymmetric:
            packed = self.scales_mins_packed[begin:end].to(dtype=torch.int32)
            values = torch.empty(
                rows,
                self.n_blocks,
                QK_K_SUB_BLOCKS,
                dtype=torch.int32,
                device=packed.device,
            )
            for pair in range(4):
                i = pair * 2
                byte = pair * 3
                values[:, :, i] = (
                    packed[:, :, byte]
                    | ((packed[:, :, byte + 1] & 0x0F) << 8)
                )
                values[:, :, i + 1] = (
                    (packed[:, :, byte + 1] >> 4)
                    | (packed[:, :, byte + 2] << 4)
                )
            scales = ((values >> 6) & 0x3F).float() / 63.0
            minima = (values & 0x3F).float() / 63.0
            return scales, minima

        sc_byte_count = 4 if self.sm_bits_scale == 4 else (5 if self.sm_bits_scale == 5 else 6)
        scale_bytes = self.scales_mins_packed[begin:end, :, :sc_byte_count].to(dtype=torch.int32)
        min_bytes = self.scales_mins_packed[begin:end, :, sc_byte_count:].to(dtype=torch.int32)
        device = scale_bytes.device

        if self.sm_bits_scale == 4:
            b_s = scale_bytes.to(torch.int64)
            packed_32 = b_s[:, :, 0].long()
            packed_32 |= b_s[:, :, 1].long() << 8
            packed_32 |= b_s[:, :, 2].long() << 16
            packed_32 |= b_s[:, :, 3].long() << 24
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                sc[:, :, i] = ((packed_32 >> (i * 4)) & 0x0F).to(torch.int32)
            scales = sc.float() / 15.0
        elif self.sm_bits_scale == 5:
            b_s = scale_bytes.to(torch.int64)
            packed_40_sc = b_s[:, :, 0].long()
            packed_40_sc |= b_s[:, :, 1].long() << 8
            packed_40_sc |= b_s[:, :, 2].long() << 16
            packed_40_sc |= b_s[:, :, 3].long() << 24
            packed_40_sc |= b_s[:, :, 4].long() << 32
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                sc[:, :, i] = ((packed_40_sc >> (i * 5)) & 0x1F).to(torch.int32)
            scales = sc.float() / 31.0
        else:
            sc = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            b = scale_bytes
            sc[:, :, 0] = (b[:, :, 0] & 0x3F)
            sc[:, :, 1] = ((b[:, :, 0] >> 6) | ((b[:, :, 1] & 0x0F) << 2))
            sc[:, :, 2] = ((b[:, :, 1] >> 4) | ((b[:, :, 2] & 0x03) << 4))
            sc[:, :, 3] = (b[:, :, 2] >> 2) & 0x3F
            sc[:, :, 4] = (b[:, :, 3] & 0x3F)
            sc[:, :, 5] = ((b[:, :, 3] >> 6) | ((b[:, :, 4] & 0x0F) << 2))
            sc[:, :, 6] = ((b[:, :, 4] >> 4) | ((b[:, :, 5] & 0x03) << 4))
            sc[:, :, 7] = (b[:, :, 5] >> 2) & 0x3F
            scales = sc.float() / 63.0

        min_bytes_last = min_bytes.shape[-1]
        if min_bytes_last == 5:
            bm = min_bytes.to(torch.int64)
            packed_40 = bm[:, :, 0]
            packed_40 |= bm[:, :, 1] << 8
            packed_40 |= bm[:, :, 2] << 16
            packed_40 |= bm[:, :, 3] << 24
            packed_40 |= bm[:, :, 4] << 32
            mn = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                mn[:, :, i] = ((packed_40 >> (i * 5)) & 0x1F).to(torch.int32)
            minima = mn.float() / 31.0
        elif min_bytes_last == 3:
            bm = min_bytes.to(torch.int64)
            packed_24 = bm[:, :, 0]
            packed_24 |= bm[:, :, 1] << 8
            packed_24 |= bm[:, :, 2] << 16
            mn = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                mn[:, :, i] = ((packed_24 >> (i * 3)) & 0x7).to(torch.int32)
            minima = mn.float() / 7.0
        else:
            bm = min_bytes.to(torch.int64)
            packed_32 = bm[:, :, 0]
            packed_32 |= bm[:, :, 1] << 8
            packed_32 |= bm[:, :, 2] << 16
            packed_32 |= bm[:, :, 3] << 24
            mn = torch.empty(rows, self.n_blocks, 8, dtype=torch.int32, device=device)
            for i in range(8):
                mn[:, :, i] = ((packed_32 >> (i * 4)) & 0x0F).to(torch.int32)
            minima = mn.float() / 15.0

        return scales, minima

    def _dequantize_rows(
        self,
        begin: int,
        end: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        codes = self._unpack_quants_rows(begin, end).float()
        scale_norm, min_norm = self._unpack_scale_min_rows(begin, end)
        d = self.d[begin:end].float()

        rows = end - begin
        code_blocks = codes.reshape(
            rows,
            self.n_blocks,
            QK_K_SUB_BLOCKS,
            QK_K_SUB_SIZE,
        )
        if self._symmetric:
            weights = (
                (d.unsqueeze(-1) * scale_norm).unsqueeze(-1) * (code_blocks - 7.5)
            )
            if self.symmetric_bias.numel() > 0:
                b = self.symmetric_bias[begin:end].float()
                weights = weights + b.unsqueeze(-1).unsqueeze(-1)
        else:
            dmin = self.dmin[begin:end].float()
            weights = (
                (d.unsqueeze(-1) * scale_norm).unsqueeze(-1) * code_blocks
                - (dmin.unsqueeze(-1) * min_norm).unsqueeze(-1)
            )
        return weights.reshape(rows, self.in_features).to(dtype=dtype)

    def dequantize_full(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Materialize this layer for codec/parity diagnostics only."""
        return self._dequantize_rows(
            0,
            self.out_features,
            self.compute_dtype if dtype is None else dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = inputs.dtype if inputs.is_floating_point() else self.compute_dtype

        if self._hadamard:
            if self._h_blocks is None:
                self._h_blocks = _get_hadamard_blocks(
                    self.in_features, self._hadamard_block_size,
                    self._hadamard_seed, inputs.device,
                )
            inputs = inputs.float()
            n_blocks = self.in_features // self._hadamard_block_size
            for i in range(n_blocks):
                start = i * self._hadamard_block_size
                end = start + self._hadamard_block_size
                inputs[..., start:end] = inputs[..., start:end] @ self._h_blocks[i].T
            inputs = inputs.to(dtype=compute_dtype)

        output = torch.empty(
            *inputs.shape[:-1],
            self.out_features,
            device=inputs.device,
            dtype=compute_dtype,
        )

        effective_tile_rows = (
            self.out_features if self.tile_rows <= 0 else min(self.tile_rows, self.out_features)
        )
        for begin in range(0, self.out_features, effective_tile_rows):
            end = min(begin + effective_tile_rows, self.out_features)
            weight = self._dequantize_rows(begin, end, compute_dtype)
            bias = self.bias[begin:end] if self.bias is not None else None
            if bias is not None and bias.dtype != compute_dtype:
                bias = bias.to(dtype=compute_dtype)
            output[..., begin:end] = nn.functional.linear(inputs, weight, bias)
        return output


def _load_packed_tensors(
    model_dir: str,
    manifest: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    weight_files = manifest.get("weight_files")
    if not weight_files:
        format_name = manifest["format"]
        weight_files = [f"{format_name}-00001-of-00001.safetensors"]

    tensors: dict[str, torch.Tensor] = {}
    for filename in weight_files:
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Packed weight file not found: {path}")
        shard = st.load_file(path, device=device)
        duplicates = set(tensors).intersection(shard)
        if duplicates:
            raise RuntimeError(f"Duplicate packed tensors: {sorted(duplicates)[:10]}")
        tensors.update(shard)
    return tensors


def load_quantized_model(
    model_dir: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    tile_rows: int | None = None,
):
    """Load only packed Q4_K weights plus explicitly residual tensors."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        torch_device = torch.device(torch_device.type, torch.cuda.current_device())
    if tile_rows is None:
        tile_rows = int(os.environ.get("Q4K_TILE_ROWS", "512"))
    device_string = str(torch_device)
    manifest_path = os.path.join(model_dir, "q4k_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as handle:
        manifest = json.load(handle)
    format_name = manifest.get("format")
    if format_name not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported packed format '{format_name}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )
    if manifest.get("block_size") != QK_K:
        raise ValueError(
            f"Unsupported block size {manifest.get('block_size')}; expected {QK_K}"
        )

    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    packed_all = _load_packed_tensors(model_dir, manifest, device_string)

    with _empty_model_context():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    quantized_layers: dict[str, dict] = manifest["quantized_layers"]
    quantized_embeddings: dict[str, dict] = manifest.get("quantized_embeddings", {})
    linear_map = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    embed_map = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
    }

    layer_packed: dict[str, dict[str, torch.Tensor]] = {}
    embed_packed: dict[str, dict[str, torch.Tensor]] = {}
    marker = f".{format_name}."
    for tensor_name, tensor in packed_all.items():
        if marker not in tensor_name:
            raise RuntimeError(f"Unexpected packed tensor name: {tensor_name}")
        layer_name, key = tensor_name.rsplit(marker, 1)
        if layer_name in quantized_embeddings:
            embed_packed.setdefault(layer_name, {})[key] = tensor
        else:
            layer_packed.setdefault(layer_name, {})[key] = tensor

    expected_layers = set(quantized_layers)
    actual_layers = set(layer_packed)
    if expected_layers != actual_layers:
        raise RuntimeError(
            "Packed layer set does not match manifest: "
            f"missing={sorted(expected_layers - actual_layers)[:20]}, "
            f"unexpected={sorted(actual_layers - expected_layers)[:20]}"
        )

    expected_embeds = set(quantized_embeddings)
    actual_embeds = set(embed_packed)
    if expected_embeds != actual_embeds:
        raise RuntimeError(
            "Packed embedding set does not match manifest: "
            f"missing={sorted(expected_embeds - actual_embeds)[:20]}, "
            f"unexpected={sorted(actual_embeds - expected_embeds)[:20]}"
        )

    required_common = {"quants_packed", "scales_mins_packed"}
    for name in sorted(expected_layers):
        if name not in linear_map:
            raise RuntimeError(f"Manifest layer '{name}' is not an nn.Linear")
        fields = layer_packed[name]
        if "sf_d_packed" in fields or "sf_dm_packed" in fields:
            raise RuntimeError(
                "This checkpoint uses the older shared/log8 scale layout. "
                "Re-quantize with q4k_affine_v2; decoding it as direct fp16 "
                "scales would produce incorrect weights."
            )
        if "d8" in fields and fields["d8"].dtype == torch.uint8:
            raise RuntimeError(
                "This checkpoint stores logarithmic uint8 scales. Re-quantize "
                "with q4k_affine_v2 because this loader does not silently "
                "reinterpret log8 values as linear scales."
            )
        missing_fields = required_common - set(fields)
        is_layer_symmetric = bool(
            quantized_layers[name].get("symmetric", False)
            or manifest.get("symmetric", False)
        )
        if is_layer_symmetric:
            if "d" not in fields:
                missing_fields.add("d")
        else:
            if not ({"d", "dmin"} <= set(fields) or {"d8", "dm8"} <= set(fields)):
                missing_fields.update({"d/dmin"})
        if missing_fields:
            raise RuntimeError(
                f"Packed layer '{name}' is missing fields: {sorted(missing_fields)}"
            )
        d_field = fields["d"] if "d" in fields else fields["d8"]
        expected_scale_dtype = quantized_layers[name].get(
            "scale_dtype", manifest.get("scale_dtype")
        )
        actual_scale_dtype = str(d_field.dtype).removeprefix("torch.")
        if expected_scale_dtype and actual_scale_dtype != expected_scale_dtype:
            raise RuntimeError(
                f"Scale dtype mismatch for '{name}': manifest={expected_scale_dtype}, "
                f"stored={actual_scale_dtype}"
            )

        original = linear_map[name]
        shape = quantized_layers[name]["shape"]
        expected_shape = [original.out_features, original.in_features]
        if list(shape) != expected_shape:
            raise RuntimeError(
                f"Shape mismatch for '{name}': manifest={shape}, model={expected_shape}"
            )
        architecture_has_bias = original.bias is not None
        manifest_has_bias = bool(
            quantized_layers[name].get("has_bias", architecture_has_bias)
        )
        if manifest.get("format_version", 1) >= 2 and manifest_has_bias != architecture_has_bias:
            raise RuntimeError(
                f"Bias metadata mismatch for '{name}': "
                f"manifest={manifest_has_bias}, architecture={architecture_has_bias}"
            )
        # Version-1 writers hard-coded has_bias=False.  Trust the architecture
        # for those checkpoints so residual Q/K/V biases are not discarded.
        has_bias = architecture_has_bias

        sm_shape = tuple(fields["scales_mins_packed"].shape)
        sm_last = sm_shape[-1]
        layer_info = quantized_layers[name]
        _sm_bits_scale = layer_info.get("sm_bits_scale")
        _sm_bits_min = layer_info.get("sm_bits_min")
        _symmetric = bool(layer_info.get("symmetric", False) or manifest.get("symmetric", False))
        if _sm_bits_scale is None or _sm_bits_min is None:
            if sm_last == 5 and (_symmetric or manifest.get("symmetric")):
                _sm_bits_scale = 5
                _sm_bits_min = 0
            else:
                _sm_bits_scale = 6
                if sm_last == 12:
                    _sm_bits_min = 6
                elif sm_last == 11:
                    _sm_bits_min = 5
                elif sm_last == 10:
                    _sm_bits_min = 4
                elif sm_last == 9:
                    _sm_bits_min = 4
                elif sm_last == 8:
                    _sm_bits_scale = 4
                    _sm_bits_min = 4
                else:
                    _sm_bits_min = 6

        _hadamard = bool(layer_info.get("hadamard", False) or manifest.get("hadamard", False))
        _hadamard_block_size = int(layer_info.get("hadamard_block_size", 256))
        _hadamard_seed = int(layer_info.get("hadamard_seed", 42))

        replacement = QuantizedLinear(
            original.in_features,
            original.out_features,
            has_bias,
            fields,
            dtype=dtype,
            tile_rows=tile_rows,
            sm_bits_scale=_sm_bits_scale,
            sm_bits_min=_sm_bits_min,
            hadamard=_hadamard,
            hadamard_block_size=_hadamard_block_size,
            hadamard_seed=_hadamard_seed,
            symmetric=_symmetric,
        )
        _set_child_module(model, name, replacement)

    for name in sorted(expected_embeds):
        if name not in embed_map:
            raise RuntimeError(f"Manifest embedding '{name}' is not an nn.Embedding")
        info = quantized_embeddings[name]
        fields = embed_packed[name]
        if "d" not in fields or "quants_packed" not in fields:
            raise RuntimeError(f"Packed embedding '{name}' is missing required fields")
        _symmetric = bool(info.get("symmetric", False) or manifest.get("symmetric", False))
        _embed_sm_bits_scale = info.get("sm_bits_scale", 6)
        _embed_sm_bits_min = info.get("sm_bits_min", 6)
        _embed_quant_bits = info.get("quant_bits", 4)
        original = embed_map[name]
        shape = info["shape"]
        expected_shape = [original.num_embeddings, original.embedding_dim]
        if list(shape) != expected_shape:
            raise RuntimeError(
                f"Shape mismatch for embedding '{name}': manifest={shape}, model={expected_shape}"
            )
        replacement = QuantizedEmbedding(
            original.num_embeddings,
            original.embedding_dim,
            fields,
            dtype=dtype,
            symmetric=_symmetric,
            sm_bits_scale=_embed_sm_bits_scale,
            sm_bits_min=_embed_sm_bits_min,
            quant_bits=_embed_quant_bits,
        )
        _set_child_module(model, name, replacement)

    residual_filename = manifest.get("residual_file", "residual.safetensors")
    residual_path = os.path.join(model_dir, residual_filename)
    if not os.path.isfile(residual_path):
        raise FileNotFoundError(f"Residual checkpoint not found: {residual_path}")
    residual_all = st.load_file(residual_path, device=device_string)

    expected_residual = set(manifest.get("residual_parameters", residual_all.keys()))
    actual_residual = set(residual_all)
    if expected_residual != actual_residual:
        raise RuntimeError(
            "Residual tensor set does not match manifest: "
            f"missing={sorted(expected_residual - actual_residual)[:20]}, "
            f"unexpected={sorted(actual_residual - expected_residual)[:20]}"
        )

    incompatible = model.load_state_dict(residual_all, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Residual state_dict mismatch: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )

    _move_real_buffers(model, torch_device)

    remaining_meta = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    remaining_meta.extend(
        name
        for name, buffer in model.named_buffers()
        if buffer.device.type == "meta"
    )
    if remaining_meta:
        raise RuntimeError(
            "Model still contains meta tensors after loading: "
            f"{remaining_meta[:20]}"
        )

    wrong_device = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device != torch_device
    ]
    wrong_device.extend(
        name
        for name, buffer in model.named_buffers()
        if buffer.device != torch_device
    )
    if wrong_device:
        raise RuntimeError(
            f"Model tensors are not all on {torch_device}: {wrong_device[:20]}"
        )

    model.eval()
    del packed_all, residual_all, layer_packed
    print(
        f"Loaded {len(expected_layers)} packed layers and "
        f"{len(expected_residual)} residual tensors on {torch_device}"
    )
    return model, tokenizer


def chat(model_dir: str, device: str = "cuda", tile_rows: int | None = None) -> None:
    model, tokenizer = load_quantized_model(
        model_dir, device=device, tile_rows=tile_rows
    )
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() == "quit":
            break
        messages.append({"role": "user", "content": user})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        output = model.generate(
            **inputs,
            max_new_tokens=16384,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
            streamer=streamer,
        )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        messages.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Packed-only Q4_K model chat")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="quantized_models/chat_model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tile-rows",
        type=int,
        default=None,
        help=(
            "Output rows dequantized per GEMM. 0 means the whole layer and is "
            "useful for parity testing against a materialized BF16 checkpoint. "
            "Default: Q4K_TILE_ROWS or 512."
        ),
    )
    arguments = parser.parse_args()
    chat(
        arguments.model_dir,
        device=arguments.device,
        tile_rows=arguments.tile_rows,
    )
