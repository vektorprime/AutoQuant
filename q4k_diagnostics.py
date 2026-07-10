#!/usr/bin/env python3
"""Inspect Q4_K packed checkpoints and compare them with an old BF16 checkpoint.

This script does not load a Transformers model. It compares serialized weights
so codec/quantizer differences can be separated from model-runtime differences.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
import safetensors.torch as st

QK_K = 256
QK_K_SUB_BLOCKS = 8
QK_K_SUB_SIZE = 32


def _load_packed(model_dir: Path, manifest: dict) -> dict[str, torch.Tensor]:
    files = manifest.get("weight_files") or [
        f"{manifest['format']}-00001-of-00001.safetensors"
    ]
    result: dict[str, torch.Tensor] = {}
    for filename in files:
        shard = st.load_file(str(model_dir / filename), device="cpu")
        overlap = set(result).intersection(shard)
        if overlap:
            raise RuntimeError(f"Duplicate packed tensors: {sorted(overlap)[:10]}")
        result.update(shard)
    return result


def _group_layers(packed: dict[str, torch.Tensor], fmt: str) -> dict[str, dict[str, torch.Tensor]]:
    marker = f".{fmt}."
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for tensor_name, tensor in packed.items():
        if marker not in tensor_name:
            raise RuntimeError(f"Unexpected packed tensor name: {tensor_name}")
        layer, key = tensor_name.rsplit(marker, 1)
        grouped.setdefault(layer, {})[key] = tensor
    return grouped


def _unpack_sm_rows(sm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = sm.to(torch.int32)
    rows, blocks, _ = packed.shape
    values = torch.empty(rows, blocks, 8, dtype=torch.int32)
    for pair in range(4):
        i = pair * 2
        b = pair * 3
        values[:, :, i] = packed[:, :, b] | ((packed[:, :, b + 1] & 0x0F) << 8)
        values[:, :, i + 1] = (packed[:, :, b + 1] >> 4) | (packed[:, :, b + 2] << 4)
    return ((values >> 6) & 0x3F).float() / 63.0, (values & 0x3F).float() / 63.0


def decode_rows(fields: dict[str, torch.Tensor], begin: int, end: int, scale_fp16: bool = False) -> torch.Tensor:
    q = fields["quants_packed"][begin:end]
    rows, half = q.shape
    codes = torch.empty(rows, half * 2, dtype=torch.uint8)
    codes[:, 0::2] = q & 0x0F
    codes[:, 1::2] = q >> 4

    sc, mn = _unpack_sm_rows(fields["scales_mins_packed"][begin:end])
    d_key = "d" if "d" in fields else "d8"
    dm_key = "dmin" if "dmin" in fields else "dm8"
    d = fields[d_key][begin:end]
    dm = fields[dm_key][begin:end]
    if scale_fp16:
        d = d.to(torch.float16)
        dm = dm.to(torch.float16)
    d = d.float()
    dm = dm.float()

    in_features = half * 2
    if in_features % QK_K:
        raise RuntimeError(f"Bad in_features={in_features}")
    blocks = in_features // QK_K
    q4 = codes.reshape(rows, blocks, QK_K_SUB_BLOCKS, QK_K_SUB_SIZE).float()
    weights = (
        (d.unsqueeze(-1) * sc).unsqueeze(-1) * q4
        - (dm.unsqueeze(-1) * mn).unsqueeze(-1)
    )
    return weights.reshape(rows, in_features)


class ReferenceWeights:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        single_path = model_dir / "model.safetensors"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            self.weight_map = index["weight_map"]
        elif single_path.is_file():
            with safe_open(str(single_path), framework="pt", device="cpu") as handle:
                self.weight_map = {key: single_path.name for key in handle.keys()}
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.safetensors.index.json under {model_dir}"
            )

    def get(self, key: str) -> torch.Tensor:
        filename = self.weight_map.get(key)
        if filename is None:
            raise KeyError(key)
        with safe_open(str(self.model_dir / filename), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def _select_layers(names: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(names):
        return names
    if count == 1:
        return [names[len(names) // 2]]
    positions = torch.linspace(0, len(names) - 1, count).round().long().tolist()
    return [names[i] for i in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Packed Q4_K directory")
    parser.add_argument(
        "--reference-dir",
        help="Optional old dequantized BF16 Hugging Face checkpoint to compare",
    )
    parser.add_argument("--layers", type=int, default=12, help="Number of layers to sample; 0=all")
    parser.add_argument("--rows", type=int, default=32, help="Output rows sampled per layer; 0=all")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "q4k_manifest.json").read_text())
    packed = _load_packed(model_dir, manifest)
    layers = _group_layers(packed, manifest["format"])
    selected = _select_layers(sorted(layers), args.layers)

    print(f"format={manifest.get('format')} version={manifest.get('format_version')}")
    print(f"manifest scale_dtype={manifest.get('scale_dtype')} refine_mode={manifest.get('refine_mode')}")
    actual_scale_dtypes = sorted({str((f.get('d') if 'd' in f else f['d8']).dtype) for f in layers.values()})
    print(f"actual serialized scale dtypes={actual_scale_dtypes}")
    print(f"quantized layers={len(layers)}, sampled={len(selected)}")

    total_values = 0
    changed_after_bf16 = 0
    squared_fp16_effect = 0.0
    max_fp16_effect = 0.0

    reference = ReferenceWeights(Path(args.reference_dir)) if args.reference_dir else None
    total_reference_values = 0
    exact_reference_values = 0
    squared_reference_error = 0.0
    max_reference_error = 0.0
    missing_reference: list[str] = []

    for name in selected:
        fields = layers[name]
        out_features = fields["quants_packed"].shape[0]
        rows = out_features if args.rows <= 0 else min(args.rows, out_features)
        # Sample the start, middle and end rather than only the first rows.
        if rows == out_features:
            row_ids = torch.arange(out_features)
        else:
            row_ids = torch.linspace(0, out_features - 1, rows).round().long().unique()

        decoded_parts = []
        decoded_fp16_parts = []
        for row in row_ids.tolist():
            decoded_parts.append(decode_rows(fields, row, row + 1, scale_fp16=False))
            decoded_fp16_parts.append(decode_rows(fields, row, row + 1, scale_fp16=True))
        decoded = torch.cat(decoded_parts, dim=0).to(torch.bfloat16).float()
        decoded_fp16 = torch.cat(decoded_fp16_parts, dim=0).to(torch.bfloat16).float()
        difference = (decoded - decoded_fp16).abs()
        total_values += difference.numel()
        changed_after_bf16 += int((difference != 0).sum())
        squared_fp16_effect += float((difference * difference).sum())
        max_fp16_effect = max(max_fp16_effect, float(difference.max()))

        if reference is not None:
            key = f"{name}.weight"
            try:
                reference_weight = reference.get(key)
            except KeyError:
                missing_reference.append(key)
                continue
            reference_rows = reference_weight[row_ids].to(torch.bfloat16).float()
            ref_difference = (decoded - reference_rows).abs()
            total_reference_values += ref_difference.numel()
            exact_reference_values += int((ref_difference == 0).sum())
            squared_reference_error += float((ref_difference * ref_difference).sum())
            max_reference_error = max(max_reference_error, float(ref_difference.max()))

    fp16_rmse = (squared_fp16_effect / max(total_values, 1)) ** 0.5
    print("\nFP32-scale decode versus forced FP16-scale decode, after BF16 weight cast:")
    print(f"  changed values: {changed_after_bf16}/{total_values} ({100*changed_after_bf16/max(total_values,1):.6f}%)")
    print(f"  RMSE: {fp16_rmse:.9g}")
    print(f"  max abs: {max_fp16_effect:.9g}")

    if reference is not None:
        ref_rmse = (squared_reference_error / max(total_reference_values, 1)) ** 0.5
        print("\nPacked decode versus old dequantized checkpoint, both viewed as BF16:")
        print(f"  exact values: {exact_reference_values}/{total_reference_values} ({100*exact_reference_values/max(total_reference_values,1):.6f}%)")
        print(f"  RMSE: {ref_rmse:.9g}")
        print(f"  max abs: {max_reference_error:.9g}")
        if missing_reference:
            print(f"  missing reference keys: {missing_reference[:10]}")
        if total_reference_values and exact_reference_values == total_reference_values:
            print("  RESULT: sampled packed weights exactly match the old BF16 checkpoint.")
        else:
            print("  RESULT: the packed checkpoint and old BF16 checkpoint encode different weights.")


if __name__ == "__main__":
    main()
