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
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = dtype
        self.tile_rows = int(tile_rows)

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
        if d_key not in packed_data or dmin_key not in packed_data:
            raise RuntimeError("Packed layer is missing d/dmin scale tensors")
        self.register_buffer("d", packed_data[d_key], persistent=False)
        self.register_buffer("dmin", packed_data[dmin_key], persistent=False)
        if not self.d.is_floating_point() or not self.dmin.is_floating_point():
            raise RuntimeError(
                "d/dmin must be linear floating-point scales. This checkpoint "
                "looks like an incompatible log8 layout."
            )
        if self.d.dtype != self.dmin.dtype:
            raise RuntimeError(
                f"d and dmin dtype mismatch: {self.d.dtype} versus {self.dmin.dtype}"
            )

        if self.in_features % QK_K != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by {QK_K}"
            )
        self.n_blocks = self.in_features // QK_K

        expected_quants = (self.out_features, self.in_features // 2)
        expected_sm = (self.out_features, self.n_blocks, 12)
        expected_scale = (self.out_features, self.n_blocks)
        if tuple(self.quants_packed.shape) != expected_quants:
            raise RuntimeError(
                f"Bad quants shape {tuple(self.quants_packed.shape)}; "
                f"expected {expected_quants}"
            )
        if tuple(self.scales_mins_packed.shape) != expected_sm:
            raise RuntimeError(
                f"Bad scale/min shape {tuple(self.scales_mins_packed.shape)}; "
                f"expected {expected_sm}"
            )
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
        packed = self.scales_mins_packed[begin:end].to(dtype=torch.int32)
        rows = end - begin
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

    def _dequantize_rows(
        self,
        begin: int,
        end: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        codes = self._unpack_quants_rows(begin, end).float()
        scale_norm, min_norm = self._unpack_scale_min_rows(begin, end)
        d = self.d[begin:end].float()
        dmin = self.dmin[begin:end].float()

        rows = end - begin
        code_blocks = codes.reshape(
            rows,
            self.n_blocks,
            QK_K_SUB_BLOCKS,
            QK_K_SUB_SIZE,
        )
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
    linear_map = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    layer_packed: dict[str, dict[str, torch.Tensor]] = {}
    marker = f".{format_name}."
    for tensor_name, tensor in packed_all.items():
        if marker not in tensor_name:
            raise RuntimeError(f"Unexpected packed tensor name: {tensor_name}")
        layer_name, key = tensor_name.rsplit(marker, 1)
        layer_packed.setdefault(layer_name, {})[key] = tensor

    expected_layers = set(quantized_layers)
    actual_layers = set(layer_packed)
    if expected_layers != actual_layers:
        raise RuntimeError(
            "Packed layer set does not match manifest: "
            f"missing={sorted(expected_layers - actual_layers)[:20]}, "
            f"unexpected={sorted(actual_layers - expected_layers)[:20]}"
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
        replacement = QuantizedLinear(
            original.in_features,
            original.out_features,
            has_bias,
            fields,
            dtype=dtype,
            tile_rows=tile_rows,
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
