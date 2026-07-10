#!/usr/bin/env python3
"""Small CPU validation suite for the packed Q4_K codec."""
from __future__ import annotations

import copy
import tempfile

import torch

import inference
import quantize


def test_codec_and_tiled_linear() -> None:
    for scale_dtype in (torch.float16, torch.float32):
        torch.manual_seed(123)
        layer = torch.nn.Linear(512, 33, bias=True, dtype=torch.float32)
        original_weight = layer.weight.detach().clone()
        packed = quantize._encode_q4k(layer, scale_dtype=scale_dtype)
        assert torch.equal(layer.weight, original_weight), "encoder mutated source weight"

        decoded = quantize.decode_q4k(packed, dtype=torch.float32)
        fields = {
            key: torch.from_numpy(value)
            for key, value in packed.items()
            if hasattr(value, "dtype")
        }
        runtime = inference.QuantizedLinear(
            512,
            33,
            has_bias=True,
            packed_data=fields,
            dtype=torch.float32,
            tile_rows=11,
        )
        runtime.bias = torch.nn.Parameter(
            layer.bias.detach().clone(), requires_grad=False
        )

        inputs = torch.randn(2, 3, 512)
        expected = torch.nn.functional.linear(inputs, decoded, layer.bias)
        actual = runtime(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_refinement_improves_weight_error() -> None:
    torch.manual_seed(456)
    layer = torch.nn.Linear(1024, 64, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight) * 0.02)

    unrefined = quantize.decode_q4k(
        quantize._encode_q4k(
            layer,
            scale_dtype=torch.float32,
            refine_iters=0,
        )
    )
    refined = quantize.decode_q4k(
        quantize._encode_q4k(
            layer,
            scale_dtype=torch.float32,
            refine_iters=3,
        )
    )
    target = layer.weight.detach().float()
    unrefined_mse = torch.mean((target - unrefined) ** 2)
    refined_mse = torch.mean((target - refined) ** 2)
    assert refined_mse < unrefined_mse, (
        f"refinement did not improve MSE: {refined_mse} >= {unrefined_mse}"
    )


def test_source_model_is_not_mutated() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(256, 16, bias=True)

    model = TinyModel()
    before = copy.deepcopy(model.state_dict())
    quantize.quantize_model(
        model,
        fmt="q4_k",
        q4k_scale_dtype=torch.float32,
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    test_codec_and_tiled_linear()
    test_refinement_improves_weight_error()
    test_source_model_is_not_mutated()
    print("All Q4_K codec tests passed.")
