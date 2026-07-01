#!/usr/bin/env python3
"""
q2_k post-training quantization for AutoModelForCausalLM.

Per-group min/max symmetric quantization — no calibration data, no iterative
optimisation.  Symmetric is hardcoded (always on).
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantizer import Quantizer

logger = logging.getLogger(__name__)

# Quantization is always symmetric — no asymmetric support.
SYMMETRIC = True


# ---------------------------------------------------------------------------
# q2_k per-layer quantization
# ---------------------------------------------------------------------------

def _quantize_one_layer(
    layer: nn.Linear,
    bits: int,
    groupsize: int,
) -> dict:
    """
    Quantize a single nn.Linear layer with symmetric per-group min/max.

    Returns dict with integer codes, scales, and zeros for compressed storage.
    The layer's weight is updated in-place with the dequantised values.
    """
    W = layer.weight.data.float()
    out_features, in_features = W.shape
    g = groupsize if groupsize != -1 else in_features
    n_groups = in_features // g

    qz = Quantizer(bits=bits, symmetric=SYMMETRIC, groupsize=groupsize)
    qz.find_params(W)
    W_q = qz.quantize(W)  # dequantised float32

    # Integer codes via reshape-based broadcasting (no repeat_interleave).
    W_r = W.reshape(out_features, n_groups, g)
    codes = torch.clamp(
        torch.round(W_r / qz.scale.unsqueeze(-1)) + qz.zero.unsqueeze(-1),
        0, qz.maxq,
    ).to(torch.uint8).reshape(out_features, in_features)

    layer.weight.data = W_q.to(layer.weight.dtype)

    return {
        "codes": codes.cpu().numpy(),
        "scales": qz.scale.cpu().numpy().astype(np.float32),
        "zeros": qz.zero.cpu().numpy().astype(np.float32),
        "shape": [out_features, in_features],
    }


def _pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 4 × 2-bit values into one uint8 (in_features must be multiple of 4)."""
    out, inp = codes.shape
    assert inp % 4 == 0
    codes = codes.reshape(out, inp // 4, 4)
    packed = (codes[..., 0]
              | (codes[..., 1] << 2)
              | (codes[..., 2] << 4)
              | (codes[..., 3] << 6))
    return packed.astype(np.uint8)


def _save_compressed(meta: dict, save_dir: str, bits: int) -> int:
    """Save integer codes (packed), scales, and zeros to disk. Returns total bytes."""
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    for name, data in meta.items():
        fname = os.path.join(save_dir, name + ".npz")
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if bits == 2 and data["shape"][1] % 4 == 0:
            codes_out = _pack_2bit(data["codes"])
        else:
            codes_out = data["codes"]

        np.savez_compressed(fname,
                            codes=codes_out,
                            scales=data["scales"],
                            zeros=data["zeros"])
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
    groupsize: int = 128,
    save_compressed_dir: str | None = None,
) -> dict:
    """
    Quantize all nn.Linear layers in *model*.

    Weights are updated in-place.  Returns per-layer code/scale/zero metadata.
    """
    model.eval()
    meta: dict = {}

    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if groupsize != -1 and layer.weight.shape[1] % groupsize != 0:
            logger.warning("Skipping %s: in_features %d not divisible by %d",
                           name, layer.weight.shape[1], groupsize)
            continue

        meta[name] = _quantize_one_layer(layer, bits, groupsize)

    if save_compressed_dir:
        _save_compressed(meta, save_compressed_dir, bits)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="q2_k quantization for causal LMs (symmetric, always on)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--groupsize", type=int, default=128,
                        help="Group size (min 16, must divide in_features)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Precision for loading the base model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default=None,
                        help="Directory to save quantized model")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    logger.info("Loading model: %s (dtype=%s)", args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    compressed_dir = os.path.join(args.save, "compressed") if args.save else None

    logger.info("Quantizing  bits=%d  groupsize=%d  symmetric=%s",
                args.bits, args.groupsize, SYMMETRIC)
    meta = quantize_model(
        model,
        bits=args.bits,
        groupsize=args.groupsize,
        save_compressed_dir=compressed_dir,
    )
    logger.info("Quantized %d linear layers.", len(meta))

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)
        logger.info("Saved dequantised model to '%s'", args.save)


if __name__ == "__main__":
    main()
