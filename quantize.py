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
GROUPSIZE = 32


# ---------------------------------------------------------------------------
# Per-layer 2-bit symmetric quantization (vectorised, CPU)
# ---------------------------------------------------------------------------

def _quantize_one_layer(
    layer: nn.Linear,
    bits: int,
    groupsize: int,
) -> dict:
    W = layer.weight.data.float()                  # CPU float32
    out_features, in_features = W.shape
    g = groupsize if groupsize != -1 else in_features
    n_groups = in_features // g
    maxq = MAXQ[bits]
    zero_pt = (maxq + 1) / 2                       # 2.0 for 2-bit

    # ---------- per-group scale (reshape → amin/amax, zero-copy view) -----
    W_r = W.reshape(out_features, n_groups, g)     # view
    xmax = torch.maximum(W_r.amin(dim=-1).abs(),
                         W_r.amax(dim=-1))          # (out, n_groups)
    scale = xmax / (maxq / 2)
    scale[scale == 0] = 1.0

    # ---------- quantise + dequantise in one broadcast pass --------------
    # scale.unsqueeze(-1) broadcasts with W_r via (out, n_groups, 1) → (out, n_groups, g)
    q = torch.clamp(torch.round(W_r / scale.unsqueeze(-1)) + zero_pt,
                    0, maxq)
    W_q = (scale.unsqueeze(-1) * (q - zero_pt)).reshape(out_features,
                                                         in_features)

    layer.weight.data = W_q.to(layer.weight.dtype)

    return {
        "codes":  q.reshape(out_features, in_features).to(torch.uint8).numpy(),
        "scales": scale.numpy().astype(np.float32),
        "zeros":  np.full(scale.shape, zero_pt, dtype=np.float32),
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


def _save_compressed(meta: dict, save_dir: str, bits: int) -> int:
    os.makedirs(save_dir, exist_ok=True)
    total_bytes = 0
    layer_info = {}

    for name, data in tqdm(meta.items(), desc="Saving compressed"):
        fname = os.path.join(save_dir, name + ".npz")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        codes_out = (_pack_2bit(data["codes"])
                     if bits == 2 and data["shape"][1] % 4 == 0
                     else data["codes"])
        np.savez(fname,
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
    groupsize: int = GROUPSIZE,
    save_compressed_dir: str | None = None,
) -> dict:
    model.eval()
    model.cpu()                                     # everything happens on CPU
    meta: dict = {}

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear)]

    for name, layer in tqdm(layers, desc="Quantizing"):
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
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Precision for loading the base model")
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
    dtype = getattr(torch, args.dtype)

    logger.info("Loading model: %s (dtype=%s)", args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    compressed_dir = os.path.join(args.save, "compressed") if args.save else None

    logger.info("Quantizing  bits=%d  groupsize=%d  symmetric=True",
                args.bits, GROUPSIZE)
    meta = quantize_model(
        model,
        bits=args.bits,
        groupsize=GROUPSIZE,
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
