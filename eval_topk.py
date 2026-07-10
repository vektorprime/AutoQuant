#!/usr/bin/env python
"""Compute argmax (top-1) agreement between quantized and reference logits.

Independent of eval_perplexity.py — uses the same reference cache and
Wikitext-2 test split.  Diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_perplexity import _tokenize_wikitext, _build_window_index, _cache_meta_path


def same_top_p(
    input_ids: torch.Tensor,
    windows: list[tuple[int, int, int]],
    device: torch.device,
    ref_logits_mmap: np.memmap,
    quant_model,
    vocab_size: int,
):
    """Returns (overlap_count, total_tokens, elapsed_seconds)."""
    overlap = 0
    n_tokens = 0
    memmap_offset = 0
    t_start = time.time()

    for begin, end, target_len in tqdm(windows, desc="Argmax agreement"):
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        with torch.inference_mode():
            cached = ref_logits_mmap[memmap_offset:memmap_offset + target_len]
            ref_target = torch.from_numpy(cached.astype(np.float32)).to(device)

            quant_output = quant_model(chunk, use_cache=False).logits
            quant_target = quant_output[0, -(target_len + 1):-1, :]

        # Top-1 / argmax agreement
        ref_argmax = ref_target.argmax(dim=-1)
        q_argmax = quant_target.argmax(dim=-1)
        overlap += (ref_argmax == q_argmax).sum().item()
        n_tokens += target_len
        memmap_offset += target_len

    elapsed = time.time() - t_start
    return overlap, n_tokens, elapsed


def _load_model(model_path: str, device, dtype):
    """Load model — uses packed loader if q4k_manifest.json exists."""
    if os.path.isfile(os.path.join(model_path, "q4k_manifest.json")):
        from inference import load_quantized_model
        model, _ = load_quantized_model(model_path, device=device, dtype=dtype)
        return model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Top-1 argmax agreement between quantized model and reference cache")
    parser.add_argument("--model", required=True, help="Path to quantized model")
    parser.add_argument("--reference", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--reference-cache", default="cache/ref_logits.mmap")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading quantized model: {args.model}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    quant_model = _load_model(args.model, device, dtype)
    vocab_size = quant_model.config.vocab_size

    print(f"Loading reference logits from: {args.reference_cache}")
    meta_path = _cache_meta_path(args.reference_cache)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Reference cache metadata not found: {meta_path}")
    with open(meta_path) as handle:
        cache_meta = json.load(handle)
    if not cache_meta.get("complete", True):
        raise ValueError("Reference cache is marked incomplete")
    if cache_meta.get("reference_model") not in (None, args.reference):
        raise ValueError(
            f"Cache was generated from {cache_meta.get('reference_model')!r}, "
            f"not {args.reference!r}"
        )
    expected = {
        "context_length": args.context_length,
        "stride": args.stride,
        "split": "test",
        "max_tokens": args.max_tokens,
    }
    mismatches = {
        key: (cache_meta.get(key), value)
        for key, value in expected.items()
        if cache_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Reference cache parameters do not match: {mismatches}")
    if cache_meta["vocab_size"] != vocab_size:
        raise ValueError(
            f"Vocab mismatch: model={vocab_size}, cache={cache_meta['vocab_size']}"
        )
    ref_mmap = np.memmap(
        args.reference_cache,
        dtype=np.float16,
        mode="r",
        shape=(cache_meta["total_tokens"], cache_meta["vocab_size"]),
    )
    print(
        f"  {cache_meta['total_tokens']} tokens cached, "
        f"vocab={cache_meta['vocab_size']}"
    )

    print("Loading Wikitext-2 (test split)")
    tokenizer = AutoTokenizer.from_pretrained(args.reference)
    input_ids = _tokenize_wikitext(tokenizer, "test")
    input_ids = input_ids[:args.max_tokens]

    windows = _build_window_index(input_ids, args.context_length, args.stride)
    eval_tokens = sum(window[2] for window in windows)
    if eval_tokens != cache_meta["total_tokens"]:
        raise ValueError(
            f"Cache contains {cache_meta['total_tokens']} target tokens, "
            f"but this run builds {eval_tokens}"
        )
    print(f"  {len(windows)} windows, {eval_tokens} eval tokens")

    overlap, n_tokens, elapsed = same_top_p(
        input_ids, windows, device, ref_mmap, quant_model, vocab_size,
    )
    pct = 100 * overlap / n_tokens if n_tokens > 0 else 0
    print(f"\nArgmax agreement: {overlap}/{n_tokens} = {pct:.3f}%")
    print(f"Tokens/sec: {n_tokens / elapsed:.1f}")


if __name__ == "__main__":
    main()
