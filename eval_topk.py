#!/usr/bin/env python
"""Compute Same-top-P overlap: percentage of positions where the quantized model's
argmax token matches the reference model's argmax.  Mirrors llama.cpp's metric.

Independent of eval_perplexity.py — uses the same reference cache and
Wikitext-2 test split.  Diagnostic only.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_perplexity import _tokenize_wikitext, _build_window_index


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

    for begin, end, target_len in tqdm(windows, desc="Same top-p"):
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        with torch.no_grad():
            cached = ref_logits_mmap[
                memmap_offset * vocab_size:(memmap_offset + target_len) * vocab_size
            ]
            ref_target = torch.from_numpy(
                cached.astype(np.float32).reshape(target_len, vocab_size)
            ).to(device)

            quant_output = quant_model(chunk).logits
            quant_target = quant_output[0, -(target_len + 1):-1, :]

        # Same top-p: argmax agreement
        ref_argmax = ref_target.argmax(dim=-1)
        q_argmax = quant_target.argmax(dim=-1)
        overlap += (ref_argmax == q_argmax).sum().item()
        n_tokens += target_len
        memmap_offset += target_len

    elapsed = time.time() - t_start
    return overlap, n_tokens, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Same-top-P: argmax overlap between quantized model and reference cache")
    parser.add_argument("--model", required=True, help="Path to quantized model")
    parser.add_argument("--reference", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--reference-cache", default="cache/ref_logits.mmap")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading quantized model: {args.model}")
    quant_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    quant_model.eval()
    vocab_size = quant_model.config.vocab_size

    print(f"Loading reference logits from: {args.reference_cache}")
    ref_mmap = np.memmap(
        args.reference_cache,
        dtype=np.float16,
        mode="r",
    )
    n_cached = ref_mmap.shape[0] // vocab_size
    print(f"  {n_cached} tokens cached, vocab={vocab_size}")

    print("Loading Wikitext-2 (test split)")
    tokenizer = AutoTokenizer.from_pretrained(args.reference)
    input_ids = _tokenize_wikitext(tokenizer, "test")
    input_ids = input_ids[:args.max_tokens]

    windows = _build_window_index(input_ids, args.context_length, args.stride)
    print(f"  {len(windows)} windows, {sum(w[2] for w in windows)} eval tokens")

    overlap, n_tokens, elapsed = same_top_p(
        input_ids, windows, device, ref_mmap, quant_model, vocab_size,
    )
    pct = 100 * overlap / n_tokens if n_tokens > 0 else 0
    print(f"\nSame top-p: {overlap}/{n_tokens} = {pct:.3f}%")
    print(f"Tokens/sec: {n_tokens / elapsed:.1f}")


if __name__ == "__main__":
    main()
