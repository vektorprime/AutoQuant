#!/usr/bin/env python3
"""Evaluate KL divergence between a reference model and a quantized model on Wikitext-2."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _tokenize_wikitext(tokenizer, split: str):
    print(f"Loading Wikitext-2 ({split} split)")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids[0]


def _build_window_index(input_ids, context_length: int, stride: int):
    """Return list of (begin, end, target_len) for sliding windows."""
    seq_len = input_ids.size(0)
    windows = []
    prev_end = 0
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + context_length, seq_len)
        target_len = min(end - max(begin, prev_end), end - begin - 1)
        if target_len > 0:
            windows.append((begin, end, target_len))
        prev_end = end
        if end == seq_len:
            break
    return windows


def _compute_kl_divergence(
    input_ids,
    windows,
    device,
    ref_model=None,
    ref_logits_mmap=None,
    ref_logits_mmap_fill=False,
    quant_model=None,
):
    """Core KL computation. Returns (total_kl, n_tokens, elapsed_seconds)."""
    total_kl = 0.0
    n_tokens = 0
    memmap_offset = 0
    t_start = time.time()

    for begin, end, target_len in tqdm(windows, desc="Evaluating KL"):
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        with torch.no_grad():
            if ref_logits_mmap is not None and not ref_logits_mmap_fill:
                cached = ref_logits_mmap[memmap_offset:memmap_offset + target_len]
                ref_target = torch.from_numpy(cached.astype(np.float32)).to(device)
            else:
                ref_output = ref_model(chunk).logits
                ref_target = ref_output[0, -(target_len + 1):-1, :]
                if ref_logits_mmap is not None and ref_logits_mmap_fill:
                    ref_logits_mmap[memmap_offset:memmap_offset + target_len] = (
                        ref_target.cpu().numpy().astype(np.float16)
                    )

            quant_output = quant_model(chunk).logits
            quant_target = quant_output[0, -(target_len + 1):-1, :]

        ref_probs = F.softmax(ref_target.float(), dim=-1)
        quant_log_probs = F.log_softmax(quant_target.float(), dim=-1)
        kl = F.kl_div(quant_log_probs, ref_probs, reduction="sum")
        total_kl += kl.item()
        n_tokens += target_len
        memmap_offset += target_len

    elapsed = time.time() - t_start if n_tokens > 0 else 0.0
    return total_kl, n_tokens, elapsed


def evaluate_kl_divergence(
    reference_model_name: str,
    quantized_model_name: str,
    context_length: int,
    stride: int | None = None,
    device: str | None = None,
    split: str = "test",
    reference_cache: str | None = None,
    max_tokens: int | None = None,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if stride is None:
        stride = context_length // 2

    dtype = torch.float16 if device == "cuda" else torch.float32
    same_ref_and_quant = (quantized_model_name == reference_model_name)

    # ------------------------------------------------------------------
    # Cache hit path: skip loading the reference model entirely
    # ------------------------------------------------------------------
    if reference_cache is not None and os.path.exists(reference_cache):
        meta_path = reference_cache.replace(".mmap", ".json")
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta["context_length"] != context_length or meta["stride"] != stride
                or meta["split"] != split or meta.get("max_tokens") != max_tokens):
            raise ValueError(
                f"Cache parameters mismatch. Cache: ctx={meta['context_length']} "
                f"stride={meta['stride']} split={meta['split']} max_tokens={meta.get('max_tokens')}. "
                f"Requested: ctx={context_length} stride={stride} split={split} max_tokens={max_tokens}"
            )

        print(f"Using cached reference logits: {reference_cache}")
        ref_tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
        input_ids = _tokenize_wikitext(ref_tokenizer, split)
        if max_tokens is not None:
            input_ids = input_ids[:max_tokens]
        windows = _build_window_index(input_ids, context_length, stride)

        ref_logits_mmap = np.memmap(
            reference_cache, dtype=np.float16, mode="r",
            shape=(meta["total_tokens"], meta["vocab_size"]),
        )

        print(f"Loading quantized model: {quantized_model_name}")
        quant_model = AutoModelForCausalLM.from_pretrained(quantized_model_name, torch_dtype=dtype)
        quant_model.eval()
        quant_model.to(device)

        total_kl, n_tokens, elapsed = _compute_kl_divergence(
            input_ids, windows, device,
            ref_logits_mmap=ref_logits_mmap,
            ref_logits_mmap_fill=False,
            quant_model=quant_model,
        )

        mean_kl = total_kl / n_tokens if n_tokens > 0 else float("inf")
        tok_s = n_tokens / elapsed if elapsed > 0 else 0
        print(f"\nMean KL divergence (ref → quant): {mean_kl:.6f}")
        print(f"Tokens/sec: {tok_s:.1f}")
        return mean_kl

    # ------------------------------------------------------------------
    # No cache or cache miss: load reference model, optionally cache
    # ------------------------------------------------------------------
    print(f"Loading reference model: {reference_model_name}")
    ref_tokenizer = AutoTokenizer.from_pretrained(reference_model_name, use_fast=True)
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_name, torch_dtype=dtype)
    ref_model.eval()
    ref_model.to(device)

    input_ids = _tokenize_wikitext(ref_tokenizer, split)
    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]
    windows = _build_window_index(input_ids, context_length, stride)

    vocab_size = ref_model.config.vocab_size
    total_tokens = sum(w[2] for w in windows)

    # Setup memmap for writing if caching
    ref_logits_mmap = None
    ref_logits_mmap_fill = False
    if reference_cache is not None:
        print(f"Caching reference logits to: {reference_cache}")
        ref_logits_mmap = np.memmap(
            reference_cache, dtype=np.float16, mode="w+",
            shape=(total_tokens, vocab_size),
        )
        ref_logits_mmap_fill = True
        # Save metadata eagerly so a crash preserves cache validity
        meta_path = reference_cache.replace(".mmap", ".json")
        with open(meta_path, "w") as f:
            json.dump({
                "vocab_size": vocab_size,
                "total_tokens": total_tokens,
                "context_length": context_length,
                "stride": stride,
                "split": split,
                "max_tokens": max_tokens,
            }, f)

    # Load quantized model
    if same_ref_and_quant:
        print("Quantized model is same as reference — sharing model instance")
        quant_model = ref_model
    else:
        print(f"Loading quantized model: {quantized_model_name}")
        quant_model = AutoModelForCausalLM.from_pretrained(quantized_model_name, torch_dtype=dtype)
        quant_model.eval()
        quant_model.to(device)

    print(f"Tokens: {input_ids.size(0):,} | Context length: {context_length} | Stride: {stride}")

    total_kl, n_tokens, elapsed = _compute_kl_divergence(
        input_ids, windows, device,
        ref_model=ref_model,
        ref_logits_mmap=ref_logits_mmap,
        ref_logits_mmap_fill=ref_logits_mmap_fill,
        quant_model=quant_model,
    )

    if ref_logits_mmap is not None:
        ref_logits_mmap.flush()
        print(f"Reference logits cached ({total_tokens:,} tokens, {vocab_size:,} vocab)")

    mean_kl = total_kl / n_tokens if n_tokens > 0 else float("inf")
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    print(f"\nMean KL divergence (ref → quant): {mean_kl:.6f}")
    print(f"Tokens/sec: {tok_s:.1f}")
    return mean_kl


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KL divergence between reference and quantized model on Wikitext-2"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B",
                        help="Path to the quantized model to evaluate")
    parser.add_argument("--reference", default="Qwen/Qwen3.5-2B",
                        help="Path to the reference (original) model")
    parser.add_argument("--context-length", type=int, required=True,
                        help="Context window size (number of tokens)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (default: context_length // 2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu (default: auto-detect)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Dataset split (default: test)")
    parser.add_argument("--reference-cache", type=str, default=None,
                        help="Path to cache reference logits (.mmap file). "
                             "On cache hit the reference model is skipped entirely.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Truncate dataset to first N tokens (default: full dataset)")
    args = parser.parse_args()

    evaluate_kl_divergence(
        reference_model_name=args.reference,
        quantized_model_name=args.model,
        context_length=args.context_length,
        stride=args.stride,
        device=args.device,
        split=args.split,
        reference_cache=args.reference_cache,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
