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


def _cache_meta_path(cache_path: str) -> str:
    root, _ = os.path.splitext(cache_path)
    return root + ".json"


def _load_model(model_path: str, device: str | torch.device, dtype):
    """Load a packed checkpoint when a manifest is present."""
    if os.path.isfile(os.path.join(model_path, "q4k_manifest.json")):
        from inference import load_quantized_model

        model, _ = load_quantized_model(model_path, device=device, dtype=dtype)
        return model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.eval()
    model.to(device)
    return model


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
    """Compute KL or fill a reference cache when quant_model is None."""
    total_kl = 0.0
    n_tokens = 0
    memmap_offset = 0
    t_start = time.time()

    for begin, end, target_len in tqdm(windows, desc="Evaluating KL"):
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        with torch.inference_mode():
            if ref_logits_mmap is not None and not ref_logits_mmap_fill:
                cached = ref_logits_mmap[memmap_offset:memmap_offset + target_len]
                ref_target = torch.from_numpy(cached.astype(np.float32)).to(device)
            else:
                if ref_model is None:
                    raise RuntimeError("Reference model is required to fill logits")
                ref_output = ref_model(chunk, use_cache=False).logits
                ref_target = ref_output[0, -(target_len + 1):-1, :]
                if ref_logits_mmap is not None and ref_logits_mmap_fill:
                    ref_logits_mmap[memmap_offset:memmap_offset + target_len] = (
                        ref_target.float().cpu().numpy().astype(np.float16)
                    )

            if quant_model is not None:
                quant_output = quant_model(chunk, use_cache=False).logits
                quant_target = quant_output[0, -(target_len + 1):-1, :]

        if quant_model is not None:
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

    torch_device = torch.device(device)
    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    device = str(torch_device)
    same_ref_and_quant = (quantized_model_name == reference_model_name)

    # ------------------------------------------------------------------
    # Cache hit path: skip loading the reference model entirely
    # ------------------------------------------------------------------
    cache_ready = False
    meta = None
    if reference_cache is not None and os.path.exists(reference_cache):
        meta_path = _cache_meta_path(reference_cache)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Reference cache exists without metadata: {meta_path}"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("complete", True):
            print("Discarding incomplete reference-logit cache")
            os.remove(reference_cache)
            os.remove(meta_path)
            meta = None
        else:
            cache_ready = True

    if cache_ready:
        if meta.get("reference_model") not in (None, reference_model_name):
            raise ValueError(
                f"Cache was generated from {meta.get('reference_model')!r}, "
                f"not {reference_model_name!r}"
            )
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
        eval_tokens = sum(window[2] for window in windows)
        if eval_tokens != meta["total_tokens"]:
            raise ValueError(
                f"Cache contains {meta['total_tokens']} target tokens, "
                f"but this run builds {eval_tokens}"
            )

        ref_logits_mmap = np.memmap(
            reference_cache, dtype=np.float16, mode="r",
            shape=(meta["total_tokens"], meta["vocab_size"]),
        )

        print(f"Loading quantized model: {quantized_model_name}")
        quant_model = _load_model(quantized_model_name, device, dtype)
        if quant_model.config.vocab_size != meta["vocab_size"]:
            raise ValueError(
                f"Vocab mismatch: model={quant_model.config.vocab_size}, "
                f"cache={meta['vocab_size']}"
            )

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

    ref_logits_mmap = None
    ref_logits_mmap_fill = False
    if reference_cache is not None:
        print(f"Caching reference logits to: {reference_cache}")
        os.makedirs(os.path.dirname(reference_cache) or ".", exist_ok=True)
        ref_logits_mmap = np.memmap(
            reference_cache, dtype=np.float16, mode="w+",
            shape=(total_tokens, vocab_size),
        )
        ref_logits_mmap_fill = True
        meta_path = _cache_meta_path(reference_cache)
        with open(meta_path, "w") as f:
            json.dump({
                "complete": False,
                "reference_model": reference_model_name,
                "logits_dtype": "float16",
                "vocab_size": vocab_size,
                "total_tokens": total_tokens,
                "context_length": context_length,
                "stride": stride,
                "split": split,
                "max_tokens": max_tokens,
            }, f, indent=2)

    # Phase 1: generate reference cache (ref model only)
    if ref_logits_mmap_fill:
        print("Generating reference logits...")
        _compute_kl_divergence(
            input_ids, windows, device,
            ref_model=ref_model,
            ref_logits_mmap=ref_logits_mmap,
            ref_logits_mmap_fill=True,
            quant_model=None,
        )
        ref_logits_mmap.flush()
        with open(meta_path) as handle:
            completed_meta = json.load(handle)
        completed_meta["complete"] = True
        temp_meta_path = meta_path + ".tmp"
        with open(temp_meta_path, "w") as handle:
            json.dump(completed_meta, handle, indent=2)
        os.replace(temp_meta_path, meta_path)
        print(f"Reference logits cached ({total_tokens:,} tokens, {vocab_size:,} vocab)")
        # Free ref model before loading quant
        del ref_model
        import gc; gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        ref_model = None
        ref_logits_mmap = np.memmap(
            reference_cache, dtype=np.float16, mode="r",
            shape=(total_tokens, vocab_size),
        )

    # Phase 2: load quantized model and evaluate
    if same_ref_and_quant:
        print("Quantized model is same as reference — reusing")
        quant_model = _load_model(reference_model_name, device, dtype)
    else:
        print(f"Loading quantized model: {quantized_model_name}")
        quant_model = _load_model(quantized_model_name, device, dtype)

    print(f"Tokens: {input_ids.size(0):,} | Context length: {context_length} | Stride: {stride}")

    if ref_logits_mmap is not None:
        total_kl, n_tokens, elapsed = _compute_kl_divergence(
            input_ids, windows, device,
            ref_logits_mmap=ref_logits_mmap,
            ref_logits_mmap_fill=False,
            quant_model=quant_model,
        )
    else:
        total_kl, n_tokens, elapsed = _compute_kl_divergence(
            input_ids, windows, device,
            ref_model=ref_model,
            ref_logits_mmap=ref_logits_mmap,
            ref_logits_mmap_fill=ref_logits_mmap_fill,
            quant_model=quant_model,
        )

    mean_kl = total_kl / n_tokens if n_tokens > 0 else float("inf")
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    print(f"\nMean KL divergence (ref → quant): {mean_kl:.6f}")
    print(f"Tokens/sec: {tok_s:.1f}")
    return mean_kl


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KL divergence between reference and quantized model on Wikitext-2"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="Path to the quantized model to evaluate")
    parser.add_argument("--reference", default="Qwen/Qwen3.5-0.8B",
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
