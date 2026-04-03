#!/usr/bin/env python3
"""Evaluate perplexity on Wikitext-2 for a causal language model."""

import argparse
import math

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_perplexity(
    model_name: str,
    context_length: int,
    stride: int | None = None,
    batch_size: int = 1,
    device: str | None = None,
    split: str = "test",
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if stride is None:
        stride = context_length // 2

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.eval()
    model.to(device)

    print(f"Loading Wikitext-2 ({split} split)")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    seq_len = input_ids.size(0)

    print(f"Tokens: {seq_len:,} | Context length: {context_length} | Stride: {stride}")

    nlls = []
    prev_end = 0

    windows = list(range(0, seq_len - 1, stride))
    for begin in tqdm(windows, desc="Evaluating"):
        end = min(begin + context_length, seq_len)
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        # Only compute loss on the new tokens (avoid double-counting overlapping context)
        target_len = min(end - max(begin, prev_end), chunk.size(1) - 1)
        if target_len <= 0:
            prev_end = end
            continue

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)

        # Re-derive loss for only the target tokens
        logits = outputs.logits  # (1, seq, vocab)
        shift_logits = logits[0, -(target_len + 1):-1, :]  # (target_len, vocab)
        shift_labels = chunk[0, -target_len:]               # (target_len,)

        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="sum")
        nlls.append(loss.item())
        prev_end = end

        if end == seq_len:
            break

    total_nll = sum(nlls)
    n_tokens = min(seq_len - 1, len(windows) * stride + context_length - stride)
    # More accurate: count the actual evaluated tokens
    n_tokens = seq_len - 1  # approximate; stride windows cover the full sequence
    ppl = math.exp(total_nll / (seq_len - 1))

    print(f"\nPerplexity: {ppl:.4f}")
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on Wikitext-2")
    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument("--context-length", type=int, required=True, help="Context window size (number of tokens)")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride (default: context_length // 2)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu (default: auto-detect)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split (default: test)")
    args = parser.parse_args()

    evaluate_perplexity(
        model_name=args.model,
        context_length=args.context_length,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
        split=args.split,
    )


if __name__ == "__main__":
    main()
