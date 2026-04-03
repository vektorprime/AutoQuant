"""Calibration data utilities for GPTQ quantization."""

import random

import numpy as np
import torch
from datasets import load_dataset


def get_wikitext2(
    nsamples: int,
    seqlen: int,
    tokenizer,
    split: str = "train",
) -> list[torch.Tensor]:
    """Random fixed-length chunks from Wikitext-2."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    samples = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.size(0) - seqlen - 1)
        samples.append(input_ids[i : i + seqlen].unsqueeze(0))
    return samples


def get_c4(
    nsamples: int,
    seqlen: int,
    tokenizer,
) -> list[torch.Tensor]:
    """Random fixed-length chunks from a single C4 shard."""
    dataset = load_dataset(
        "allenai/c4",
        "en",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    samples = []
    while len(samples) < nsamples:
        idx = random.randint(0, len(dataset) - 1)
        ids = tokenizer(
            dataset[idx]["text"], return_tensors="pt", truncation=False
        ).input_ids[0]
        if ids.size(0) < seqlen:
            continue
        i = random.randint(0, ids.size(0) - seqlen - 1)
        samples.append(ids[i : i + seqlen].unsqueeze(0))
    return samples


def get_ptb(
    nsamples: int,
    seqlen: int,
    tokenizer,
) -> list[torch.Tensor]:
    """Random fixed-length chunks from Penn Treebank."""
    dataset = load_dataset("ptb_text_only", "penn_treebank", split="train")
    text = "\n".join(dataset["sentence"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    samples = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.size(0) - seqlen - 1)
        samples.append(input_ids[i : i + seqlen].unsqueeze(0))
    return samples


_REGISTRY = {
    "wikitext2": get_wikitext2,
    "c4": get_c4,
    "ptb": get_ptb,
}


def get_calibration_data(
    dataset: str,
    nsamples: int,
    seqlen: int,
    tokenizer,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Return a list of *nsamples* int64 tensors of shape (1, seqlen).

    Args:
        dataset:  One of "wikitext2", "c4", "ptb".
        nsamples: Number of calibration sequences.
        seqlen:   Token length of each sequence (should match the quantization
                  context length).
        tokenizer: HuggingFace tokenizer for the target model.
        seed:     RNG seed for reproducibility.
    """
    if dataset not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(_REGISTRY)}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return _REGISTRY[dataset](nsamples, seqlen, tokenizer)
