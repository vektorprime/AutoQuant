# AutoQuant

Inspired by [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy.

AutoQuant is an autonomous quantization research system. An AI agent modifies the quantization
algorithm, measures the result against a reference baseline, and iterates to discover novel
quantization techniques.

## Current Experiment: Q4_K

The goal is to discover a novel quantization technique that outperforms Q4_K — the industry
standard 4-bit group quantization format. The target technique must be:

1. **Smaller** than Q4_K in compressed storage size
2. **Same or better** KL divergence against FP16 reference
3. **Same or better** same-top-P agreement (argmax overlap with FP16 reference)

The baseline is Q4_K applied to Qwen/Qwen3.5-0.8B. The agent freely explores any
quantization scheme (2-bit, 3-bit, mixed precision, novel encoding, etc.) as long as
the compressed size is strictly smaller than Q4_K's.

## Files

- **quantize.py** — The quantization script (editable by the agent)
- **quantizer.py** — Reference Quantizer class (read-only)
- **data_utils.py** — Calibration data utilities (read-only)
- **eval_perplexity.py** — KL divergence evaluation (read-only)
- **eval_topk.py** — Same-top-P agreement evaluation (read-only)
- **program.md** — The experiment protocol for the AI agent

## Quick start

Prepare environment with up-to-date `torch`, `transformers`, and `datasets` packages.

## Running the agent

Prompt: "Read program.md and let's start a new experiment."
