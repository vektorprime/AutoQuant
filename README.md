# AutoQuant

inspired by [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy.

![teaser](assets/progress.png)

The idea: give an AI agent a post-training quantization setup and let it experiment autonomously overnight. It modifies the code, quantizes the model, checks if the result improved, keeps or discards, and repeats.


## How it works

The repo is deliberately kept small and only really has five files that matter:
- **quantize.py** — the quantization script with the algorithm
- **quantizer.py** — the quantizer class
- **data_utils.py** — data preparation utilities
- **eval_perplexity.py** — KL divergence evaluation script
- **program.md** — the experiment description for the agent

The starting point is q2_k (2-bit symmetric group quantization) — a simple per-group
min/max approach without calibration data or iterative optimisation. The agent is free
to explore other quantization algorithms — it is not limited to a specific method.
The model is quantized using the `quantize.py` script and saved.
After quantization, KL divergence (lower = better) is evaluated against the reference
model using `eval_perplexity.py`.

The goal of the agent is to achieve the lowest possible KL divergence for a fixed
quantization configuration — 2-bit, groupsize 128, symmetric.

## Quick start

* Prepare environment with up-to-date `torch`, `transformers`, and `datasets` packages.

## Running the agent

Prompt something like this:
```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```
During the experiment the agent with ask which model to quantize. In the example provided above, the agent quantizes `Llama-3.1-8B-Instruct`.