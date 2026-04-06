# AutoQuant

inspired by [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy.

![teaser](assets/progress.png)

The idea: give an AI agent a post-training quantization setup and let it experiment autonomously overnight. It modifies the code, quantizes the model, checks if the result improved, keeps or discards, and repeats.


## How it works

The repo is deliberately kept small and only really has five files that matter:
- **quantize.py** — the quantization script with the algorithm
- **quantizer.py** — the quantizer class
- **data_utils.py** — data preparation utilities
- **eval_perplexity.py** — perplexity evaluation script
- **program.md** — the experiment description for the agent

The starting point is vanilla [GPTQ](https://arxiv.org/abs/2210.17323) implementation without any additional tweaks or modifications. The model is quantized using the `quantize.py` script, which is a modified version of the original GPTQ script and saved.
After the quantization, the perplexity is evaluated using the `eval_perplexity.py` script.

The goal of the agent is to achieve as small perplexity as possible for a fixed quantization configuration - (bits, groupsize, symmetric).

## Quick start

* Prepare environment with up-to-date `torch`, `transformers`, and `datasets` packages.

## Running the agent

Prompt something like this:
```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```
During the experiment the agent with ask which model to quantize. In the example provided above, the agent quantizes `Llama-3.1-8B-Instruct`.