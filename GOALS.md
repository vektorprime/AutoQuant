# Q4_K
## Goal: Find a novel quantization technique that is SMALLER than Q4_K (434.6 MB) while matching or beating its KL (≤0.0935) and top-P (≥83.56%)
## Model: Qwen/Qwen3.5-0.8B
## Baseline recorded — experiments completed
## Best so far: Q4_K + zlib on codes+metadata (q4_k_zlib2) at 408.5 MB — KL=0.0935, top-P=83.56%
## 9 experiments run: 2 improvements, 5 regressions, 1 crash, 1 baseline
## Remaining directions: Hadamard rotation, vector quantization, AQLM-style additive codebooks
