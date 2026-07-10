# Idea Ledger — Q4_K experiment

## exp-q4k-baseline

Hypothesis: Q4_K (GGML-style 4-bit block quantization) provides baseline to beat
Algorithm family: baseline
Outcome: KL=0.093508, top-P=83.557%, size=434.6 MB — BASELINE

## exp-20260709-002

Hypothesis: Hadamard rotation before Q3_K quantization reduces weight distribution non-uniformity enabling 3-bit to match Q4_K quality
Algorithm family: rotation+quantization
Outcome: KL=0.511 regressed 5.5x, top-P=62.07% regressed 21.5pp

## exp-20260709-003

Hypothesis: Hadamard-rotated Q4_K will make weight distribution uniform improving Q4_K block quantization
Algorithm family: rotation+quantization
Outcome: KL=0.435 regressed 4.7x, top-P=not evaluated (KL > 0.09, experiment failed)

## Invalid experiments (reverted)
- zlib/bz2 post-hoc compression — NOT a quantization technique, just file compression
- XOR delta + zlib — same issue
- Larger sub-blocks (4 vs 8) — quality regression
- 3-bit Q4_K variant — quality regression
- Per-channel bias correction — slight regression
- 5-bit scales/mins — slight regression
- Hadamard-rotated Q3_K — severe quality regression (KL 0.511)
- AQLM dual 2-bit codebook — catastrophic quality (KL 4.51), too slow (3.5 min)
- Q4_K sub4 (4 sub-blocks of 64) — KL 0.139, top-P 79.92% both regressed
- Hadamard-rotated Q4_K (256×256 Hadamard per block) — KL 0.435 massive regression
- Q4_K symmetric (no dmin/offset) — KL 0.523 massive regression, 405.2 MB
- NF4 (QLoRA levels, groupsize=64, 4.25 bpw) — KL 0.475 massive regression, 408.6 MB
- Q4_K quantile clipping (5%/95% percentiles instead of min/max) — KL 1.078 worst yet
- Q4_K sub4 + LS refinement (4 sub-blocks of 64, LS-optimized d/dmin, 4.31 bpw) — KL 0.511, 411.1 MB

## exp-20260710-004

Hypothesis: Vector quantization — global 256-entry 4-vector codebook trained once on first layer and shared across all layers, 2.03 bpw
Algorithm family: vector-quantization
Outcome: KL=6.162 catastrophic regression 66x, top-P=6.36% regressed 77.2pp, size=189.0 MB (smaller but unusable quality). VQ with per-block codebooks too slow (>5min). Global codebook fails because weight distributions vary across layers. Time: 7:20 > 5min limit.

## exp-20260710-005

Hypothesis: Groups of 16 output channels share 16 optimized quantization levels (1D K-means trained) → 4.03 bpw vs Q4_K 4.5 bpw
Algorithm family: shared-quantization-levels
Outcome: KL=0.437 regressed 4.7x (vs 0.093), top-P=65.23% regressed 18.3pp, size=381.6 MB (smaller but not matching quality). Sharing levels across channels degrades per-channel precision too much. Time: 8:19 > 5min.
