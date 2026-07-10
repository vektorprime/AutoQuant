# Idea Ledger — Q4_K experiment

## exp-q4k-baseline

Hypothesis: Q4_K (GGML-style 4-bit block quantization) provides baseline to beat
Algorithm family: baseline
Outcome: KL=0.093508, top-P=83.557%, size=434.6 MB — BASELINE

## exp-20260709-002

Hypothesis: Hadamard rotation before Q3_K quantization reduces weight distribution non-uniformity enabling 3-bit to match Q4_K quality
Algorithm family: rotation+quantization
Outcome: KL=0.511 regressed 5.5x, top-P=62.07% regressed 21.5pp

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
