# Idea Ledger — Q4_K experiment

## exp-q4k-baseline

Hypothesis: Q4_K (GGML-style 4-bit block quantization) provides baseline to beat
Algorithm family: baseline
Outcome: KL=0.093508, top-P=83.557%, size=434.6 MB — BASELINE

## Invalid experiments (reverted)
- zlib/bz2 post-hoc compression — NOT a quantization technique, just file compression
- XOR delta + zlib — same issue
- Larger sub-blocks (4 vs 8) — quality regression
- 3-bit Q4_K variant — quality regression
- Per-channel bias correction — slight regression
- 5-bit scales/mins — slight regression
