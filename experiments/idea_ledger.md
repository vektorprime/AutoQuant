# Idea Ledger — Q4_K experiment

## exp-q4k-baseline

Hypothesis: Q4_K (GGML-style 4-bit block quantization) provides baseline to beat
Algorithm family: baseline
Outcome: KL=0.093508, top-P=83.557%, size=434.6 MB — BASELINE

## exp-20260709-002

Hypothesis: Hadamard rotation before Q3_K quantization reduces weight distribution non-uniformity, enabling 3-bit to match Q4_K quality
Algorithm family: rotation+quantization
Changed code: _fwht _quantize_one_layer_q3k_hadamard _pack_3bit
Representation change: 3-bit block quantization with random Hadamard pre-rotate
Storage risk: none (smaller than Q4_K at 331 MB vs 434.6 MB)
VRAM risk: none (FWHT clone overhead is per-layer, freed immediately)
Expected win: lower size with quality preserved
Outcome: KL=0.511 (regressed 5.5x), top-P=62.07% (regressed 21.5pp) — FWHT bug fixed but quality too poor at 3-bit

## Invalid experiments (reverted)
- zlib/bz2 post-hoc compression — NOT a quantization technique, just file compression
- XOR delta + zlib — same issue
- Larger sub-blocks (4 vs 8) — quality regression
- 3-bit Q4_K variant — quality regression
- Per-channel bias correction — slight regression
- 5-bit scales/mins — slight regression
- Hadamard-rotated Q3_K — severe quality regression (KL 0.511, top-P 62.07%)
