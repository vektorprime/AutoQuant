# Idea Ledger — Q4_K experiment

## exp-q4k-baseline

Hypothesis: Q4_K (GGML-style 4-bit block quantization) provides a strong quality baseline at ~435 MB
Algorithm family: baseline
Changed code: _quantize_one_layer_q4k + _save_compressed + quantize_model additions
Representation change: 4-bit weights per superblock (256), 6-bit scale/min per sub-block (32)
Storage risk: 434.6 MB (target baseline to beat)
VRAM risk: 0 MB (CPU quantization)
Expected win: establish quality bar for future experiments
Outcome: KL=0.093508, top-P=83.557%, size=434.6 MB — BASELINE RECORDED

## exp-20260710-002

Hypothesis: Applying zlib (DEFLATE) entropy coding to Q4_K's packed 4-bit code arrays reduces storage size with zero quality change
Algorithm family: entropy_coding
Changed code: _save_compressed (zlib compression path for q4_k_zlib format)
Representation change: zlib-compressed packed codes instead of raw uint8 arrays
Storage risk: none (compression only, codes decompress to identical values)
VRAM risk: none (CPU-only modification)
Expected win: smaller size at identical quality
Outcome: KL=0.093508 (same), top-P=83.557% (same), size=423.3 MB (-11.3 MB, -2.6%) — GLOBAL BEST
