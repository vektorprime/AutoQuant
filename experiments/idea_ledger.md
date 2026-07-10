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

## exp-20260710-003

Hypothesis: XOR delta coding of packed codes before zlib would reduce entropy and improve compression
Algorithm family: entropy_coding
Changed code: _save_compressed (xor+zlib path for q4_k_xor_zlib format)
Representation change: XOR-delta bytes instead of raw bytes before zlib
Storage risk: none (lossless compression)
VRAM risk: none
Expected win: better zlib compression from lower-entropy delta stream
Outcome: 434.4 MB — WORSE than raw zlib (423.3 MB). XOR randomizes the byte stream. REVERTED.

## exp-20260710-004

Hypothesis: Q4_K with 4 larger sub-blocks (64 weights) instead of 8 (32 weights) reduces metadata overhead while keeping 4-bit codes
Algorithm family: block_geometry
Changed code: _quantize_one_layer_q4k parameterized with n_sub_blocks, new q4_k_lb format
Representation change: 4 sub-blocks of 64 instead of 8 of 32
Storage risk: saved 6 bytes per superblock (scales/mins halved)
VRAM risk: none
Expected win: ~416 MB at similar quality
Outcome: 411.1 MB but KL=0.1391 (+49%), top-P=79.92% (-3.64pp). Fine-grained scale adaptation is essential. REVERTED.

## exp-20260710-005

Hypothesis: Zlib-compressing both packed codes AND scale/min metadata gives additive savings
Algorithm family: entropy_coding
Changed code: _save_compressed (q4_k_zlib2 path compresses quants + scales_6bit + mins_6bit)
Representation change: zlib on all three large arrays per layer
Storage risk: none (lossless, Q4_K algorithm unchanged)
VRAM risk: none
Expected win: additional 5-10 MB over codes-only zlib
Outcome: KL=0.093508 (same), top-P=83.557% (same), size=408.5 MB (-26.1 MB, -6.0%) — GLOBAL BEST
