# Idea Ledger (0.8B)

## exp-20260702-003

Hypothesis: Different layer types benefit from different error diffusion coefficients — attention layers need stronger diffusion to compensate for structured weight patterns
Algorithm family: layer_policy
Changed code: _quantize_one_layer (diffusion param) + quantize_model (per-layer-type dispatch)
Representation change: none
Storage risk: none
VRAM risk: none
Expected win: lower KL
Outcome: KL improved (5.6339 vs 6.0186 baseline)

## exp-20260702-005

Hypothesis: Refining group scales globally (against original weights) after per-group error-diffusion quantization reduces reconstruction error because the scales from the sequential loop were optimized for error-diffused weights, not original weights.
Algorithm family: scale_optimization
Changed code: _quantize_one_layer — added W_orig clone before loop, vectorized global scale refinement after loop
Representation change: none
Storage risk: none
VRAM risk: none
Expected win: lower KL
Outcome: KL improved (6.02 → 5.65, -6.1%)

## exp-20260702-006

Hypothesis: Removing activation-weighted error diffusion (inverse eps_w) would improve KL by avoiding biased over-amplification of low-importance channels during error propagation
Algorithm family: error_compensation
Changed code: _quantize_one_layer — removed eps_w weighting in error diffusion block
Representation change: none
Storage risk: none
VRAM risk: none
Expected win: lower KL
Outcome: KL regressed (5.91 vs 5.63 best, +4.9%)

## exp-20260702-008

Hypothesis: Quantizing the codebook values themselves to 4-bit (16 centroids per layer via uniform quantiles) reduces storage from 8 bytes/group to 2 bytes/group while preserving most of the codebook's expressiveness
Algorithm family: codebook_compression
Changed code: _quantize_one_layer — added _quantize_codebook_values post-processing; _save_compressed — new format for codebook_indices + codebook_table
Representation change: codebook values quantized to 4-bit (16 per-layer centroids, packed uint16 indices per group)
Storage risk: -187 MB (430→243), target met at 253.6 MB
VRAM risk: none (same memory profile)
Expected win: similar KL to full codebook (3.36) with much lower storage
Outcome: KL massively regressed (7.23 vs 3.36 best, +115%) — 4-bit codebook quantization destroys the expressiveness gained from k-means

## exp-20260702-007

Hypothesis: Learning explicit per-group 4-value codebooks via k-means gives more expressive quantization levels than the hardcoded {-2s, -s, 0, s} formula, reducing reconstruction error
Algorithm family: codebook
Changed code: _quantize_one_layer — replaced scale-based levels with k-means learned 4-value codebook per (out_channel, group); _save_compressed — stores codebook instead of scales
Representation change: scales replaced by 4 bf16 codebook values per group (8 bytes/group vs 2 bytes/group)
Storage risk: +206 MB (scales 2→8 bytes/group); total 430 MB, well under 1575 MB limit
VRAM risk: none (same memory profile, k-means iterates within single-group view)
Expected win: lower KL (more expressive quantization levels)
Outcome: KL massively improved (3.3595 vs 5.6339 best, -40.4%) — NEW GLOBAL BEST

