# Idea Ledger (0.8B)

## exp-20260702-011

Hypothesis: Sharing codebooks across K=2 output channels at gs=32 halves metadata (~117→58 MB), getting total under 260 MB while preserving Q8 expressiveness (KL expected near ~3.0)
Algorithm family: codebook
Changed code: _quantize_one_layer — K parameter for channel-shared codebook blocks; _save_compressed — unchanged
Representation change: codebook_q shape from [out, n_groups, 4] to [out/K, n_groups, 4]; codes unchanged
Storage risk: -52 MB (305→253 at gs=32 K=2)
VRAM risk: none (same memory profile)
Expected win: KL near 3.04 (Q8 expressiveness preserved, slight loss from sharing)
Outcome: KL regressed (4.40 vs 4.25 best under cap) — channel sharing loses too much per-channel specificity; different K values may help

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

## exp-20260702-010

Hypothesis: Increasing groupsize from 32 to 64 halves codebook metadata (~100 MB → ~50 MB), bringing Q8 codebook under the 260 MB size cap while retaining k-means expressiveness
Algorithm family: scale_optimization
Changed code: MAX_COMPRESSED_MB constant (1575 → 260)
Representation change: none (same Q8 codebook format, larger groups)
Storage risk: -51 MB (305 → 253.6)
VRAM risk: none
Expected win: fits cap; KL slightly higher due to coarser groups
Outcome: KL=4.25 (fits 253.6 MB under 260 MB cap; first valid Q8 codebook result under cap)

## exp-20260702-013

Hypothesis: Better K-means codebook quality (quantile init, 3 trials × 20 iters, activation-weighted L2 distance, multi-seed) improves reconstruction enough to partially compensate for coarser gs=64 groups, bringing KL closer to gs=32 quality
Algorithm family: codebook
Changed code: _quantize_one_layer — replaced L1 symmetric-init 5-iter K-means with quantile-init 3-trial × 20-iter activation-weighted L2 K-means
Representation change: none (same Q8 codebook format)
Storage risk: none
VRAM risk: none
Expected win: KL from 4.25 toward 3.5 at gs=64
Outcome: KL improved (3.78 vs 4.25, -11.1%) — NEW GLOBAL BEST under 260 MB cap

## exp-20260702-009

Hypothesis: Quantizing codebook values to Q8 (per-layer min/max + uint8 centroids) reduces storage from 430→305 MB while preserving near-lossless fidelity (256 levels vs 16 for 4-bit)
Algorithm family: codebook_compression
Changed code: _quantize_one_layer — add Q8 per-layer codebook quantization + re-gather; _save_compressed — new q8_codebook format
Representation change: codebook stored as uint8 per layer with per-layer float32 cb_min/cb_max (1 byte/centroid vs 2 bytes for bf16)
Storage risk: -125 MB (430→305)
VRAM risk: none (same memory profile)
Expected win: KL very close to 3.36 (256 levels near-lossless for bf16 centroids)
Outcome: KL 3.4775 — slight regression vs bf16 (3.36, +3.5%) but far better than 4-bit (7.23). The re-gather from Q8-perturbed centroids introduces minor extra error; recomputing assignments against dequantized codebook may close the gap.

## exp-20260702-014

Hypothesis: Sorting input channels by activation importance before grouping creates more homogeneous groups, enabling K-means centroids to better represent weight distributions and reducing reconstruction error
Algorithm family: activation_weighted
Changed code: _quantize_one_layer — sort W columns + act_stats by activation importance at start, unsort W_q_full before saving
Representation change: none (same Q8 codebook format, same storage)
Storage risk: none
VRAM risk: none (sort indices are tiny per-layer)
Expected win: KL from 3.78 toward 3.5 by making groups more homogeneous
Outcome: KL 3.5352 (-6.4% vs 3.78) — NEW GLOBAL BEST under 260 MB cap. Channel reordering makes K-means groups more homogeneous, allowing centroids to better capture weight distributions at gs=64 granularity.

## exp-20260702-016

Hypothesis: After K-means quantization with non-symmetric centroids, the per-channel quantization error has a DC bias. Subtracting `mean(W_orig - W_q, dim=-1)` from the dequantized weights removes systematic per-channel shift and improves reconstruction.
Algorithm family: codebook
Changed code: _quantize_one_layer — clone W_orig before loop, add per-channel bias correction after Q8 re-gather
Representation change: none (bias baked into dequantized weights, zero extra storage)
Storage risk: none
VRAM risk: +1 float32 clone per layer temporarily (max ~1.2 GB for lm_head)
Expected win: lower KL by removing systematic per-channel quantization bias
Outcome: KL improved (3.4614 vs 3.5352 best, -2.1%) — NEW GLOBAL BEST under 260 MB cap

## exp-20260702-012

Hypothesis: Per-layer groupsize (attn=32, mlp=64) already implemented by _get_layer_groupsize; explicit run to confirm it was active in exp-010
Algorithm family: layer_policy
Changed code: none (code already implements this via _get_layer_groupsize with groupsize=64)
Representation change: none
Storage risk: none (same 253.6 MB as exp-010)
VRAM risk: none
Expected win: same KL as exp-010 (~4.25)
Outcome: KL=4.248302 — bit-identical to exp-010, confirming _get_layer_groupsize was already active. Per-layer groupsize alone does not beat uniform gs=32 (KL=3.04), and gs=32 everywhere exceeds 260 MB cap.

