# Idea Ledger

## q4k-9b-asym64 — Asymmetric precision: 6-bit scales + 4-bit mins + fp16 d/dmin

**Hypothesis:** Pushing mins further to 4-bit preserves quality since mins (additive offsets) are far less sensitive than scales (multiplicative factors) for Top-P.
**Status:** regression (near-pass)
**KL divergence:** 0.053406  |  **Top-P:** 87.597%  |  **Size:** 6.45 GB

**Implementation:**
- Extended `sm_bits_min` choices to include 4 (8×4-bit mins in 4 bytes + 8×6-bit scales in 6 bytes = 10 bytes/superblock)
- Added `_pack_values_4bit()` for 8×4-bit → 4 bytes packing
- Updated `decode_q4k()` and `QuantizedLinear._unpack_scale_min_rows()` with sm_last=10 branch
- Shape validation updated to accept last dim 10, 11, or 12
- 4-bit mins use 15 levels (vs 31 for 5-bit, 63 for 6-bit)
- Stacked with --q4k-ddmin-fp16 for cumulative savings

**Result:**
Top-P regressed from 88.17% (asym65) to 87.60% (-0.57%), below the 88.0% threshold by 0.4 percentage points. KL actually IMPROVED significantly (0.0597→0.0534), suggesting coarser mins act as regularization. The near-pass at 6.45 GB (saves 31 MB vs asym65) suggests 4-bit mins are borderline viable; further quality improvements (e.g., stochastic rounding, per-layer bit allocation) might push it over the threshold.

**Lesson:**
Mins at 4-bit precision are just below the quality threshold for Top-P. The asymmetric hypothesis (mins tolerate lower precision than scales) holds directionally but has limits. The KL improvement suggests that coarser mins may actually help distribution matching by smoothing outliers. Future experiments should consider per-layer adaptive min precision (4-bit for deep layers, 5-bit for early layers) or non-linear encoding that preserves more precision where it matters.

---

## q4k-9b-dmindrop — Per-superblock dmin elimination

**Hypothesis:** dmin can be derived from d via learned per-channel factor alpha, saving half the d/dmin storage while LS re-optimization absorbs the constraint.
**Status:** regression
**KL divergence:** 0.108213  |  **Top-P:** 83.396%  |  **Size:** 6.42 GB

**Implementation:**
- After LS solve: compute per-channel alpha = median(dmin/d), set dmin = alpha*d
- Reassign codes with constrained dmin, solve for d only (single-parameter LS)
- Store dmin_alphas as per-output-channel tensor instead of per-superblock dmin
- QuantizedLinear loads dmin_alphas and computes dmin = alpha * d at inference time

**Result:**
Both per-layer and per-channel alpha approaches failed catastrophically (KL +84%, Top-P -4.8%). The d-only LS cannot compensate for the dmin constraint because dmin and d appear in the reconstruction as `d*(s*q - alpha*m)` — changing d scales both the scale and min terms proportionally, so the relative error cannot be corrected. The dmin/d ratio varies significantly even within the same output channel across superblocks.

**Lesson:**
The dmin elimination approach is fundamentally flawed for this encoding scheme. d and dmin represent independent degrees of freedom (scale range and offset), and constraining one as a fixed multiple of the other destroys reconstruction quality regardless of LS refinement. The min term needs per-superblock freedom. Future experiments aiming to compress d/dmin metadata should use delta encoding or shared codebooks rather than elimination.

---

## q4k-9b-asym65 — Asymmetric precision: 6-bit scales + 5-bit mins + fp16 d/dmin

**Hypothesis:** Scales multiply codes (affecting ranking = Top-P) while mins add offsets (affecting distribution bias = KL). Reducing mins to 5-bit while keeping scales at 6-bit preserves Top-P better than symmetric 5-bit reduction.
**Status:** success
**KL divergence:** 0.059674  |  **Top-P:** 88.172%  |  **Size:** 6.48 GB

**Implementation:**
- Added `sm_bits_min` parameter to `_encode_q4k()` (default 6, can be 5)
- Mins quantized to 31 levels (5-bit) instead of 63 (6-bit); scales stay at 6-bit/63 levels
- Separate packing: 8 × 6-bit scales in 6 bytes + 8 × 5-bit mins in 5 bytes = 11 bytes per superblock (saves 1 byte vs standard 12-byte interleaved)
- Helper functions: `_pack_values_6bit()` and `_pack_values_5bit()` for bit-packing
- `decode_q4k()` updated to detect format from `scales_mins_packed` last dim (12=standard, 11=asymmetric)
- `QuantizedLinear._unpack_scale_min_rows()` in inference.py updated with asymmetric unpacking path
- `_sm_asymmetric` flag auto-detected from packed tensor shape
- Combined with `--q4k-ddmin-fp16` for cumulative savings

**Result:**
KL (0.0597) is only 0.0008 above baseline (0.0588) and well below the 0.065 threshold. Top-P (88.17%) is within 0.05% of baseline (88.12%) and above the 88.0% threshold — very close to ddfp16 (88.20%). 5-bit mins cause minimal quality loss because additive offsets affect the distribution mean (KL) more than ranking (Top-P). LS refinement absorbs most of the rounding error. Size is 6.48 GB, beating ddfp16's 6.51 GB by ~32 MB.

**Lesson:**
Asymmetric precision works: mins are less sensitive to bit reduction than scales. This validates the hypothesis that multiplicative factors (scales) dominate Top-P while additive factors (mins) affect KL more. The 1-byte/superblock savings is modest but meaningful. The technique is general and could be extended to 4-bit mins (saving another byte) or combined with other compression methods. The key design principle for future experiments: preserve multiplicative precision (scales, d) at the expense of additive precision (mins, biases) when Top-P margin is tight.

---

## q4k-9b-smshare4 — Sub-block scale/min sharing (G=4)

**Hypothesis:** Halving the number of scale/min pairs by sharing across pairs of sub-blocks saves 0.185 GB while LS re-optimization absorbs the spatial resolution loss.
**Status:** regression
**KL divergence:** 0.095  |  **Top-P:** (not run; KL fails by wide margin)  |  **Size:** 6.18 GB

**Implementation:**
- Added `sm_groups` parameter to `_encode_q4k()` and downstream plumbing
- Before quantizing scales/mins: merge raw sub-block statistics (min/max) by taking group-level min/max over 64-weight groups
- Expand merged values back to 8 sub-blocks for code assignment and LS
- Pack only G=4 scale/min pairs (6 bits each) using separate packable format → 6 bytes per superblock (50% savings)
- Auto-detect sm_groups from packed tensor shape in inference.py, expand back to 8 sub-blocks

**Result:**
The group-level min/max over 64 weights gives a wider range than any individual 32-weight sub-block. This increases the effective quantization step (d_sub) for both sub-blocks, causing higher quantization error. LS at the superblock level (d/dmin) cannot compensate for per-2-sub-block errors. KL regressed massively from 0.059 to 0.095.

**Lesson:**
Sub-block scale/min sharing fundamentally fails because the combined range of multiple sub-blocks is wider than individual ranges, increasing quantization error everywhere. LS only has 2 DOF per superblock (d/dmin) vs 8 sub-blocks, so it cannot absorb spatial resolution loss. This same failure mode would apply to any sharing scheme (G=2, G=4).

---

## q4k-9b-sm5bit — 5-bit scale/min precision

**Hypothesis:** Reducing scale/min precision from 6-bit to 5-bit, with LS solving for optimal d/dmin given the coarser values, saves 0.062 GB while maintaining quality close to baseline.
**Status:** regression (near-pass)
**KL divergence:** 0.062346  |  **Top-P:** 87.347%  |  **Size:** 6.45 GB

**Implementation:**
- Added `sm_bits` parameter to `_encode_q4k()` (4, 5, or 6)
- Quantize scales/mins to 31 levels (5-bit) instead of 63 (6-bit)
- Single LS pass with the 5-bit values (no post-LS re-quantization — found to be harmful)
- Pack 8 × 5-bit scales + 8 × 5-bit mins = 10 bytes per superblock using `_pack_values_Nbit`
- Auto-detect bit width from packed tensor shape in inference.py
- Post-LS re-quantization was explored but removed: it perturbed codes and degraded quality (KL=0.068 with it vs 0.062 without)

**Result:**
KL divergence (0.0623) is within the ≤0.065 threshold and close to baseline (0.059). However, Top-P agreement (87.35%) falls below the 88.0% threshold by 0.65 percentage points. The pattern suggests that 5-bit precision introduces small per-sub-block perturbations that maintain the overall output distribution (KL good) but flip specific argmax decisions (Top-P failing). The fundamental limitation: scales/mins have per-sub-block errors (one per 32 weights) but LS can only optimize d/dmin (one pair per 256 weights), so the error cannot be fully absorbed.

**Lesson:**
Scale/min precision reduction is a viable compression target but the current 5-bit level is borderline. The Top-P failure is ~0.65% below threshold. Possible improvements: (a) try 6-bit with 5-bit mins only (asymmetric), (b) use stochastic rounding for the quantization instead of round-to-nearest, (c) add an alternating refinement pass that re-optimizes codes and scales/mins jointly (not just d/dmin). The near-pass result suggests this approach could work with minor refinements.

---

## q4k-9b-ddfp16 — fp16 d/dmin with LS re-optimization

**Hypothesis:** Storing d/dmin at fp16 precision (vs fp32) saves 0.125 GB; LS re-optimization after rounding absorbs the precision loss.
**Status:** success
**KL divergence:** 0.058364  |  **Top-P:** 88.197%  |  **Size:** 6.51 GB

**Implementation:**
- Added `ddmin_store_dtype` parameter to `_encode_q4k()` (quantize.py)
- After the existing LS refinement: round d/dmin to fp16, reassign codes with rounded values, re-run LS solve with new codes (legacy_exact path)
- Store final d/dmin as fp16 (2 bytes each vs 4 bytes each)
- `--q4k-ddmin-fp16` CLI flag in quantize.py
- No inference.py changes needed (`.float()` casts handle fp16→fp32 conversion)

**Result:**
Both KL and Top-P slightly IMPROVED over baseline (KL 0.058364 vs 0.058838, Top-P 88.197% vs 88.122%). The fp16 rounding error is negligible (~0.05% relative), and the extra LS re-solve iteration effectively serves as additional refinement. Size reduced from 6.63 GB to 6.51 GB (0.12 GB savings).

**Lesson:**
LS-based refinement can absorb metadata storage precision loss when applied after rounding. This is a general technique applicable to any quantization scheme that uses least-squares optimization. The key insight: codes adapt to the stored (rounded) d/dmin, so the reconstruction is consistent with the actual stored values. This technique could be stacked with other compression ideas (e.g., shared d/dmin with smaller K, or delta-encoded scales/mins).

---

## q4k-9b-fp16embed — Explicit bf16 embedding storage for residual

**Hypothesis:** Storing embed_tokens and lm_head at half-precision in residual.safetensors saves space vs fp32 while preserving quality since the model natively runs in bf16.
**Status:** success (no-op)
**KL divergence:** 0.059  |  **Top-P:** 88.12%  |  **Size:** 6.63 GB

**Implementation:**
- In `_save_packed_q4k()` (quantize.py ~line 556): cast tensors containing "embed_tokens" or "lm_head" to torch.bfloat16 before saving
- 3-line change to residual saving loop; no inference.py modifications

**Result:**
Model already loads in bf16 (via `torch_dtype=bfloat16`), so embeddings were already 2 bytes/element. The bf16→bf16 cast is a no-op — no space savings. Quality is identical to baseline (0.058838 KL, 88.122% top-P) since quantization weights are unchanged. The `load_state_dict(assign=True)` in inference.py does NOT automatically cast dtypes — an initial fp16 attempt caused a dtype mismatch error.

**Lesson:**
Embedding dtype is determined by the model loading dtype (`--dtype bfloat16` by default). To actually save space, the model would need to be loaded in fp32. The explicit bf16 cast is harmless (preserves baseline quality) but provides no benefit under current loading config. Future experiments aiming for space reduction should target the packed quantization weights, not residual tensors.

---

## q4k-9b-cb8 — Codebook-based scale/min encoding (8-bit, per-layer)

**Hypothesis:** Replacing 12-byte packed scale/min pairs with an 8-bit index into a layer-level codebook of 256 most common (scale,min) pairs saves ~0.12 GB while preserving quality through LS compensation.
**Status:** regression
**KL divergence:** 0.074  |  **Top-P:** 86.95%  |  **Size:** 6.36 GB

**Implementation:**
- `_codebook_encode_scales()` in quantize.py: collects all (scale,min) pairs across all output channels, builds frequency histogram, selects top-256 pairs as codebook, snaps each sub-block to nearest codebook entry via L1 distance
- Codebook encoding happens BEFORE code assignment and LS refinement, so snapped values govern both
- Decoder (`inference.py`): loads `scales_mins_cb` [1, 256, 2] uint8 and `scales_mins_idx` [out_c, n_blocks, 8] uint8, looks up scale/min from codebook
- per-layer codebook (cb_K=1), 8-bit indices (1 byte per sub-block)
- Storage: 8 bytes per superblock vs baseline 12 → 33% savings on scale/min storage

**Result:**
KL regressed from 0.059 to 0.074 (+26%), top-P from 88.12% to 86.95%. With only 256 codebook entries covering ~12% of the ~2100 unique (scale,min) pairs per layer, most sub-blocks get snapped to a different pair. The L1-distance snapping introduces per-sub-block scale/min errors of 1-3 units. LS refinement operates at the superblock level (d/dmin are per-superblock) and cannot compensate for per-sub-block relative errors across the 8 sub-blocks. The cumulative effect across 201 layers degrades output quality significantly.

**Lesson:**
Codebook-based scale/min encoding with per-layer global codebook and 256 entries introduces too much per-sub-block error. Improving coverage would require either (a) more entries (wider indices, reducing storage savings) or (b) per-group codebooks (increasing codebook overhead). Neither preserves the 33% storage savings target. Future experiments should consider techniques that preserve per-sub-block precision, such as delta encoding from a reference pair within each superblock.

---

## q4k-9b-dshare32 — Shared d/dmin K=32

**Hypothesis:** Sharing d/dmin across K=32 output channels with 4-bit per-channel multiplicative scale factors preserves quality while saving ~0.19 GB.
**Status:** regression
**KL divergence:** 0.177  |  **Top-P:** 78.02%  |  **Size:** 6.42 GB

**Implementation:**
- Post-processes LS-refined d/dmin via `_share_d_scales()` in quantize.py
- Computes shared d = max of 32 adjacent channels; per-channel 4-bit factor = d_i / d_shared quantized to 1-15
- Stores shared d/dmin at 1/32 the original size, plus packed 4-bit factors (2 per byte)
- Decoder reconstructs: `effective_d = d_shared * (factor / 15.0)`

**Result:**
4-bit factors are too coarse — 4 bits can only represent 16 levels (1/16, 2/16, ..., 1.0). The large d/dmin variation across channels within a group of 32 means many channels get significant scale errors. LS refinement ran before sharing, so d/dmin were optimized for the full-precision values, not the shared+factor approximation.

**Lesson:**
Metadata sharing at the d/dmin level needs wider factor bits or smaller K. 4-bit factors at K=32 lose too much per-channel precision. Try smaller K (4 or 8) or 6-bit factors. Also consider applying sharing BEFORE LS refinement so LS compensates.

---

## q4k-9b-baseline — Q4_K baseline

**Hypothesis:** Q4_K with legacy_exact LS refinement and fp32 scales on Qwen3.5-9B establishes the quality bar for all future experiments.
**Status:** success
**KL divergence:** 0.059  |  **Top-P:** 88.12%  |  **Size:** 6.63 GB

**Implementation:**
- 256-weight superblocks, 8 sub-blocks of 32
- Per-superblock fp32 d/dmin, 8×6-bit scales/mins packed in 12 bytes
- 4-bit quants packed as uint8 byte-pairs
- legacy_exact LS: codes assigned once, d/dmin solved with codes held fixed
- Packed-only safetensors format (q4k_affine_v2)

**Result:**
Baseline quality matches the old BF16 dequantized model (0.059 KL, 88.12% top-P). The legacy_exact refine mode was essential — alternating mode or no LS both regress.

**Lesson:**
The legacy_exact LS algorithm is critical for quality. Any technique that modifies codes must be applied BEFORE LS refinement so d/dmin are optimized for the actual stored codes.
