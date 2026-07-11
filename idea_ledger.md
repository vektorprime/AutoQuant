# Idea Ledger

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
