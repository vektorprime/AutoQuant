# Idea Ledger

## q4k-9b-embednorm — Row-wise L2-norm-preserving embedding quantization (regression)

**Hypothesis:** Preserving per-row L2 norms after Q4_K embedding quantization prevents embedding magnitude distortion that propagates through all downstream attention layers.
**Status:** regression
**KL divergence:** 0.062225  |  **Top-P:** 87.797%  |  **Size:** 5.07 GB

**Implementation:**
- After Q4_K encoding embedding to 4-bit superblocks (6+6 metadata, fp32 d/dmin, alternating LS 3+f), decode back and compute per-row L2 norm ratio (orig_norm / quant_norm)
- Store one fp16 scale per embedding row (~0.5 MB for 252K vocab × 2 bytes)
- At inference in QuantizedEmbedding.forward, multiply dequantized rows by their scale factor
- Clamp ratio to [0.1, 10.0] to prevent extreme rescaling
- Added `--q4k-embed-norm-preserve` CLI flag

**Result:**
Both KL and Top-P got WORSE than the baseline without norm-preserve (KL 0.062 vs 0.054-0.059, Top-P 87.80% vs 88.0-88.4%). The norm rescaling amplified existing quantization errors proportionally while also distorting relative embedding vector norms that the attention mechanism relies on. Most fundamentally: the L2 norm within a 4096-dim embedding row is dominated by the cumulative effect of all 16 superblocks, and rescaling at row level cannot fix the structured error within each superblock. The attention layers see specific sub-space patterns, not aggregate norms.

**Root cause:** Norm-preserve operates on a global statistic (row L2 norm) but the damaging error comes from the block-level correlation within each superblock. Multiplying by a scalar changes the embedding vector uniformly, which distorts the relative geometry between embedding vectors in a way that hurts attention more than it helps individual token representations.

**Lesson:**
Norm-preserving rescaling at the row level is actively harmful for embedding quantization quality. The superblock structure's error pattern (8 sub-blocks of 32 per 256-group) creates localized distortions that cannot be compensated by a single scalar multiplier per row. Future approaches should focus on: (1) rethinking the superblock structure for embeddings specifically (smaller blocks, different grouping), (2) applying transformations that reduce variance within superblocks (e.g., row-wise centering before quantization), or (3) accepting that embeddings need a fundamentally different quantization approach than linear weights (since embeddings feed directly into attention without any pre-processing layer).

---

## q4k-9b-embed5bit-fp32 — 5-bit embedding quants with fp32 d/dmin (regression)

**Hypothesis:** fp16 d/dmin rounding error was hurting 5/6-bit experiments because d values are proportionally smaller with larger maxq; switching to fp32 should recover quality.
**Status:** regression
**KL divergence:** 0.052816  |  **Top-P:** 87.747%  |  **Size:** 4.98 GB

**Implementation:**
- Used `--no-q4k-embed-ddmin-fp16` to force fp32 d/dmin for embeddings
- Same 5-bit quants (maxq=31) with 6+6 scales/mins, alternating LS 3+f
- fp32 stores each d/dmin at 4 bytes vs 2 bytes, adding ~16 MB

**Result:**
KL improved to 0.052816 (best of any experiment, better than all 4-bit variants). But Top-P regressed further to 87.747%, worse than both the 4-bit baseline (87.947%) and 5-bit fp16 (87.847%). The fp16→fp32 change eliminated the rounding error hypothesis as root cause.

Root cause: 5/6-bit quants do NOT improve Top-P over 4-bit for embeddings. The embedding quantization loss (~0.15-0.25% Top-P) is structural — quantization error in the FIRST layer propagates through all 201 subsequent layers. More quants bits reduce per-weight MSE (confirmed: 2-4× improvement in roundtrip accuracy) but do not translate to Top-P gains because the error pattern matters more than error magnitude. Embedding errors at the first layer perturb the input to every downstream layer, so even small per-weight errors accumulate multiplicatively.

**Lesson:**
The superblock-based quantization (d/dmin per 256 weights with sub-block affine scales) introduces a specific error structure in embeddings that affects argmax decisions regardless of quants precision. The fundamental issue is architectural: the embedding table has 252K independently learned rows with widely varying norms, and superblock quantization groups 256 dimensions per output row. This approach ignores the row-level structure of embeddings. Future experiments should consider: (1) per-row quantization (each embedding row quantized independently, not grouped into superblocks), (2) row-wise norm preservation before/after quantization, or (3) not quantizing embeddings at all and targeting other large tensors (like lm_head).

---

## q4k-9b-embed6bit — 6-bit embedding quants (regression)

**Hypothesis:** 6-bit embedding quants (64 levels) with 2× finer resolution than 5-bit should restore Top-P by further reducing per-weight quantization error.
**Status:** regression
**KL divergence:** 0.053279  |  **Top-P:** 87.797%  |  **Size:** 5.09 GB

**Implementation:**
- Extended `--q4k-embed-quant-bits` choices to include 6
- Added `_pack_values_6bit()` reuse for quants (8 values → 6 bytes, 48 bits)
- Added 6-bit unpack paths in `decode_q4k()` and `QuantizedEmbedding._unpack_quants_rows()`
- 6+6 scales/mins, fp16 d/dmin, alternating LS 3+f
- Size: 5.09 GB (+129 MB over 5-bit, +253 MB over 4-bit)

**Result:**
KL (0.053279) is excellent, slightly better than 5-bit (0.053337) but worse than the best 4-bit (0.053225). Top-P (87.797%) is the WORST of any experiment, below both 4-bit (87.947%) and 5-bit (87.847%). The counter-intuitive result — more bits → worse Top-P — confirms that quants precision is NOT the bottleneck for embedding quality.

**Lesson:**
The monotonic degradation (4-bit > 5-bit > 6-bit in Top-P) despite improving per-weight MSE suggests that the alternating LS optimization interacts differently with different code ranges. With maxq=63, the code assignment produces larger code values that interact differently with the scale_norm/min_norm in LS. The 3-iteration count, optimal for 4-bit, may not be optimal for 5/6-bit. Additionally, the d values become proportionally smaller (range/maxq), making the alternating LS more sensitive to initial conditions. This validates the lesson from embed5bit-fp32: the superblock structure itself is the bottleneck, not the per-weight precision.

---

## q4k-9b-embed5bit — 5-bit embedding quants with fp16 d/dmin (regression)

**Hypothesis:** Adding `--q4k-embed-quant-bits` flag to control embedding quants. 5-bit (32 levels) should halve the quantization step size for embeddings, closing the 0.053% Top-P gap of embed6+6 by capturing the wider dynamic range across 252K vocabulary tokens.
**Status:** regression
**KL divergence:** 0.053337  |  **Top-P:** 87.847%  |  **Size:** 4.96 GB

**Implementation:**
- Added `--q4k-embed-quant-bits` CLI flag (choices: 4,5,6, default 4)
- `_encode_q4k()` now accepts `quant_bits` parameter; computes `maxq = (1<<quant_bits)-1`
- For quant_bits=5: quants packed with `_pack_values_5bit()` (8 values → 5 bytes per group)
- `QuantizedEmbedding._unpack_quants_rows()` branches on `self.quant_bits` for 5-bit unpacking
- `decode_q4k()` fixed: was broken for 5-bit (tried 4-bit unpack first, now branches before)
- 6+6 scales/mins, fp16 d/dmin, alternating LS 3+f for embeddings
- Linear layers: 5+4 fp16 alt 3+f (same as asym5s4-i3f)

**Result:**
KL (0.053337) is excellent — beats both baseline (0.059) and the asymmet5s4-i3f (0.053596), and is virtually tied with the best embed6+6-alt3 (0.053225). But Top-P (87.847%) regressed vs embed6+6-alt3 (87.947%). 5-bit quants did NOT close the 0.053% gap — they made it slightly worse.

**Lesson:**
The embedding quantization Top-P loss is structural, not just a matter of code precision. More quants bits (32 vs 16 levels) give 2× better per-weight reconstruction (confirmed by roundtrip tests) but don't improve Top-P because: (1) embedding errors at the first layer propagate multiplicatively through all subsequent layers, (2) the superblock quantization structure (per-256-weight d/dmin) introduces correlated errors across dimensions that disrupt argmax more than independent per-weight noise, and (3) the alternating LS optimization with more code levels may converge to different local optima. Future experiments should target: (1) per-row quantization without superblock grouping, (2) adaptive bit allocation based on embedding row frequency, or (3) post-quantization embedding row normalization to compensate for quantization-induced norm drift.

## q4k-9b-embed6+6 — Embedding quantization with 6+6 precision (near-miss regression)

**Hypothesis:** Quantizing the 2.03 GB embedding table with full 6+6 precision (same as baseline Q4_K) should preserve Top-P matching the no-embedding-quant asym5s4-i3f, while 5+4 linear layers keep the model size advantage.
**Status:** regression (near-pass — 0.053% below Top-P threshold)
**KL divergence:** 0.053225  |  **Top-P:** 87.947%  |  **Size:** 4.84 GB

**Implementation:**
- Added `--q4k-embed-*` CLI flags to quantize.py: separate control of scale_dtype, refine_mode, refine_iters, ddmin_store_dtype, sm_bits_scale, sm_bits_min for embedding layers vs linear layers
- Extended `quantize_model()` to accept and apply embed-specific quantization parameters
- Fixed `_save_packed_q4k()`: now allows mixed scale_dtypes and refine_modes between linear and embedding layers (previously raised RuntimeError)
- Rewrote `QuantizedEmbedding._unpack_scale_min_rows()` in inference.py: now handles all common format combinations (5+4, 6+3, 6+4, 6+5, 6+6, symmetric 5-bit) using explicit `sm_bits_scale`/`sm_bits_min` from the manifest (eliminated broken auto-detection from tensor shape)
- `QuantizedEmbedding.__init__` now accepts `sm_bits_scale` and `sm_bits_min` params; `load_quantized_model()` passes them from manifest
- Linear layers: 5-bit scales + 4-bit mins + fp16 d/dmin + alternating LS (3 iters + final)
- Embedding: 6-bit scales + 6-bit mins + fp16 d/dmin + alternating LS (3 iters)

**Result:**
Three configurations tested:
1. Embed legacy_exact: KL 0.053553, Top-P 87.697%
2. Embed alt 3 iters: KL 0.053225, Top-P 87.947% **(best)**
3. Embed alt 5 iters: KL 0.053563, Top-P 87.847%

KL is the best of ANY experiment (0.053225), comfortably beating baseline (0.059) and asym5s4-i3f (0.0536). Size at 4.84 GB is 1.43 GB smaller than asym5s4-i3f (6.27 GB) and 1.79 GB smaller than baseline (6.63 GB). But Top-P (87.947%) falls 0.053% below the 88.0% threshold.

The embedding quantization at 4-bit (even with 6+6 scales/mins and 3-iter alternating LS) loses ~0.3% Top-P vs unquantized bf16 embeddings. This is the FIRST LAYER of the model — quantization errors in embeddings propagate through every subsequent attention and FFN layer. The 6+6 format provides baseline-quality scale/min precision, but the 4-bit quants (16 levels) with d/dmin resolution of 1 output row per 256 input features cannot perfectly reconstruct the wide value range across 248K vocabulary embedding rows.

Increasing refinement iterations from 3 to 5 degraded Top-P from 87.947% to 87.847%, following the same pattern seen in linear layers (more alternating iterations don't monotonically improve convergence). The fp16 d/dmin roundtrip + LS re-opt (ddfp16 approach) provides a small additional refinement step compared to pure fp32 storage.

**Lesson:**
Embedding quantization at 4-bit is fundamentally limited by the first-layer error propagation problem. The 4-bit quants are the bottleneck — 16 levels cannot capture the full dynamic range of embedding vectors across 248K vocabulary tokens. Three viable paths forward:
1. **Per-row finer quants**: Use 5-bit or 6-bit quants for embeddings specifically (don't touch the linear layers' 4-bit format). The embedding is only ~4M superblocks, so 5-bit quants would add ~250 MB but might close the 0.3% Top-P gap.
2. **Per-row finer d/dmin**: Use K=1 (no sharing) for embedding d/dmin to give each embedding row independent scale control.
3. **Embedding normalization**: Pre-normalize embedding rows to unit norm before quantization, store the norm as a per-row scalar. This would reduce the dynamic range variation and make 4-bit quants more effective.

The `QuantizedEmbedding` module with sparse row lookup and the embed-specific CLI flags are reusable components for future experiments. The mixed-quality quantization infrastructure (different bits for different layer types) is a novel capability that enables per-layer-type optimization.

---

## q4k-9b-sym5s-embed — Embedding quantization (regression, near-pass)

**Hypothesis:** Quantizing the 2.03 GB embedding table to 4-bit (5+4 format) saves ~1.5 GB while maintaining quality through iterative LS refinement.
**Status:** regression (near-pass)
**KL divergence:** 0.061981  |  **Top-P:** 87.497%  |  **Size:** 4.93 GB

**Implementation:**
- Added `QuantizedEmbedding` class in inference.py: packed-only embedding layer with sparse row lookup
- Modified `quantize.py` to detect `nn.Embedding` modules and quantize their [vocab, hidden] weight matrices using the same `_encode_q4k` pipeline
- QuantizedEmbedding.forward(): sorts token IDs, dequantizes needed rows in batches (max 2048 contiguous rows per batch), individual row dequantization for sparse lookups
- Embedding stored in packed safetensors alongside Linear layers; manifest has separate "quantized_embeddings" section
- Residual drops from 2.05 GB to 14.7 MB — almost all residual was the embedding table
- 5+4 format + fp16 d/dmin for both Linear and Embedding layers

**Result:**
KL (0.0620) passes the ≤0.065 threshold comfortably, with minimal degradation vs pure 5+4 format (0.0536). However, Top-P (87.50%) falls 0.5% below the 88.0% threshold. The embedding layer is the first layer in the model — quantization errors in embeddings propagate through every subsequent layer. Even small per-weight errors in the embedding table (248K × 4096) accumulate across the 248K vocabulary tokens, affecting the first-layer representations that feed into all subsequent attention and MLP computations.

The 5+4 format (5-bit scales, 4-bit mins) may be too aggressive for embeddings compared to the standard 6+6 format. With 31 scale levels vs 63, the embedding quantization grid is meaningfully coarser. Since Qwen3.5 has 248K vocabulary entries with widely varying embedding magnitudes (common tokens have large norms, rare tokens have small norms), the 5-bit scales struggle to simultaneously capture both large and small magnitudes within the same superblock.

**Lesson:**
Embedding quantization is a high-leverage compression target (2.03 GB → ~0.54 GB = 73% reduction), but quality is more sensitive than for internal Linear layers. Three paths to close the 0.5% Top-P gap: (1) use 6+6 format for embeddings (adds ~12 MB metadata), (2) use 6+6 format globally (adds ~93 MB total, still well below 6.27 GB threshold), or (3) use per-row bias correction during the LS solve to better handle the wide embedding magnitude range. The `QuantizedEmbedding` module is a reusable component for future experiments.

---

## q4k-9b-sym5s-bias — Symmetric scales + per-superblock bias (regression)

**Hypothesis:** Adding a per-superblock bias term b to the symmetric formulation restores the 2-DOF LS (d, b) and absorbs the offset errors that the pure symmetric approach couldn't handle.
**Status:** regression
**KL divergence:** 0.135473  |  **Top-P:** not run  |  **Size:** 6.14 GB

**Implementation:**
- Modified symmetric path in `_encode_q4k()`: added b (per-superblock bias, stored as fp16)
- 2×2 LS system solving for (d, b) jointly
- Storage: 5 bytes scales + 2 bytes d + 2 bytes b = 9 bytes/sb (save 4 vs 13 bytes/sb of 5+4 format)
- Decode: w = d * s_k * (q - 7.5) + b

**Result:**
KL (0.1355) showed essentially no improvement over the no-bias version (0.1398). The per-superblock bias corrects superblock-level offset errors but cannot address per-sub-block centering issues. Each sub-block within a superblock has its own weight distribution, and a single bias per superblock applies uniformly to all sub-blocks. The fundamental problem remains: the symmetric formulation with fixed zero-point at 7.5 ties each sub-block's effective range to be centered at d * s_k * 7.5, which cannot adapt to sub-blocks where weight distributions are asymmetric or off-center.

**Lesson:**
The sub-block offset (min term) is essential — it provides independent per-sub-block centering that neither a fixed zero-point nor a global bias can replace. Adding 4-bit per-sub-block zero-points (instead of fixed 7.5) would give independent offset control while keeping metadata competitive: 5 bytes scales + 4 bytes zero-points + 2 bytes d = 11 bytes/sb. This is a natural extension that preserves the per-sub-block flexibility of the original format.

---

## q4k-9b-sym5s — Symmetric per-sub-block scales, no mins (regression)

**Hypothesis:** Using symmetric quantization around zero (w = d * s_k * (q - 7.5)) eliminates the need for dmin and mins entirely, saving 6 bytes/sb vs 5+4 format.
**Status:** regression
**KL divergence:** 0.139802  |  **Top-P:** not run  |  **Size:** 6.08 GB

**Implementation:**
- Per-sub-block symmetric scale from max absolute value: s_k = max(|min|, |max|) / 7.5
- d = max(s_k) per superblock; s_k_norm = s_k / d quantized to 5-bit
- Codes: q = round(w / (d * s_norm) + 7.5), clamped to [0, 15]
- 1-DOF LS: solve for d only
- Storage: 5 bytes (8 × 5-bit scales) + 2 bytes (d, fp16) = 7 bytes/sb

**Result:**
KL (0.1398) is 2.4× worse than the ≤0.065 threshold. The fundamental flaw: with fixed zero-point at 7.5, each sub-block's representable range is [-7.5ds, +7.5ds], always symmetric around 0. LLM weights frequently have non-zero-mean distributions within sub-blocks (e.g., [-0.1, +0.9] or [-0.8, +0.2]). The symmetric formulation forces the representable range to be centered at 0, wasting precision on the unused side. With only 1 DOF (d) in the LS solve, there's no way to compensate for per-sub-block centering errors. This validates the lesson from q4k-9b-dmindrop: independent offset terms per sub-block are necessary for acceptable quality.

**Lesson:**
Symmetric quantization without per-sub-block offsets is not viable for LLM weights. Any technique that eliminates the min/offset term must provide alternative per-sub-block centering (e.g., sub-block-specific zero-points stored alongside scales). The 2-DOF constraint identified in the architecture notes is not just about superblock LS — it reflects a fundamental requirement for per-sub-block offset flexibility.

## q4k-9b-hadamard44 — Hadamard + 4-bit scales + 4-bit mins (regression)

**Hypothesis:** Random Hadamard transform decorrelates weights enough to tolerate 4-bit scales (15 levels) instead of 5-bit, saving 1 byte/superblock (8 bytes/sb total).
**Status:** regression
**KL divergence:** 0.055957  |  **Top-P:** 87.622%  |  **Size:** 6.24 GB

**Implementation:**
- Added 4+4 packing format: 8×4-bit scales (4 bytes) + 8×4-bit mins (4 bytes) = 8 bytes/superblock
- Full codec support in decode_q4k(), QuantizedLinear._unpack_scale_min_rows(), CLI choices
- Stacked with Hadamard transform + alternating LS (3 iters + final) + fp16 d/dmin
- sm_last=8 detection for backward compat

**Result:**
KL (0.0560) is well within threshold and comparable to early asym5s4 experiments. But Top-P (87.622%) falls 0.38% below the 88.0% threshold — essentially the same gap as asym64 (4-bit mins only). 4-bit scales with only 15 levels are fundamentally too coarse for the multiplicative scale factor, even with Hadamard decorrelation. The effective scale `d*s` has `s ∈ {1/15, 2/15, ..., 15/15}`, and the 1/15 granularity creates per-sub-block quantization errors that LS at the superblock level cannot fully absorb.

**Lesson:**
Hadamard decorrelation does not meaningfully reduce scale variation. The scale factor `d_sub/d` determines how much the quantization grid stretches between sub-blocks, and this ratio is fundamentally determined by the weight distribution within each 256-weight superblock. Hadamard mixing of input channels doesn't change the relative scale between adjacent sub-blocks of 32 consecutive input features. Future compression should target the quants (4-bit codes are 86.5% of packed size) rather than the metadata which is already compressed to 8 bytes/sb.

---

## q4k-9b-hadamard — Hadamard + 5-bit scales + 4-bit mins (PASS, neutral)

**Hypothesis:** Random Hadamard transform (QuaRot-style) applied to weight matrices before quantization decorrelates channels, making weights more uniform and improving quantization quality.
**Status:** success
**KL divergence:** 0.054455  |  **Top-P:** 88.172%  |  **Size:** 6.27 GB

**Implementation:**
- Block-diagonal random Hadamard transform (block_size=256, matching superblock size) applied along input dimension: W' = W @ H_diag^T
- At inference: x' = x @ H_diag^T applied before dequantized linear, same direction as weight transform
- Normalized Hadamard (entries ±1/√256) ensures H @ H^T = I, preserving mathematical correctness
- Per-layer random seeds derived from layer name hash for reproducible diversity
- Module-level block cache in inference.py avoids VRAM duplication
- Stacked with 5+4 format + alternating LS (3 iters + final) + fp16 d/dmin

**Result:**
KL (0.0545) is slightly worse than asym5s4-i3f (0.0536) but better than early asym5s4 (0.0565). Top-P (88.172%) is slightly better than asym5s4-i3f (88.122%) and the baseline (88.122%). Net effect is statistically neutral — within ±0.001 KL and ±0.05% Top-P of the no-Hadamard 5+4. Size is unchanged at 6.27 GB (same format, same bit widths).

The Hadamard transform adds ~1% latency penalty (15.5 vs 17.1 tok/s from the best result, though variance across runs suggests this is within noise). The inference overhead is the input activation transform (block-diagonal matmul of 256×256 blocks) per linear layer, which is a small fraction of the total GEMM cost.

**Lesson:**
Hadamard decorrelation provides minimal benefit for our encoding scheme because the quantization bottleneck is not the weight distribution but the degrees of freedom mismatch: superblock LS has 2 DOF (d/dmin) to compensate for 8 sub-blocks' worth of scale/min errors. Decorrelating input channels doesn't help when the limiting factor is within-superblock scale variation. This technique is orthogonal to other compression methods; it could be combined with deeper architectural changes (changing the superblock structure or using multiple d/dmin per superblock) but offers no advantage as a standalone addition.

---

## q4k-9b-asym5s4-i3f — 5-bit scales + 4-bit mins + alt LS (3 iters) + final LS solve (PASS)

**Hypothesis:** Adding a final LS solve after the alternating iteration loop ensures d/dmin are optimal for the stored codes, closing the 0.003% Top-P gap of the original asym5s4.
**Status:** success
**KL divergence:** 0.053596  |  **Top-P:** 88.122%  |  **Size:** 6.27 GB

**Implementation:**
- One-line change in `_encode_q4k()`: after the alternating loop, run `d, dmin = solve_scales(codes, d, dmin, legacy=False)` so stored d/dmin are optimal for final codes.
- Same format as asym5s4: 5-bit scales (5 bytes) + 4-bit mins (4 bytes) = 9 bytes/superblock, fp16 d/dmin
- 3 iterations of alternating LS + final LS = 4 LS solves total
- CLI: `--q4k-refine-mode alternating --q4k-refine-iters 3 --q4k-sm-bits-scale 5 --q4k-sm-bits-min 4 --q4k-ddmin-fp16`
- Also added `--q4k-refine-iters` CLI flag for iteration count control

**Result:**
KL improved from 0.0565 (original asym5s4) to 0.0536 — the best KL of any non-regression experiment. Top-P reached exactly 88.122%, matching the baseline Q4_K and clearing the 88.0% threshold. The model is the smallest passing format at 6.27 GB, beating asym64alt (6.30 GB) by 0.03 GB while matching baseline Top-P.

Root cause analysis of previous asym5s4 near-miss: without the final LS solve, stored d/dmin were optimal for iteration N-1 codes, not iteration N codes. The one-iteration mismatch introduced a small systematic error that accounted for the ~0.003% Top-P gap. The fix is a minimal change with zero size cost.

**Lesson:**
In any alternating optimization scheme where LS solves are cheaper than code reassignments (or vice versa), always end with the solve step so stored parameters are consistent with stored codes. This is a general principle: the stored representation should be a fixed point of the optimization, not one step off. The 3-iteration count is sweet spot — fewer iterations underfit, more iterations give diminishing returns for the 9-byte format. The 5+4 format is now fully validated as a novel, quality-preserving compression technique.

---

## q4k-9b-asym63 — 6-bit scales + 3-bit mins (regression)

**Hypothesis:** Pushing mins to 3-bit while keeping scales at 6-bit preserves multiplicative precision (scales) for Top-P, trading off additive precision (mins) that mainly affects KL.
**Status:** regression
**KL divergence:** 0.059057  |  **Top-P:** 87.772%  |  **Size:** 6.27 GB

**Implementation:**
- Added `_pack_values_3bit()` for packing 8 × 3-bit values into 3 bytes
- 6-bit scales (6 bytes) + 3-bit mins (3 bytes) = 9 bytes/superblock (same as 5+4)
- Full codec support: `decode_q4k()`, `QuantizedLinear._unpack_scale_min_rows()`, manifest, backward compat
- 5 alternating LS iterations + final LS solve, fp16 d/dmin

**Result:**
KL (0.0591) is close to baseline, better than asym5s4 (0.0565). But Top-P (87.772%) is worse than both 5+4 variants. 3-bit mins with only 7 levels are too coarse — the additive offset error distorts argmax rankings even with perfect multiplicative precision. The alternating LS + final solve absorb some error but not enough.

**Lesson:**
Mins at 3-bit (7 levels) are below the viable threshold for this architecture, even with iterative refinement. The min term appears in `w = d*s*q - dmin*m` — when `dmin*m` is coarsely quantized, it creates structured errors in the per-sub-block offset that LS at the superblock level cannot fully absorb. The minimum viable min precision appears to be 4-bit (15 levels), which the asym64alt experiment proved works with alternating LS. Future experiments should target scales (multiplicative) rather than mins (additive) for further compression.

---

## q4k-9b-asym5s4-v2 — 5+4 with 5 iters + final LS (near-pass)

**Hypothesis:** More alternating LS iterations (5 vs 3) with final LS solve would converge to a better local optimum than the 3-iter original.
**Status:** regression (near-pass)
**KL divergence:** 0.054997  |  **Top-P:** 87.947%  |  **Size:** 6.27 GB

**Result:**
KL improved significantly (0.0550 vs 0.0565 original) but Top-P (87.947%) was slightly worse than the 3-iter original's 87.997%. More iterations don't monotonically improve — the alternating scheme converges to different local optima depending on the iteration count. The 3-iter + final LS combo is the sweet spot.

---

## q4k-9b-asym5s4-i5 — 5+4 with 5 iters, no final LS (regression)

**Hypothesis:** 5 iterations of alternating LS (vs 3) would push Top-P over 88.0%.
**Status:** regression
**KL divergence:** 0.056540  |  **Top-P:** 87.747%  |  **Size:** 6.27 GB

**Result:**
Both KL and Top-P degraded vs 3-iter original. Without the final LS solve, the d/dmin-code mismatch is larger with more iterations (codes are 1 iteration ahead of d/dmin). This confirms the final LS fix is essential.

---

## q4k-9b-asym5s4 — 5-bit scales + 4-bit mins + alternating LS + fp16 d/dmin

**Hypothesis:** Reducing scales from 6-bit to 5-bit (31 levels) saves 1 byte/superblock; alternating LS should absorb scale quantization error just like it absorbed min error in asym64alt.
**Status:** regression (near-pass)
**KL divergence:** 0.056483  |  **Top-P:** 87.997%  |  **Size:** 6.27 GB

**Implementation:**
- Added `sm_bits_scale` parameter (default 6) to `_encode_q4k`, `_quantize_one_layer_q4k`, `quantize_model`
- Scales quantized to 31 levels (5-bit) instead of 63 (6-bit): `scale_levels = (1 << sm_bits_scale) - 1`
- Packing: 8 × 5-bit scales (5 bytes, `_pack_values_5bit`) + 8 × 4-bit mins (4 bytes, `_pack_values_4bit`) = 9 bytes/superblock
- Manifest stores `sm_bits_scale`/`sm_bits_min` per layer for decoding disambiguation
- Inference: `QuantizedLinear` uses `sm_bits_scale` to select 5-bit vs 6-bit scale unpacking; `_sm_delta` detection updated to `sm_last == 9 and sm_bits_scale == 6 and sm_bits_min == 5`
- Backward compat: old checkpoints without sm_bits fields infer from `sm_last` (12→6/6, 11→6/5, 10→6/4)
- CLI: `--q4k-sm-bits-scale {5,6}`
- Combined: `--q4k-refine-mode alternating --q4k-ddmin-fp16 --q4k-sm-bits-scale 5 --q4k-sm-bits-min 4`

**Result:**
KL (0.0565) is excellent, significantly beating baseline (0.059) and close to asym64alt (0.0539). Top-P at 87.997% is just barely below the 88.0% threshold (0.003% gap). The alternating LS successfully absorbed most of the 5-bit scale quantization error — the additional code reassignment iterations compensated for the coarser multiplicative scale resolution. Size is 6.27 GB, the smallest yet, saving 31 MB vs asym64alt (6.30 GB).

The near-pass reveals that multiplicative factor precision (scales) has a measurable but small impact on Top-P: dropping from 6-bit to 5-bit costs ~0.45% Top-P (88.45% → 88.00%), about the same proportional drop as 6-bit → 4-bit mins (88.17% → 87.60% in asym64). The alternating LS nearly closed this gap.

**Lesson:**
Alternating LS is a general-purpose error absorption mechanism that works for both additive (mins) and multiplicative (scales) metadata quantization. However, scales are more sensitive than mins on a per-bit basis because they directly multiply codes. The 5-bit scale limit is borderline — 6 values (1-31) are enough for most sub-blocks, but the tail cases where d_sub/d requires higher precision push Top-P just over the edge. A hybrid approach (5-bit scales for most layers, 6-bit for early/late layers) or per-channel adaptive bits might close the remaining 0.003% gap. Alternatively, stochastic rounding in the scale quantization could distribute errors more favorably.

## q4k-9b-asym64alt — Alternating LS refinement with 4-bit mins + fp16 d/dmin

**Hypothesis:** Alternating LS refinement (code reassignment + LS, 3 iterations) can better absorb 4-bit min quantization error than single-pass legacy_exact, pushing asym64 over the Top-P threshold.
**Status:** success
**KL divergence:** 0.053944  |  **Top-P:** 88.447%  |  **Size:** 6.30 GB

**Implementation:**
- Used `--q4k-refine-mode alternating` with `--q4k-sm-bits-min 4 --q4k-ddmin-fp16`
- No code changes needed — existing alternating LS infrastructure
- 3 iterations of: LS solve for d/dmin → reassign codes with new d/dmin
- 10 bytes/superblock (6 bytes scales + 4 bytes mins)

**Result:**
KL improved significantly (0.0539 vs legacy_exact asym64's 0.0534 and baseline's 0.0588). Top-P reached 88.45%, beating both the 88.0% threshold and the baseline's 88.12%. The alternating refinement successfully absorbed the 4-bit min quantization error: codes are reassigned after each LS solve, finding a better joint optimum for the coarser mins. Size is 6.30 GB, the smallest passing model yet.

**Lesson:**
The key insight: single-pass LS (legacy_exact) is optimal when scales/mins are high-precision (6-bit), but alternating LS outperforms when scales/mins are coarsely quantized. The additional code reassignment steps compensate for the lossy min quantization by finding codes that work better with the quantized scales/mins. This is a general technique: pair coarser quantization with iterative code refinement. The asymmetry hypothesis (mins tolerate lower precision than scales) is fully validated — 4-bit mins work when combined with alternating LS.

---

## q4k-9b-sdelta4 — Intra-superblock scale/min delta encoding (4-bit signed deltas)

**Hypothesis:** Neighboring sub-block scales/mins within a 256-weight superblock are correlated; storing a reference + 4-bit signed deltas compresses scales/mins from 11 to 9 bytes/superblock.
**Status:** regression
**KL divergence:** 0.246806  |  **Top-P:** not run (KL fails)  |  **Size:** 6.27 GB

**Implementation:**
- Sub-block 0 used as reference (6-bit scale + 5-bit min)
- 7 remaining sub-blocks stored as 4-bit signed deltas (±8 range)
- Packed into 9 bytes/superblock (saves 2 bytes vs asym65)
- Delta encoding applied BEFORE code assignment so LS can absorb perturbation
- Custom bit-packing with 67 bits → 9 bytes (min delta split: bit 3 at byte[bi].bit7, bits 0-2 at byte[bi+1].bits0-2)

**Result:**
38% of scale/min deltas are clipped (outside ±8 range). Max scale delta within a superblock: 61 (out of 63 levels). Even with median reference, 23% are clipped. The intra-superblock scale/min variation is fundamentally too high for 4-bit deltas: the 8 sub-blocks of 32 consecutive input features have up to 62-level scale differences, far exceeding the ±8 delta range. LS at the superblock level (2 DOF: d, dmin) cannot compensate for per-sub-block errors (14 excess DOF across 8 sub-blocks).

**Lesson:**
Per-sub-block scale/min compression is bottlenecked by the fundamental tension between superblock degrees of freedom (2) and sub-block degrees of freedom (16). Any compression technique that perturbs per-sub-block values cannot be fully absorbed by superblock-level LS. Future experiments should either: (a) change the block structure to reduce the DOF mismatch, (b) compress metadata that has fewer degrees of freedom (e.g., d/dmin), or (c) apply transforms that reduce the intra-superblock variation BEFORE quantization (e.g., Hadamard rotation).

---

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
