# Idea Ledger

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
