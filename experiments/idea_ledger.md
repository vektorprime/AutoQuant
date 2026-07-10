# Idea Ledger — AutoQuant Q4_K experiments

## exp-20260710-013 (Sub6)
Hypothesis: Reducing sub-blocks from 8 to 6 saves 3 bytes/superblock; LS can compensate.
Outcome: REGRESSED — KL 0.108, Top-P 82.34%.

## exp-20260710-014 (Packed 6-bit)
Hypothesis: Packing 6-bit scales/mins into 12 bytes (vs 16) saves ~12 MB.
Outcome: WIN — 422.8 MB, same quality as baseline (KL=0.093508).

## exp-20260710-015 (Packed 6-bit + LS)
Hypothesis: LS refinement + packed storage = smaller AND better.
Outcome: WIN — 422.8 MB, KL=0.092, Top-P=83.86%. New best.

## exp-20260710-016 (Packed 5-bit)
Hypothesis: 5-bit scales/mins packed into 10 bytes saves 2 more bytes/superblock.
Outcome: REGRESSED — KL=0.096, Top-P=82.96%. 5-bit precision loss too large.

## exp-20260710-017 (Asymmetric q//2)
Hypothesis: Halve q and double d for low-range blocks → 3-bit storage.
Outcome: CATASTROPHIC — KL=13.82. q//2 breaks reconstruction.

## exp-20260710-018 (Re-quantize after LS)
Hypothesis: Re-quantize q with optimized d/dmin, then LS again.
Outcome: REGRESSED — KL=0.0932 (worse than 0.092). Re-Q disrupts LS convergence.

## exp-20260710-019 (Float8 d/dmin)
Hypothesis: Log-scale 8-bit encoding of d/dmin saves 2 bytes/superblock.
Outcome: WIN — 416.9 MB, KL=0.092, Top-P=83.86%. CURRENT GLOBAL BEST.

## exp-20260710-020 (Proper 3/4-bit from start)
Hypothesis: Classify blocks BEFORE quantization, use 3-bit for 50% from the start.
Outcome: REGRESSED — KL=0.254. 3-bit too aggressive even with LS.

## exp-20260710-021 (Shared d/dmin across channels)
Hypothesis: Share d/dmin across K=4 output channels, store shared values (float8) + 4-bit per-channel scale factors. Storage-only compression (dequantized weights use precise d/dmin).
Outcome: WIN — 415.6 MB, KL=0.092, Top-P=83.86%. NEW GLOBAL BEST.

## Summary
- Best: Shared d/dmin (K=4) + float8 shared values + 4-bit scale factors + packed 6-bit scales/mins + 3-pass LS → 415.6 MB, KL=0.092, Top-P=83.86%
- Storage breakdown: ~398.5 MB weights + ~12.4 MB packed sc/m + ~3.4 MB shared d/dmin (float8) + ~1.3 MB sf (4-bit packed)
- Learned: Key insight — compressed storage format can differ from dequantized weight computation. Storage-only compression preserves quality while reducing size. 4-bit scale factors for d/dmin ratios are precise enough for lossless storage reconstruction. Separate scale factors for d and dmin required (cannot share).
