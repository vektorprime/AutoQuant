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
Hypothesis: Share d/dmin across K=4 output channels, store shared values (float8) + 4-bit per-channel scale factors. Storage-only compression.
Outcome: WIN — 415.6 MB, KL=0.092, Top-P=83.86%.

## exp-20260710-022 (Delta-encode shared d/dmin)
Hypothesis: Shared d/dmin values across sequential superblocks are correlated. Store first block as 6-bit base, remaining as 4-bit signed deltas.
Outcome: WIN — 411.8 MB, KL=0.092, Top-P=83.86%. NEW GLOBAL BEST.

## exp-20260710-impw (Importance-weighted LS)
Hypothesis: Weight LS error by activation importance from calibration data.
Outcome: REGRESSED — KL=0.090 (improved), Top-P=83.60% (degraded). Weighting improves KL but harms Top-P agreement.

## exp-shared-sm (Shared scales/mins across channels)
Hypothesis: Share 6-bit scales/mins across K_sc=2 output channels with 4-bit per-channel scale factors (storage-only).
Outcome: REGRESSED — 417.8 MB (larger). sf arrays add more overhead than shared sm_packed saves.

## exp-20260710-023 (Shared scales/mins K=4 with 2-bit deltas)
Hypothesis: Share 6-bit scales/mins across K_sc=4 channels with 2-bit multiplicative deltas packed into 4 bytes/superblock/channel (vs previous 12). Storage-only.
Outcome: **WIN** — 397.2 MB, KL=0.092, Top-P=83.86%. **NEW GLOBAL BEST**. Saves 14.6 MB vs previous best (14.6/411.8 = 3.5%). Key improvement: K=4 sharing halved shared sm cost, 2-bit deltas only 4 bytes/channel vs 12 saved.

## exp-20260710-024 (Combined 10-bit scale+min)
Hypothesis: Scale and min within a sub-block are correlated. 10-bit joint coding (5+5) replaces 6+6, saves 2 bytes per group per superblock.
Outcome: **WIN** — 395.8 MB, KL=0.092, Top-P=83.86%. **NEW GLOBAL BEST**.

## Summary
- Best: K=4 shared d/dmin + delta-encoded d/dmin + K=4 shared scales/mins with 2-bit deltas + 10-bit joint scale/min coding + 3-pass LS → 395.8 MB, KL=0.092, Top-P=83.86%
