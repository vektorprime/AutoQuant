# Idea Ledger

## exp-20260701-003

Hypothesis:
Using the 90th-percentile abs-max instead of the absolute maximum will be more
robust to outliers, reducing quantization error.

Algorithm family:
scale_optimization

Changed code:
_quantize_one_layer

Representation change:
none (still 2-bit symmetric)

Storage risk:
none

VRAM risk:
none

Expected win:
lower KL at same groupsize

Outcome:
KL regressed (13.55 vs baseline 8.53 at groupsize=32)

## exp-20260701-005

Hypothesis:
A no-clipping scale `s = max(amax, |amin|/2)` ensures the 2-bit range `{-2s, -s, 0, s}`
exactly covers `[amin, amax]` with zero clipping, which should reduce distortion.

Algorithm family:
scale_optimization

Changed code:
_quantize_one_layer

Representation change:
none

Storage risk:
none

VRAM risk:
none

Expected win:
lower KL at same groupsize

Outcome:
KL regressed (11.71, groupsize=32).  Larger scale = coarser bins = higher MSE.

## exp-20260701-006

Hypothesis:
Error-diffusion quantization (Floyd-Steinberg residual propagation across groups)
will allow later groups to compensate for earlier quantization errors, reducing KL.

Algorithm family:
error_compensation

Changed code:
_quantize_one_layer

Representation change:
none (still 2-bit symmetric `{-2s, -s, 0, s}`)

Storage risk:
none (no extra metadata)

VRAM risk:
none (in-place residual add)

Expected win:
lower KL at same groupsize

Outcome:
KL improved (8.53 → 8.43, groupsize=32).  Best current result.

## exp-20260701-008

Hypothesis:
Iterative least-squares scale refinement (3 EM-like iterations per group) on top of
error-diffusion further reduces per-group weight MSE, lowering KL.

Algorithm family:
mse_opt

Changed code:
_quantize_one_layer — added 3-iteration scale refinement loop after initial maxabs scale

Representation change:
none (still 2-bit symmetric `{-2s, -s, 0, s}`)

Storage risk:
none (no extra metadata)

VRAM risk:
none (in-place tensor reuse)

Expected win:
lower KL at same groupsize

Outcome:
KL improved (8.43 → 6.69, groupsize=32).  New global best.
