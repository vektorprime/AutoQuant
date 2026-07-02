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

## exp-20260701-009

Hypothesis:
Activation-weighted MSE scale using calibration-computed E[x²] per input channel
better aligns quantization error with downstream KL than plain weight MSE.

Algorithm family:
activation_weighted

Changed code:
_quantize_one_layer — added calibration pass, weighted scale refinement loop

Representation change:
none (still 2-bit symmetric `{-2s, -s, 0, s}`)

Storage risk:
none

VRAM risk:
none (in-place reuse, calibration stats are 1 scalar per input channel)

Expected win:
lower KL at same groupsize

Outcome:
KL improved (6.69 → 6.34, groupsize=32).  New global best.

## exp-20260701-010

Hypothesis:
The tiny SSM projections `linear_attn.in_proj_a` and `linear_attn.in_proj_b`
(32K elements each) have negligible storage impact but quantizing them at 2-bit
may hurt quality disproportionately.  Skipping them entirely should improve KL.

Algorithm family:
layer_policy

Changed code:
quantize_model — added `_NEVER_QUANTIZE` skip list

Representation change:
none (skipped layers stay at BF16; quantized layers unchanged)

Storage risk:
trivial (36 skipped layers × 32K elements × 2 bytes = ~2.4 MB still at bf16)

VRAM risk:
none

Expected win:
lower KL with minimal size increase

Outcome:
KL improved (6.34 → 5.94, groupsize=32).  New global best.

## exp-20260701-009

Hypothesis:
Using activation-weighted MSE (E[x^2] per input channel from Wikitext-2 train split)
in the scale refinement loop produces scales better aligned with output KL than
plain weight MSE.

Algorithm family:
activation_weighted

Changed code:
_quantize_one_layer (weighted scale update), _collect_input_stats (new), main/quantize_model (plumbing)

Representation change:
none (still 2-bit symmetric {-2s, -s, 0, s})

Storage risk:
none (no extra metadata)

VRAM risk:
transient during calibration (16 forward passes on GPU), well under 8 GB

Expected win:
lower KL at same groupsize

Outcome:
KL improved (6.69 -> 6.34, groupsize=32). New global best.

## exp-20260702-012

Hypothesis:
Attention projection layers (q/k/v/o_proj) and lm_head directly shape model outputs and are
more sensitive to quantization error. Using finer groupsize=16 for them gives better scale
resolution, while MLP layers (gate/up/down_proj) tolerate coarser groupsize=32.

Algorithm family:
layer_policy

Changed code:
quantize_model — added _get_layer_groupsize() per-layer policy, integration in layer loop

Representation change:
none (still 2-bit symmetric {-2s, -s, 0, s})

Storage risk:
moderate (1089 MB vs 898 MB baseline; well under 1575 MB limit)

VRAM risk:
none (same calibration + CPU quantization pattern)

Expected win:
lower KL from finer quantization of sensitive layers

Outcome:
KL improved (5.94 -> 5.24, groupsize=32 with gs=16 for attention+lm_head). New global best.

