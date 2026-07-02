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
