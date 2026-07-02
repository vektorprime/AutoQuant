# In-progress experiments (one agent at a time)

exp-20260702-002: Adaptive error-diffusion per layer type (attn=0.3, mlp=0.7, lm_head=0.1) with activation-weighted MSE + per-layer gs policy
exp-20260702-003: Clipped-scale quantization using 99th-percentile instead of maxabs for initial scale with MSE refinement
exp-20260702-004: Layer-type-specific error diffusion (attn=0.7, mlp=0.3, lm_head=0.1, default=0.5)
exp-20260702-005: Post-quantization global scale refinement against original weights
