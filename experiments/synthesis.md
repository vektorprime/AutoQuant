# Synthesis checkpoint — 2026-07-02

- Current global best KL: 5.24
- Best code sha: 7b7f546879402b89579d4ce94219c338b8374c73
- Best groupsize: 32 (with 16 for attention+lm_head per-layer policy)
- Ideas that improved:
  1. Error-diffusion (Floyd-Steinberg, diffusion=0.5): 8.53 → 8.43
  2. MSE-optimal per-group scale (3 EM iterations): 8.43 → 6.69
  3. Activation-weighted MSE (E[x^2] calibration): 6.69 → 6.34
  4. Skip SSM tiny projections (linear_attn.in_proj_a/b): 6.34 → 5.94
  5. Per-layer groupsize policy (gs=16 for attention+lm_head, gs=32 for MLP): 5.94 → 5.24
- Ideas that regressed:
  1. 90th-percentile scale outlier robustness: 8.53 → 13.55
  2. No-clipping scale max(amax, |amin|/2): 8.53 → 11.71
  3. bf16 compact format (scalar zero_pt + bf16 scales): 5.94 → 6.28
- Failure patterns:
  - Outlier-robust scale methods (percentile, no-clip) hurt at 2-bit; MSE-optimal is consistently better
  - Format changes (compact storage) that alter representation gave no KL benefit
- Next three highest-priority hypotheses:
  1. Adaptive error-diffusion coefficient per layer-type (e.g. MLP gets higher diffusion)
  2. Activation-weighted scale with diagonally-approximated Hessian (second-order info)
  3. Cross-channel grouping: share scales across 2 adjacent output channels to reduce storage, allowing groupsize=16 everywhere
- Current phase: Phase 1 (no-format-change improvements) progressing into Phase 2 (activation-aware)
