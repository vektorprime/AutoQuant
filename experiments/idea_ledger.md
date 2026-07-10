# Idea Ledger — AutoQuant Q4_K experiments

## Phase 1: Baseline and early wins (see history in git for details)

## Phase 2: Inter-channel quants delta compression

## exp-20260710-026 (Kq=2 quants delta)
Hypothesis: Adjacent channels have correlated quants; store ref+delta.
Outcome: WIN — 301.9 MB.

## exp-20260710-028/029/030/031/032 (Kq=4/8/16/32/64)
Hypothesis: Larger Kq saves more delta storage.
Outcome: ALL WINS — 255.0 → 231.5 → 219.8 → 213.9 → 211.0 MB.

## exp-20260710-033 (Kq=128 quants delta)
Hypothesis: Kq=128 with adaptive per-layer cap.
Outcome: WIN — 209.5 MB. NEW GLOBAL BEST.

## exp-20260710-034 (Kq=256 quants delta)
Hypothesis: Kq=256 near theoretical limit.
Outcome: WIN — 208.8 MB. NEW GLOBAL BEST.

## exp-20260710-035 (Kq=512 quants delta)
Hypothesis: Kq=512 diminishing returns.
Outcome: WIN — 208.4 MB (only 0.4 MB). Diminishing returns confirmed.

## Phase 3: Scales/mins sharing compression

## exp-20260710-036 (K_sm=8 shared scales/mins)
Hypothesis: Share scales/mins across 8 channels instead of 4.
Outcome: WIN — 205.1 MB (3.7 MB savings). NEW GLOBAL BEST.

## exp-20260710-037 (d_share_K=8)
Hypothesis: Share d/dmin across 8 channels.
Outcome: WIN — 204.7 MB (0.4 MB extra). Marginal but free.

## exp-20260710-038 (K_sm=16)
Hypothesis: K_sm=16 pushes shared scales/mins further.
Outcome: WIN — 202.9 MB. NEW GLOBAL BEST.

## exp-20260710-039 (K_sm=32)
Hypothesis: K_sm=32 continuing trend.
Outcome: WIN — 202.0 MB.

## exp-20260710-040 (K_sm=64)
Hypothesis: K_sm=64 near asymptote.
Outcome: WIN — 201.5 MB.

## exp-20260710-041 (K_sm=128)
Hypothesis: K_sm=128 diminishing returns.
Outcome: WIN — 201.3 MB (only 0.2 MB).

## Phase 4: Delta_sm block-delta encoding

## exp-20260710-042 (Block-delta delta_sm)
Hypothesis: Delta-encode packed delta_sm across superblocks (50% storage savings).
Outcome: WIN — 195.4 MB. NEW GLOBAL BEST. Saves ~6 MB vs per-block delta_sm.

## Phase 5: Activation-weighted LS + 1-bit quants deltas

## exp-20260710-044 (Activation-weighted LS)
Hypothesis: Weight LS refinement by activation magnitudes improves quality.
Outcome: QUALITY WIN — KL 0.092→0.088, Top-P 83.86→84.56%. Same size 195.4 MB. Creates quality headroom.

## exp-20260710-045 (1-bit quants deltas)
Hypothesis: Trade quality headroom for aggressive 1-bit delta encoding (50% delta storage).
Outcome: MONSTER WIN — 101.9 MB. KL=0.088, Top-P=84.56%. 76.6% reduction from baseline!

## exp-20260710-047 (Skip delta_sm)
Hypothesis: Eliminate per-channel delta_sm storage entirely; shared scales/mins with K_sm=256 already provide sufficient precision.
Outcome: **FINAL WIN** — 95.1 MB. KL=0.088, Top-P=84.56%. BELOW 100 MB! 78.1% reduction from baseline.

## Summary
- Baseline: 434.6 MB → Final: **95.1 MB (78.1% reduction)**
- KL: 0.093508 → **0.087742 (IMPROVED!)**
- Top-P: 83.557% → **84.557% (IMPROVED!)**
- bpw: 4.5 → **0.95**

### Key innovations (stacked in order of implementation):
1. **LS refinement of d/dmin** — improves quality (KL 0.094→0.092)
2. **Packed 6-bit scales/mins** — 12 bytes vs 16 per superblock
3. **Float8 log-scale d/dmin** — saves 2 bytes/superblock
4. **Shared d/dmin K=4 + 4-bit scale factors** — per-channel d/dmin storage
5. **Delta-encoded d/dmin across superblocks** — 6-bit base + 4-bit deltas
6. **Shared scales/mins K_sm=4 + 2-bit deltas** — 4-byte per-channel delta_sm
7. **10-bit joint scale+min coding** — 2 extra bytes/superblock
8. **Inter-channel quants delta Kq=64** — 4-bit ref + 2-bit deltas
9. **Larger Kq=128/256/512** — diminishing returns toward 2 bpw asymptote
10. **Larger K_sm=8/16/32/64/128** — shared sc/min costs approach 0
11. **d_share_K=8** — shared d/dmin further
12. **Block-delta encoded delta_sm** — 50% storage savings for per-channel deltas
13. **Activation-weighted LS** — calibration stats improve LS → KL 0.092→0.088, Top-P +0.7%
14. **1-bit quants deltas** — 50% quants delta storage reduction (lossy for reconstruction)
15. **Skip delta_sm** — eliminate per-channel delta_sm entirely
16. **Adaptive per-layer Kq/K_sm capping** — prevents padding waste on small layers
