# Synthesis checkpoint — 2026-07-10

- Q4_K baseline: KL=0.0935, top-P=83.56%, size=434.6 MB
- **Quality bar to beat**:
  - Size must be **< 434.6 MB** (strictly smaller than Q4_K)
  - KL must be **≤ 0.0935** (same or better than Q4_K)
  - Top-P must be **≥ 83.56%** (same or better than Q4_K)
- Current global best: Q4_K baseline (no experiments beyond baseline yet)
- Current phase: Phase 1 (bit-width frontier mapping)
- Ideas that improved: none yet
- Ideas that regressed: none yet
- Failure patterns: none yet
- Next three highest-priority hypotheses:
  1. Re-evaluate existing 2-bit K-means pipeline (KL=2.47, 243 MB) — much smaller but 26× worse KL; revisit with Q4_K quality target
  2. 3-bit K-means with Q8 codebook compression at gs=64 (~325 MB expected) — 8 levels instead of 4, likely much better KL
  3. Mixed-precision: 4-bit attention + 2-bit FFN — saves size on less sensitive layers
