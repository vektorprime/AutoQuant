# Synthesis checkpoint — 2026-07-10

- Q4_K baseline: KL=0.0935, top-P=83.56%, size=434.6 MB
- **Quality bar to beat**:
  - Size must be **< 434.6 MB** (strictly smaller than Q4_K)
  - KL must be **≤ 0.0935** (same or better than Q4_K)
  - Top-P must be **≥ 83.56%** (same or better than Q4_K)
- Current global best: Q4_K baseline (no valid novel experiments yet)
- Current phase: Phase 1 (novel encoding and compression schemes)
- Next hypotheses to try:
  1. Multi-codebook additive quantization (AQLM-style: W ≈ Q1 + Q2)
  2. Hadamard rotation + quantization
  3. Vector quantization of weight blocks
  4. Codebook deduplication with delta coding
