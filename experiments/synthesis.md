# Synthesis checkpoint — 2026-07-10T03:25Z

- Q4_K baseline: KL=0.0935, top-P=83.56%, size=434.6 MB
- Current global best: Q4_K + zlib codes+metadata at 408.5 MB (KL=0.0935, top-P=83.56%)
- Best code sha: c1d5df5 (q4_k_zlib2)
- Best effective bits: 4.5 nominal
- Ideas that improved:
  1. Zlib on codes: 423.3 MB (-2.6%) — same quality
  2. Zlib on codes+metadata: 408.5 MB (-6.0%) — same quality, GLOBAL BEST
- Ideas that regressed:
  1. XOR delta+zlib: 434.4 MB (increases entropy) — REVERTED
  2. Large sub-blocks (4 vs 8): KL +49%, top-P -3.6pp — REVERTED
  3. 3-bit Q4_K 16SB: KL 3.6x worse, top-P -14.1pp — REVERTED
  4. Bz2 on codes: 432.5 MB (worse than zlib) — REVERTED
  5. Per-channel bias: KL +0.3%, top-P -0.22pp — REVERTED
  6. 5-bit scales/mins: KL +3.3%, top-P -0.44pp — REVERTED
- Failure patterns:
  1. LZMA too slow for 151 layers (~3min for saving alone)
  2. XOR delta randomizes already-near-random byte stream
  3. 3-bit quantization (even with fine adaptation) can't close the gap to 4-bit
  4. Sub-block scale precision is critical — reducing below 6-bit hurts quality measurably
  5. Per-channel bias correction over-corrects zero-mean Q4_K errors
- Key insight: Q4_K's 6-bit per-sub-block scales are at the precision limit — reducing them regresses quality. The codes (89% of storage) are near-random, limiting compressibility.
- Next phase: Phase 2 — representation innovation (Hadamard rotation, vector quantization, learned codebooks)
- Next three highest-priority hypotheses:
  1. Hadamard rotation of sub-blocks before quantization — uniformizes distributions, potentially enabling 3.5-bit or larger groups
  2. Vector quantization: 4 weights → 8-bit codebook index → 2 bits/weight effective
  3. Multi-codebook additive (AQLM-style) with large groups to amortize codebook overhead
