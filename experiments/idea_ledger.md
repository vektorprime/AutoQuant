# Idea Ledger — AutoQuant Q4_K experiments

## exp-20260710-013

Hypothesis: Reducing sub-blocks from 8 to 6 (irregular sizes [43,43,43,43,42,42]) saves 3 bytes metadata per superblock (~2.1% smaller) and LS refinement can compensate for the coarser sub-block granularity.
Algorithm family: Q4_K sub-block reduction
Changed code: _quantize_one_layer_q4k_sub6_ls (new function), constants QK_K_SUB6_BLOCKS/SIZES, routing in quantize_model
Representation change: 6 sub-blocks instead of 8 within QK_K=256 superblock
Storage risk: saves ~11.8 MB (3 bytes per 256 weights)
VRAM risk: none
Expected win: smaller size with LS compensating quality
Outcome: REGRESSED — KL 0.107867 (vs baseline 0.093508), Top-P 82.34% (vs 83.56%). 6 sub-blocks lose too much sub-block granularity; LS cannot compensate.
