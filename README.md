# Q4_K follow-up fixes

## Files

- `inference.py`: preserves serialized FP32 `d/dmin`, initializes model buffers correctly, validates the checkpoint strictly, avoids the permanent FP32 metadata cache, and supports a full-layer parity mode.
- `quantize.py`: adds `--q4k-refine-mode legacy_exact|alternating|none`.
- `q4k_diagnostics.py`: compares packed bytes with an old dequantized BF16 checkpoint without loading the full model.

## Why the previous "fp32" benchmark was not actually fp32

The supplied inference code contains:

```python
packed_data[d_key].to(dtype=torch.float16)
```

and the same for `dmin`. That downcasts FP32 scales on load. The corrected loader retains their serialized dtype and validates it against the manifest.

## Reproduce the historical d46e089 quantizer

The historical path assigned `q` once and held it fixed during the LS solve. The newer patch alternated LS solves with new code assignments. Those are different quantizers even if the latter has lower weight MSE.

```bash
python quantize.py \
  --model Qwen/Qwen3.5-9B \
  --format q4_k \
  --q4k-scale-dtype float32 \
  --q4k-refine-mode legacy_exact \
  --save quantized_models/qwen35-9b-q4k-legacy-match
```

Use `alternating` when optimizing the new format rather than reproducing the old benchmark.

## Separate codec, quantizer, and GEMM differences

Compare sampled packed weights with the old dequantized checkpoint:

```bash
python q4k_diagnostics.py \
  quantized_models/qwen35-9b-q4k-legacy-match \
  --reference-dir path/to/old-dequantized-bf16-model \
  --layers 12 \
  --rows 32
```

- Exact BF16 match means the codec and quantizer match the old checkpoint.
- A mismatch means the serialized weights differ; runtime changes cannot fix that.

Then run the evaluator with one full GEMM per linear layer:

```bash
Q4K_TILE_ROWS=0 python eval_perplexity.py ...
Q4K_TILE_ROWS=0 python eval_topk.py ...
```

`Q4K_TILE_ROWS=0` temporarily dequantizes the whole current layer. It uses more peak VRAM, but it removes output-row tiling as a source of numerical drift.

Interpretation:

1. Full-layer mode matches the old benchmark but tile 512 does not: remaining difference is GEMM kernel/rounding caused by tiling.
2. Full-layer mode still differs and diagnostics report different weights: quantizer semantics differ; use `legacy_exact` and re-quantize.
3. Diagnostics report exact weights but full-layer model still differs: check non-weight state and attention backend. The corrected loader avoids manual RoPE reconstruction by creating meta parameters while letting constructor buffers initialize normally.

## Important

Re-quantization is required for `legacy_exact`. Changing only `inference.py` cannot change codes or LS-refined scales already stored in an existing checkpoint.
