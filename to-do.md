# Mock Run Issues — AutoQuant Auto Loop

## 1. Reference cache deleted unnecessarily
- `rm -rf cache/` between runs defeats the purpose of caching.
- The cache should be created once and preserved across all experiment runs.
- The auto loop must not delete the cache directory.

## 2. Repeated model metadata checks on every run
- `quantize.py` does ~13 HTTP HEAD requests to HuggingFace on every invocation.
- **Mitigation**: Use `HF_HUB_OFFLINE=1` environment variable — verified working
  (eval ran without any HTTP requests).
- Must set this in the auto loop's shell environment.

## 3. Quantization speed
- With 4 calib samples × seqlen=1024: **~5 min** for 24 blocks (~12s/block) on an RTX 3080.
- With 128 calib samples (default): per-block time increases ~2-3x → estimated **~12 min** total.
- Bottlenecks per block:
  - Forward pass through all linear layers with all calib samples (Hessian accumulation)
  - GPTQ column-wise error propagation on large MLP layers (in_features=6144)
- The auto loop needs fast iterations. Suggestion: use fewer calib samples initially,
  then run best candidate with more samples.
- Groupsize can be as low as 16 (must divide in_features). Smaller groupsize = finer
  quantization = better accuracy but larger scale/zero overhead in compressed storage.

## 4. Quantize from BF16 base and store compressed weights to disk ✅
- **Done**: Load model in `bfloat16`, quantize all nn.Linear layers using per-group
  min/max, save packed 2-bit codes + scales + zeros to `compressed/` subdirectory.
- Compressed size for Qwen3.5-2B: **328 MB** (vs 3.6 GB float16 dequantised).
- Eval compatibility: dequantised weights saved via `model.save_pretrained()`.

## 5. No `.gitignore` — git tries to stage large cache files
- Without `.gitignore`, `git add -A` tried to stage `cache/ref_logits.mmap` (238MB),
  causing the commit to time out.
- **Fix**: Created `.gitignore` with `cache/`, `__pycache__/`, `quantized_models/`, `*.mmap`.
- Must verify `.gitignore` exists at experiment setup time.

## 6. README.md outdated ✅
- **Fixed**: Updated to reference KL divergence (not perplexity), q2_k (not GPTQ).

## 7. Replace GPTQ with q2_k ✅
- **Done**: Rewrote `quantize.py` — removed GPTQLayer, Hessian accumulation, Cholesky,
  calibration data, and block-wise error propagation.
- New algorithm: simple per-layer Quantizer.find_params() + Quantizer.quantize()
  for all nn.Linear layers. No calibration data needed.
- Quantization time: ~4 min for 187 layers on RTX 3080 (single-threaded Python loop).

## 8. Pipeline verified end-to-end

## 9. Quantization speed — single-threaded bottleneck
- The per-layer Python loop over 187 linear layers runs sequentially on one CPU core.
- Each iteration does GPU ops (find_params, quantize, code extraction) followed by
  blocking GPU→CPU copies for codes/scales/zeros.
- Total: ~4 min for q2_k on Qwen3.5-2B (187 layers, comparable to GPTQ's ~5 min
  for 24 blocks × ~8 layers each).
- Potential optimisations: accumulate all metadata on GPU and do a single bulk
  CPU transfer at the end, or batch-process small layers together.

## 10. Track size, KLD, and inference speed — decide on KLD
- Each experiment should record: compressed model size (MB), KL divergence, and
  inference speed (tokens/sec during eval).
- **Decision criterion**: keep changes only if KLD improves (lower). Size and speed
  are tracked for visibility but do not determine whether to advance.
- The `results.tsv` header and recording logic should include these additional columns.
- Reference cache creation: works, ~28s for 5000 tokens
- Quantization: works, ~5 min for 4 calib samples
- Eval with cache hit: works, ~28s (no reference model loaded)
- Cleanup: `rm -rf quantized_models/<tag>` works
- Results recording: tab-separated TSV works
- Git commit + branch: works (after fixes above)
