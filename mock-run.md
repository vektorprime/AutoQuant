# Mock Run Issues — AutoQuant Auto Loop

## 1. Reference cache deleted unnecessarily
- `rm -rf cache/` between runs defeats the purpose of caching.
- The cache should be created once and preserved across all experiment runs.
- The auto loop must not delete the cache directory.

## 2. Repeated model metadata checks on every run
- `quantize.py` does ~13 HTTP HEAD requests to HuggingFace on every invocation
  (config.json, tokenizer_config.json, model.safetensors.index.json, etc.).
- The actual weights ARE cached locally; these are just version/availability checks.
- This adds ~5-10 seconds of overhead per experiment — negligible vs quantization time,
  but still wasteful. Could be mitigated by setting `HF_HUB_OFFLINE=1` after first run.

## 3. Quantization speed
- With 4 calib samples × seqlen=1024: **~5 min** for 24 blocks (~12s/block) on an RTX 3080.
- With 128 calib samples (default): per-block time increases ~2-3x → estimated **~12 min** total.
- Bottlenecks per block:
  - Forward pass through all linear layers with all calib samples (Hessian accumulation)
  - GPTQ column-wise error propagation on large MLP layers (in_features=6144)
- The auto loop needs fast iterations. Suggestion: use fewer calib samples initially,
  then run best candidate with more samples.

