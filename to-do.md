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

## 4. Quantize from BF16 base and store compressed weights to disk
- Currently `quantize.py` loads the model in float16, quantizes in-place (simulated
  quantization), and saves dequantized float16 weights via `model.save_pretrained()` —
  file size stays at ~3.6GB, no compression on disk.
- This is a consequence of the current approach: GPTQ stores dequantized values
  (e.g., "0.0342" instead of int code "3"). The eval script loads the float16
  safetensors back, so no actual 4-bit storage is achieved.
- We need to:
  1. Load the base model in BF16 (preserve full precision before quantization).
  2. Apply quantization and store the actual compressed representation — integer
     codes + scales + zeros — so the file size reflects real 4-bit compression
     (~0.9GB for 4-bit instead of ~3.6GB).
  3. The eval script must be able to load and reconstruct (dequantize) these
     compressed weights for inference.

## 5. No `.gitignore` — git tries to stage large cache files
- Without `.gitignore`, `git add -A` tried to stage `cache/ref_logits.mmap` (238MB),
  causing the commit to time out.
- **Fix**: Created `.gitignore` with `cache/`, `__pycache__/`, `quantized_models/`, `*.mmap`.
- Must verify `.gitignore` exists at experiment setup time.

## 6. README.md outdated
- Still references "perplexity" as the metric — should say KL divergence.
- Still frames the project as purely GPTQ — the agent can explore any quantization
  approach (q4_k, AWQ-style, etc.) as long as it fits `quantize.py`.
- Starting point is q4_k (4-bit symmetric group quantization), not vanilla GPTQ.

## 7. Replace GPTQ with q4_k
- The current `quantize.py` implements vanilla GPTQ with optimal rounding.
- The goal is to replace it with q4_k — a simpler, faster 4-bit quantization that
  uses per-block (group) min/max scaling without iterative error propagation.
- q4_k should be significantly faster (no Cholesky, no column-wise updates) and
  provides a clean baseline before exploring more sophisticated algorithms.

## 8. Pipeline verified end-to-end
- Reference cache creation: works, ~28s for 5000 tokens
- Quantization: works, ~5 min for 4 calib samples
- Eval with cache hit: works, ~28s (no reference model loaded)
- Cleanup: `rm -rf quantized_models/<tag>` works
- Results recording: tab-separated TSV works
- Git commit + branch: works (after fixes above)
