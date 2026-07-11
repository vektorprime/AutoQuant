# Q4_K — AutoQuant experiment (Qwen3.5-9B)

This experiment searches for a **novel quantization technique** that is **smaller**
than the Q4_K format while matching or beating Q4_K's quality on two metrics:
- **KL divergence** against BF16 reference (lower = better)
- **Top-P (argmax) agreement** with BF16 reference (higher = better)

The baseline is **Q4_K** (`legacy_exact` LS, fp32 scales) applied to
**Qwen/Qwen3.5-9B** using packed-only inference. Every proposed technique must
produce a model **smaller** than Q4_K's packed size with **no quality regression**.

---

## Baseline

| Metric | Value |
|---|---|
| Model | Qwen/Qwen3.5-9B |
| Format | `q4k_affine_v2` (safetensors + manifest + residual) |
| Refine mode | `legacy_exact` (LS on d/dmin, codes held fixed) |
| Scale dtype | fp32 |
| KL divergence | 0.059 |
| Top-P agreement | 88.12% |
| Packed size | 6.63 GB (4.58 GB packed + 2.05 GB residual) |
| Eval params | ctx=256 stride=128 max_tokens=4000 |
| Reference cache | `cache/ref_logits_9B.mmap` |

---

## Architecture

### Files you edit
- `quantize.py` — quantization algorithm, encode/decode, safetensors save
- `inference.py` — packed-only GPU loader, `QuantizedLinear` with tiled dequant

### Files you read
- `eval_perplexity.py` — KL divergence evaluation (routes through `load_quantized_model`)
- `eval_topk.py` — top-P agreement evaluation (routes through `load_quantized_model`)
- `q4k_diagnostics.py` — packed-vs-checkpoint comparison tool
- `test_q4k_codec.py` — encode/decode roundtrip tests

### Format architecture
```
quantized_models/<tag>/
  q4k_manifest.json          — layer list, shapes, scale dtype, biases
  q4k_affine_v2-*.safetensors — packed quants, d/dmin, scales/mins
  residual.safetensors        — embeddings, norms, biases, non-quant layers
  config.json, tokenizer files
```

### QuantizedLinear (inference.py)
- Per-superblock: d (fp32), dmin (fp32), 8×6-bit scales/mins packed in 12 bytes
- 4-bit quants packed as uint8 byte-pairs
- Tiled dequantization: 512-row tiles, weights decoded per tile
- No persistent VRAM cache — decode happens each forward pass

---

## What counts as NOVEL

**Valid:**
- New quantization algorithms / codebook structures / encoding schemes
- New metadata compression: shared d/dmin, shared scales/mins, delta encoding of metadata
- New quants compression: inter-channel delta, sub-block delta, learned references
- Transformations applied to weights before/after quantization (e.g. rotation)
- Techniques that generalize to arbitrary models (not Qwen-specific)

**NOT valid:**
- Mixing existing quant types without novelty
- Parameter tuning without algorithmic change
- Skipping layers or quantizing only a subset
- Generic post-hoc compression (zlib, bz2, etc.)
- Exploiting Qwen-specific architecture

---

## Sub-agent experiment protocol

Experiments are run by sub-agents. Each sub-agent:
1. Implements the technique in `quantize.py` and `inference.py`
2. Quantizes: `CUDA_VISIBLE_DEVICES=2 .venv/bin/python quantize.py --model Qwen/Qwen3.5-9B --format q4_k --q4k-scale-dtype float32 --q4k-refine-mode legacy_exact [--new-flags] --save quantized_models/qwen35-9b-<tag>`
3. Evals KL: `CUDA_VISIBLE_DEVICES=2 Q4K_TILE_ROWS=512 .venv/bin/python eval_perplexity.py --model quantized_models/qwen35-9b-<tag> --reference Qwen/Qwen3.5-9B --context-length 256 --max-tokens 4000 --reference-cache cache/ref_logits_9B.mmap`
4. Evals top-P: `CUDA_VISIBLE_DEVICES=2 Q4K_TILE_ROWS=512 .venv/bin/python eval_topk.py --model quantized_models/qwen35-9b-<tag> --reference Qwen/Qwen3.5-9B --reference-cache cache/ref_logits_9B.mmap --context-length 256 --max-tokens 4000 --stride 128`
5. Records results in `results.tsv` and `idea_ledger.md`
6. Commits and pushes

### Sub-agent rules
- Only ONE sub-agent runs at a time
- Always revert failed experiments (do NOT commit regression code)
- Use GPU 2 (`CUDA_VISIBLE_DEVICES=2` — RTX 3080, 20 GB)
- Never use `torch.compile` (it breaks on per-group tensor access)
- Never create persistent VRAM caches in `QuantizedLinear` (OOM risk)
- Always pass `Q4K_TILE_ROWS=512`
- Reference cache: `cache/ref_logits_9B.mmap` (DO NOT regenerate)
- HF_HUB_OFFLINE=1 for all eval commands

### Quality bar
| Metric | Threshold |
|---|---|
| KL divergence | ≤ 0.065 (within 0.006 of baseline) |
| Top-P agreement | ≥ 88.0% (within 0.5% of baseline) |
| Model size | < 6.63 GB (strictly smaller than baseline) |

---

## Git conventions
- Branch: `BASE` (main development)
- Commit code before evaling
- Revert failed experiments (git reset --hard) — do NOT merge broken code
- Push after each working commit
- Record every experiment in `results.tsv` — include regressions
- Update `idea_ledger.md` after each experiment with hypothesis, implementation details, and failure analysis (see format below)

---

## Idea ledger (`idea_ledger.md`)

Every experiment — success or failure — must record a structured entry:

```markdown
## <exp_id> — <technique name>

**Hypothesis:** <one sentence>
**Status:** success | regression | failure
**KL divergence:** <value>  |  **Top-P:** <value>  |  **Size:** <value> GB

**Implementation:**
- <how it works, what changes were made to quantize.py / inference.py>
- <key parameters (Kq, K_sm, bit widths, sharing factors)>

**Result:**
<why it succeeded or failed — root cause analysis>

**Lesson:**
<what this teaches us for future experiments>
```

After each experiment, the sub-agent appends a new entry to the top of `idea_ledger.md`.

---

## GPU layout
| CUDA index | GPU | VRAM | Status |
|---|---|---|---|
| 0 | RTX 3080 | 20 GB | Kernel issues (causal_conv1d) |
| 1 | RTX 3050 | 6 GB | Too small for 9B |
| 2 | RTX 3080 | 20 GB | **Use this for evals** |
| 3 | RTX 5090 | 32 GB | No causal_conv1d kernels |
