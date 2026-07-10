# Q4_K — AutoQuant experiment

This experiment searches for a **novel quantization technique** that is **smaller**
than the Q4_K format while matching or beating Q4_K's quality on two metrics:
- **KL divergence** against FP16 reference (lower = better)
- **Same-top-P agreement** with FP16 reference (higher = better)

The baseline is **Q4_K** applied to **Qwen/Qwen3.5-0.8B**. Every proposed technique
must produce a compressed model strictly smaller than Q4_K's compressed size.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on the technique name (e.g. `Q4_K`).
   The branch `autoresearch/<tag>` or the standalone branch name must not already exist.
2. **Create the branch**: `git checkout -b <tag>` from current HEAD.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `eval_perplexity.py` — fixed KL evaluation, read-only.
   - `eval_topk.py` — same-top-P evaluation, read-only.
   - `quantizer.py` — **read-only** reference. Understand the packed format,
     dequantization path, supported metadata, and what custom schemes are
     actually representable before inventing a new format.
   - `quantize.py` — the file containing the quantization algorithm (**you edit this**).
   - `data_utils.py` — data preparation utilities (read-only).
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
   The Q4_K baseline will be recorded as the first row.
5. **Initialize experiments/**: Create `experiments/idea_ledger.md` and
   `experiments/failures.tsv` (header row only).
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, establish the Q4_K baseline, then start experimentation.

---

## Research phases

The search follows a phased plan. Start at Phase 0 and move forward as each
phase is exhausted or hits diminishing returns. You may revisit earlier phases
after discoveries in later phases.

### Phase 0 — Q4_K baseline establishment

Before any experimentation, establish the Q4_K baseline:

1. Implement Q4_K quantization in `quantize.py`:
   - Superblock size: 256 weights
   - Per superblock: fp16 scale (`d`) and fp16 min (`dmin`)
   - 16 sub-blocks of 16 weights each
   - Per sub-block: 6-bit scale delta (12 bytes per superblock for all 16 sub-scales)
   - Per weight: 4-bit quantized value (128 bytes per superblock)
   - Total: 144 bytes per 256 weights → ~4.5 bits per weight

2. Quantize Qwen3.5-0.8B with Q4_K and save to `quantized_models/q4k_baseline`

3. Measure compressed size and record it

4. Generate Q4_K reference logits cache (optional — may reuse FP16 reference)

5. Run `eval_perplexity.py` against FP16 reference → record KL divergence

6. Run `eval_topk.py` against FP16 reference → record same-top-P agreement

7. Record the Q4_K baseline row in `results.tsv` with `exp_id = exp-q4k-baseline`

These three numbers (size, KLD, top-P) are the **quality bar**. Every proposed
technique must be **smaller** with KLD **≤ baseline** and top-P **≥ baseline**.

### Phase 1 — Bit-width frontier mapping

Explore the size-vs-quality frontier across different bit-widths and schemes:

- **3-bit quantization** with codebook compression (~340-380 MB expected)
- **2-bit quantization** variants (already explored in prior runs — may already
  beat Q4_K; verify against the new baseline)
- **2.5-bit schemes** (e.g., 3-bit codes with Huffman/entropy coding)
- **Mixed-precision**: higher bits for attention, lower for MLP
- **Per-channel variable bit-width**: assign more bits to high-activation channels

Record the Pareto frontier — for each size point, what's the best achievable KLD
and top-P? This guides later phases toward the most promising region of the
size/quality space.

### Phase 2 — Codebook and representation innovation

Go beyond simple per-group quantization:

- **Multi-codebook additive quantization**: W ≈ Q₁ + Q₂ + ... + Qₖ where each
  Qᵢ uses a small codebook (e.g., two 2-bit codebooks = 4 bits of expressiveness
  with less storage than a single 4-bit codebook)
- **Learned non-uniform quantization grids**: optimize quantization levels
  per tensor/channel via gradient descent or iterative refinement
- **Across-layer codebook sharing**: identify layers with similar weight
  distributions and share a single codebook pool
- **Vector/subspace quantization**: quantize groups of weights jointly using
  vector quantization (e.g., 4 weights → one 8-bit index into a 256-entry
  codebook of 4-vectors = 2 bits per weight)
- **GPTVQ-style**: larger block sizes with optimized lattice codebooks
- **Lookup-free quantization (LFQ)**: directly quantize to integer lattice
  without storing explicit codebooks

### Phase 3 — Pre/post-processing transforms

Transformations applied to weights before or after quantization:

- **Hadamard/random rotation**: multiply weight matrices by orthogonal transforms
  to make distributions more uniform before quantization (QuaRot-inspired).
  The inverse transform is fused into the next layer at load time.
- **Channel reordering/permutation**: find optimal input channel ordering that
  makes groups more homogeneous (already explored — revisit with new formats)
- **Outlier channel splitting**: isolate a small number of high-magnitude
  channels at higher precision, quantize the rest aggressively
- **Low-rank correction**: store a low-rank residual (SVD of quantization error)
  alongside quantized weights

### Phase 4 — Entropy coding and compression

Post-quantization coding to squeeze out redundancy:

- **Huffman/arithmetic coding** of quantized codes — exploit non-uniform code
  distributions for additional compression
- **Run-length encoding** for repetitive code patterns
- **Dictionary compression** across layers
- **Deduplication**: if two layers have identical quantized weights (unlikely
  but possible after aggressive quantization), store once

### Phase 5 — Hybrid approaches

Combine wins from earlier phases:

- Multi-codebook + Hadamard rotation + entropy coding
- Mixed-precision + outlier splitting + vector quantization
- The best composable techniques should multiply their benefits

---

## Experimentation

Each experiment runs on a single GPU. The workflow is:

1. Run `quantize.py` to quantize the model and save it to `quantized_models/<tag>`
2. Load the quantized model from `quantized_models/<tag>` and run `eval_perplexity.py`
   to evaluate KL divergence against the FP16 reference model.
3. Run `eval_topk.py` to measure same-top-P agreement.
4. Record both results in `results.tsv`.
5. Delete the quantized model from `quantized_models/<tag>`.

All experiments MUST be run with the following arguments for quantize.py:
- `--model Qwen/Qwen3.5-0.8B` (fixed — the reference model)
- `--bits <N>` — bit width (2, 3, 4, etc.; no longer fixed to 2)
- `--groupsize <N>` — group size (min 16, must divide `in_features`; default 32)
- `--dtype bfloat16` — load the base model in BF16 precision (fixed)
- `--save quantized_models/<tag>` — output directory

Symmetric quantization is **hardcoded** (always on). There is no `--symmetric` flag.

### Resource limits (enforced)

- **Compressed model size**: must be **strictly less than Q4_K's compressed size**
  (to be measured in Phase 0). `quantize.py` will enforce this limit.
  Smaller groupsize = more metadata = larger output.
- **VRAM**: max **8 GB** during quantization (checked via `nvidia-smi`). No
  allocating auxiliary tensors that persist across layers.

**What you CAN do:**
- Modify `quantize.py` — this is the only file you edit.
- Change the quantization algorithm **completely**. Invent new schemes, use
  iterative optimisation, implement multi-codebook quantization, add rotation
  transforms, use entropy coding, or anything else that maps float weights to
  a compact representation with size < Q4_K's.
- Use any bit-width (2, 3, 4, or even fractional effective bits via coding).
- Make multiple passes over the weights — extra compute is fine as long as
  VRAM stays under 8 GB.
- Add new functions, classes, or imports within `quantize.py` (no new packages).
- Vary `--groupsize` (≥ 16).
- **Extend `main()`** to add optional quantization-internal arguments, provided
  all official runs use consistent base arguments.
- **Any storage format** is allowed as long as the FP16 dequantized weights
  can be reconstructed from the compressed file and loaded via
  `AutoModelForCausalLM.from_pretrained()`.

**What you CANNOT do:**
- Modify `eval_perplexity.py` or `eval_topk.py`. They are read-only.
- Modify `quantizer.py` (the `Quantizer` class and `quantize_tensor` function).
- Modify `data_utils.py`.
- Install new packages or add dependencies.
- Exceed Q4_K's compressed size.
- Exceed 8 GB VRAM.
- Use more than one GPU.
- Use the Wikitext-2 test split during quantization (calibration only from train/val).

**The goal is simple: find the smallest possible representation that matches or
beats Q4_K's KL divergence and same-top-P agreement.**

### Why `quantizer.py` is off-limits

The `Quantizer` class and `quantize_tensor` function are the **default primitives**
that define affine quantization. The restriction means you cannot *edit* these
specific primitives — you can, however, write your own quantization logic from
scratch inside `quantize.py`.

---

### Calibration data rules

Quantization **may** use calibration data from a **non-test** split.
Do **not** use Wikitext-2 test split (`eval_perplexity.py --split test`) or the
eval reference logits (`cache/ref_logits.mmap`) during quantization.

**Allowed calibration statistics:**
- Per-layer input activation second moments `E[x_j^2]` (one scalar per input channel).
- Per-channel activation scales.
- Small streaming diagonal Hessian approximations.
- Wikitext-2 **train** or **validation** split.

**Forbidden calibration state:**
- Cached full activation tensors across layers (violates VRAM limit).
- Test-set activations.
- Reference logits from `eval_perplexity.py`.

---

### Diagnostics

Diagnostic runs are **allowed** if:
- They are **not** written to `results.tsv`.
- They do **not** use the Wikitext-2 test split.
- They do **not** modify `eval_perplexity.py` or `eval_topk.py`.
- They are clearly logged under `runs/<exp_id>/diagnostics/`.

Allowed diagnostics:
- Synthetic tensor roundtrip tests.
- Per-layer quantization error summaries.
- Calibration-split proxy KL or loss.
- Size and VRAM estimates before full runs.
- Layer sensitivity experiments on non-test data.

---

### Performance guidance

**Ranked idea queue.** Try these in order — early ideas are higher-probability
improvements:

1. **Re-evaluate 2-bit K-means** — The existing 2-bit pipeline (K-means codebooks,
   Q8 compression, error diffusion, activation weighting, channel sorting) may
   already beat Q4_K. Verify this first.

2. **3-bit with codebook compression** — 3-bit offers 8 levels per group vs 4,
   dramatically reducing quantization error. With Q8 codebook compression and
   a groupsize of 64-128, the storage may still be under Q4_K's size.

3. **Multi-codebook additive (AQLM-style)** — Represent W ≈ Q₁ + Q₂ where each
   uses a small 2-bit codebook. Two 2-bit codebooks = 4-bit expressiveness but
   less storage than 4-bit (only pay for two small codebooks per group).

4. **Hadamard rotation + quantization** — For each weight matrix W (out × in),
   apply a random Hadamard transform H: W' = W @ H. Quantize W'. At load time,
   apply H^T during the forward pass. This makes weight distributions more
   uniform, reducing quantization error.

5. **Variable bit-width per layer** — Attention layers (q/k/v/o-proj) are more
   sensitive than MLP layers (gate/up/down). Spend 3-4 bits on attention and
   2 bits on MLP. The average bit-width determines total size.

6. **Entropy coding of quantized codes** — After quantization, apply Huffman
   coding to the per-weight codes. Non-uniform code distributions (some levels
   are more common) yield additional compression at zero quality cost.

---

### Performance rules

The quantization loop iterates over ~187 `nn.Linear` layers in a Python `for` loop.
Keep these rules in mind:

- **Use vectorised PyTorch operations** — `reshape`, broadcasting, `torch.clamp`,
  `torch.round`. One GPU kernel handles millions of weights.
- **NEVER iterate over individual weights, rows, or columns in Python.**
- **Avoid `.item()` calls inside loops over layers.** It synchronises CUDA.
- **Avoid trial-by-error grid searches** that multiply per-layer work.
- **Avoid operations that allocate full-sized tensor copies.** Prefer views,
  in-place ops, and broadcasting over allocation.
- **`eval_perplexity.py` automatically prints `Tokens/sec`.**

---

## Integrity Rules — what the agent MUST NOT do

These rules exist to prevent the agent from gaming the benchmark. Violating any
of them invalidates the experiment.

### Evaluation integrity
- **Do not modify `eval_perplexity.py` or `eval_topk.py`.** No exceptions.
- **Do not run eval with different parameters between experiments.** `--context-length`,
  `--stride`, `--max-tokens`, `--reference-cache`, `--split`, and `--device` must be
  identical for every run in the same experiment branch.
- **Do not change the reference model.** The `--reference` argument must always point
  to `Qwen/Qwen3.5-0.8B` (FP16).
- **Do not corrupt or replace the reference cache.** Once created, the `.mmap` cache
  must not be modified, truncated, or regenerated with different parameters.
- **Do not fabricate eval results.** Every row in `results.tsv` must come from actual
  eval runs whose output was captured verbatim.

### Quantization integrity
- **Do not skip quantization.** Every experiment must run `quantize.py` with `--save`
  pointing to a _new_ `quantized_models/<tag>` directory, then run eval against that
  freshly saved model.
- **Do not re-use old quantized models.** `rm -rf quantized_models/<tag>` after every
  eval. The next experiment must produce a new quantization from scratch.
- **Do not partially quantize.** All `nn.Linear` layers must be quantized (no skipping
  layers to cheat on KL or top-P).
- **Do not change `--dtype bfloat16`.** This must always be passed.
- **Do not exceed Q4_K's compressed size.**
- **Do not modify `quantizer.py`.**
- **Do not increase VRAM beyond 8 GB.**
- **Do not consume the test set during quantization.** Calibration data and evaluation
  data must be disjoint. `eval_perplexity.py` uses Wikitext-2 test split.
- **KL divergence or top-P regression is not acceptable.** Any change that worsens
  either metric is a regression and must be reverted. The quantization must produce
  identical results regardless of device (CPU/GPU) or caching strategy.

### Git and record-keeping integrity
- **Do not edit `results.tsv` directly** except to append a new row after a completed
  experiment. Never change or delete past rows.
- **Do not skip recording bad results.** Every experiment must be recorded.
- **Do not `git commit --amend` or rewrite history** after the fact.
- **Do not delete or force-push branches.**
- **Do not run concurrent experiments.** One experiment at a time on one GPU.
- **Do not skip the cleanup step.** After each eval, delete `quantized_models/<tag>`.
- **Do not change the Python environment** (no `pip install`, no version bumps) between
  experiments in the same branch.

---

## Logging results

Each experiment records into **three** files:

### results.tsv (tab-separated header + rows)

Header:
```
timestamp	exp_id	code_sha	parent_sha	description	status	kl_divergence	top_p_agreement	bits	groupsize	symmetric	format	context_length	max_tokens	size_mb	peak_vram_mb	tokens_per_sec
```

Fields:
| field | source |
|---|---|
| `timestamp` | ISO-8601 time of eval completion |
| `exp_id` | Unique experiment ID (e.g. `exp-20260710-001`) |
| `code_sha` | `git rev-parse HEAD` **of the code commit** (not the result commit) |
| `parent_sha` | `git rev-parse HEAD~1` from before the code change |
| `description` | One-line idea description (no tabs, no commas) |
| `status` | `success` or the error type |
| `kl_divergence` | Mean KL from eval output |
| `top_p_agreement` | Same-top-P percentage from eval_topk.py output |
| `bits` | Average/effective bits per weight |
| `groupsize` | Groupsize used (or equivalent block size) |
| `symmetric` | Always `true` |
| `format` | Quantization scheme tag (e.g. `q4_k`, `q2_kmeans`, `aqlm_2x2`, `hadamard_q3`) |
| `context_length` | Always `1024` |
| `max_tokens` | Always `5000` |
| `size_mb` | Compressed model size in MB |
| `peak_vram_mb` | Peak VRAM during quantization |
| `tokens_per_sec` | Tokens/sec from eval output |

### idea_ledger.md (experiments/)

For each completed experiment, add a structured entry:

```markdown
## <exp_id>

Hypothesis: <one-sentence hypothesis>
Algorithm family: <tag>
Changed code: <function/region>
Representation change: <none | what changed>
Storage risk: <none | size increase estimate>
VRAM risk: <none | what extra is allocated>
Expected win: <lower KL / higher top-P / both>
Outcome: <KL improved/regressed/failed>
```

### failures.tsv (experiments/)

Log crashes, OOMs, NaNs, size-limit failures, and invalid-format attempts here
(**not** in `results.tsv`).

Header:
```
timestamp	exp_id	code_sha	description	status	error_signature	bits	groupsize	size_mb	peak_vram_mb	notes
```

---

## Synthesis checkpoints

**Every 5 completed official experiments**, pause and write a synthesis to
`experiments/synthesis.md`:

```markdown
# Synthesis checkpoint — <date>

- Q4_K baseline: KL=<X>, top-P=<Y>, size=<Z> MB
- Current global best: KL=<X>, top-P=<Y>, size=<Z> MB
- Best code sha: <sha>
- Best effective bits: <N>
- Ideas that improved: <list>
- Ideas that regressed: <list>
- Failure patterns: <list>
- Next three highest-priority hypotheses:
  1. <hypothesis>
  2. <hypothesis>
  3. <hypothesis>
- Current phase: <phase number and name>
```

---

## Concurrent agents

When multiple agents work in parallel (each in their own workspace clone),
they coordinate through a shared Git remote. The rules below ensure zero
conflict and zero wasted work.

### Branch discipline

**Each agent works on its own experiment branch.** Never modify the main
branch directly. Create a branch like `exp/20260710-variable-bit` and do
all work there.

### Claim protocol

1. Generate a unique `EXP_ID`:
   `exp-$(date +%Y%m%d)-$(printf '%04x' $(( RANDOM * 65536 + RANDOM )) | head -c 4)`
2. Read `experiments/in_progress.md` — if any claim is live, choose a different
   idea or wait 10s and pull again
3. Write one line: `EXP_ID: <one-line idea description>`
4. `git add experiments/in_progress.md && git commit -m "claim: EXP_ID"
   && git pull origin <main> && git push -u origin HEAD`
5. If push fails (race), return to step 1
6. **Do NOT start work** until `git pull origin <main>` confirms your claim
   is visible on the remote

### Experiment protocol (no conflicts guaranteed)

1. **Commit code**: `git add quantize.py && git commit -m "exp: <description>"`
2. **Push code**: `git push`
3. **Quantize + eval** (safe — no one else touches this branch)
4. **Append results.tsv and idea_ledger.md**
5. **Commit results**: `git add results.tsv experiments/idea_ledger.md runs/
   && git commit -m "record: <description> (KL=<value>, top-P=<value>)"`
6. **Push results**: `git push`
7. **Clear claim**: remove your line from in_progress.md, commit, push
8. **Decide keep/revert** on YOUR branch — no impact on main

### Promoting to main

Once your experiment finishes and you have the recorded result:

- **If this is the new global best** (smaller + same/better KLD + same/better top-P
  compared to the current best under the Q4_K size cap):
  1. `git checkout <main> && git pull origin <main>`
  2. `git merge --no-ff exp/YYYYMMDD-<keyword> -m "merge: <description>"`
  3. `git push origin <main>`
- **If NOT the new global best**: do nothing — your branch records the
  experiment history. Main stays unchanged.

---

## Operational notes

- **Never use `pkill`.** It hangs the session. Use `kill <PID>` instead.
- **Peak VRAM** can be read from `torch.cuda.max_memory_allocated()` after
  quantization completes.

---

## The experiment loop

The active `quantize.py` at the branch tip should **always** be the global best
implementation. The search must not drift into worse code.

### Pre-loop checks (do these ONCE at the start)

1. **Verify the branch**: `git branch --show-current`.
2. **Read Q4_K baseline**: See first row of `results.tsv` for size, KLD, and top-P targets.
3. **Confirm reference cache exists**: `ls cache/ref_logits_0.8B.mmap` — must be present.
   Do NOT delete or regenerate it.
4. **Confirm environment**: `HF_HUB_OFFLINE=1` is set in the shell for all commands.

### Loop iteration

1. **Read state**:
   ```
   git log --oneline -3
   tail -5 results.tsv
   cat experiments/idea_ledger.md | head -80
   cat experiments/failures.tsv  # avoid repeating crashes
   ```

2. **Propose one experiment**:
   - Hypothesis (one sentence).
   - Algorithm family tag.
   - Expected storage impact (must end up < Q4_K size).
   - Expected VRAM impact.
   - Why it is novel versus prior runs (check idea_ledger.md).

3. **Modify only `quantize.py`**.

4. **Run smoke checks** (optional, recommended):
   ```
   python -m py_compile quantize.py
   ```

5. **Commit the code BEFORE eval**:
   ```bash
   git add quantize.py
   git commit -m "exp: <description>"
   CODE_SHA=$(git rev-parse HEAD)
   PARENT_SHA=$(git rev-parse HEAD~1)
   EXP_ID="exp-$(date +%Y%m%d)-$(printf '%03d' $(wc -l < results.tsv))"
   mkdir -p runs/$EXP_ID
   ```

6. **Run quantization and evals**:
   ```bash
   # Quantize
   HF_HUB_OFFLINE=1 .venv/bin/python quantize.py \
       --model Qwen/Qwen3.5-0.8B --bits <N> \
       --dtype bfloat16 --groupsize <N> \
       --save quantized_models/<tag> 2>&1 | tee runs/$EXP_ID/quantize.log

   # Record size
   du -sm quantized_models/<tag>/compressed | tee runs/$EXP_ID/size_mb.txt

   # Evaluate KL divergence
   HF_HUB_OFFLINE=1 .venv/bin/python eval_perplexity.py \
       --model quantized_models/<tag> \
       --reference Qwen/Qwen3.5-0.8B \
       --context-length 1024 --max-tokens 5000 \
       --reference-cache cache/ref_logits_0.8B.mmap 2>&1 | tee runs/$EXP_ID/eval.log

   # Evaluate same-top-P
   HF_HUB_OFFLINE=1 .venv/bin/python eval_topk.py \
       --model quantized_models/<tag> \
       --reference Qwen/Qwen3.5-0.8B \
       --reference-cache cache/ref_logits_0.8B.mmap \
       --context-length 1024 --max-tokens 5000 2>&1 | tee runs/$EXP_ID/topk.log

   # Cleanup
   rm -rf quantized_models/<tag>
   ```
   Extract `SIZE_MB`, `KL`, `TOP_P`, and `TOK_PER_SEC` from the log files.

7. **On crash / OOM / NaN / size-limit failure**:
   - Append one row to `experiments/failures.tsv`.
   - Commit the failure log:
     ```bash
     git add experiments/failures.tsv runs/$EXP_ID/
     git commit -m "fail: <description>"
     ```
   - Revert the code commit:
     ```bash
     git revert --no-edit $CODE_SHA
     ```
   - Continue to next iteration.

8. **On valid eval** — append one TSV row to `results.tsv` using actual values:
   ```
   <timestamp>	<EXP_ID>	<CODE_SHA>	<PARENT_SHA>	<description>	success	<KL>	<TOP_P>	<bits>	<groupsize>	true	<format>	1024	5000	<SIZE_MB>	<VRAM_MB>	<TOK_PER_SEC>
   ```
   Tab-separated, no commas in description.

   Update `experiments/idea_ledger.md` with the structured experiment entry.

9. **Record the result permanently**:
   ```bash
   git add results.tsv experiments/idea_ledger.md runs/$EXP_ID/
   git commit -m "record: <description> (KL=<value>, top-P=<value>)"
   ```

10. **Decide whether to keep the code**:
    - **If this result beats the current best across all three metrics**
      (smaller, same/better KLD, same/better top-P): keep the code.
    - **If NOT a new global best**: `git revert --no-edit $CODE_SHA`.
      The result remains recorded forever. The branch tip returns to the
      previous-best code.

    Exception: during **Phase 0 baseline establishment**, only the Q4_K
    baseline is recorded. No algorithm changes are made.

11. **Every 5 valid official runs**, write a synthesis checkpoint to
    `experiments/synthesis.md` and commit it.

12. Go to step 1.
