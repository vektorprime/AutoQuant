# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jul1`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `eval_perplexity.py` — fixed evaluation, read-only.
   - `quantizer.py` — **read-only** reference.  Understand the packed format,
     dequantization path, supported metadata, and what custom schemes are
     actually representable before inventing a new format.
   - `quantize.py` — the file containing the quantization algorithm (**you edit this**).
   - `data_utils.py` — data preparation utilities (read-only).
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
   The baseline will be recorded after the first run.
5. **Initialize experiments/**: Create `experiments/idea_ledger.md` and
   `experiments/failures.tsv` (header row only).
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Research phases

The search follows a phased plan.  Start at Phase 0 and move forward as each
phase is exhausted or hits diminishing returns.  You may revisit earlier phases
after discoveries in later phases.

### Phase 0 — Baseline sweep and format audit

Run the default implementation at feasible groupsizes:

| groupsize | expected size | notes |
|---|---|---|
| 16 | ~1411 MB | finest granularity |
| 32 | ~941 MB  | default |
| 64 | ~500 MB  | coarser |
| 128| ~328 MB  | coarsest |
| 256| ~200 MB  | may be too coarse |

Stop any groupsize that exceeds the 1575 MB size limit or 8 GB VRAM limit.
This establishes the size/KL frontier.  **No algorithm changes** during Phase 0
— these runs are pure baselines.

Also inspect (reading `quantizer.py` and `eval_perplexity.py`):
- How weights are packed, decompressed, and mapped back to tensors.
- How scales/zeros are stored per group.
- Whether per-layer groupsize is possible.
- Whether non-uniform codebooks can be stored.
- Whether layer-specific metadata is supported by the save/load round-trip.

### Phase 1 — No-format-change improvements

These are the safest first experiments — they change only the scale/computation
without altering the stored representation:

- MSE-optimal scale instead of maxabs scale.
- Activation-weighted MSE scale (see Calibration data rules).
- Clipped-scale variants (closed-form or small grid search, avoiding `.item()` loops).
- Layer-type-specific scale rules (different policy for attention vs MLP).
- Per-layer effective groupsize if the format supports it.
- **Cross-channel grouping** — currently groups partition `in_features`
  independently per output channel.  What if scales are shared across nearby
  output channels (e.g. 2×2 blocks of `(out, in)` groups)?  This reduces
  scale storage overhead without changing group granularity.

### Phase 2 — Activation-aware methods

Try AWQ/GPTQ-inspired approximations within the constraints:

- Stream calibration data (do not store full activations).
- Store only diagonal activation statistics per linear layer (e.g. `E[x²]`).
- Optimize per-group quantization using weighted reconstruction error.
- The weighted objective `loss = Σⱼ E[xⱼ²] · (Wᵢⱼ - Wqᵢⱼ)²` is **far more aligned
  with KL** than plain weight MSE, while fitting the VRAM rule.

### Phase 3 — Representation changes

Only after proving the eval loader supports them:

- Non-uniform 2-bit codebooks (e.g. per-group lookup tables).  For example,
  instead of the hardcoded `{-2s, -s, 0, s}`, store four learned values per group
  such as `{-1.5s, -0.8s, 0.3s, 1.2s}`.  Only feasible if the eval loader's
  dequantization path can consume per-group codebooks.
- Layer-level codebooks (shared across groups).
- Per-group asymmetric offsets or learned levels.
- Outlier-preserving transforms that do not exceed the 1575 MB size limit.
- Error-feedback schemes spanning multiple layers (not just within one layer).

---

## Experimentation

Each experiment runs on a single GPU. The workflow is:

1. Run `quantize.py` to quantize the model and save it to `quantized_models/<tag>`
2. Load the quantized model from `quantized_models/<tag>` and run `eval_perplexity.py` to evaluate KL divergence against the reference model. Save the results to `results.tsv`.
3. Delete the quantized model from `quantized_models/<tag>`.

All experiments MUST be run with the following arguments for quantize.py:
* `--bits 2` — quantize to 2 bits (fixed, never change)
* `--groupsize <N>` — group size (min 16, must divide `in_features`; default 32)
* `--dtype bfloat16` — load the base model in BF16 precision (fixed, never change)

Symmetric quantization is **hardcoded** (always on).  There is no `--symmetric` flag.

### Resource limits (enforced)

* **Compressed model size**: max **1575 MB** (1500 MB + 5% tolerance).
  `quantize.py` will exit with an error if the compressed `.npz` files exceed this.
  Smaller groupsize = more metadata = larger output.  Use `--groupsize` to stay
  under the limit.
* **VRAM**: max **8 GB** during quantization (checked via `nvidia-smi`).  No
  allocating auxiliary tensors that persist across layers.  Views, in-place ops,
  and broadcasting are fine — copies and sorts are not.

**What you CAN do:**
- Modify `quantize.py` — this is the only file you edit.
- Change the quantization algorithm **completely**.  You are not limited to tweaking
  the existing formula — replace the entire `_quantize_one_layer` function, invent
  a new quantization scheme, use iterative optimisation, implement error-diffusion
  across layers, try non-uniform quantization, or anything else that maps float
  weights to 2-bit representations.  The only hard requirements are the bit-width
  and the size/VRAM limits.
- Make multiple passes over the weights (iterative refinement, error feedback) —
  extra compute is fine as long as VRAM stays under 8 GB.
- Add new functions, classes, or imports within `quantize.py` (no external packages).
- Vary `--groupsize` (≥ 16) to trade off between finer quantization and compressed size.
- **Extend `main()`** to add optional quantization-internal arguments or structured
  logging, provided all official runs still pass `--bits 2`, `--dtype bfloat16`, and a
  valid `--groupsize`, and eval integrity is unchanged.

**What you CANNOT do:**
- Modify `eval_perplexity.py`. It is read-only. It contains the fixed evaluation.
- Modify `quantizer.py` (the `Quantizer` class and `quantize_tensor` function).
- Modify `data_utils.py`.
- Install new packages or add dependencies beyond those already in the environment.
- Add modifications that increase the size of the compressed model beyond 1575 MB
  (1500 MB + 5% tolerance).  Low-rank corrections, extra stored tensors, or storing
  weights at > 2 bits per value are all forbidden if they push the final compressed
  `.npz` total above the limit.
- **Increase VRAM usage beyond 8 GB.**  GPU memory consumption must stay at or
  below 8 GB peak.  Extra computation (FLOPs) is acceptable, but VRAM is strictly
  capped.  No caching intermediate activations, no allocating auxiliary tensors
  that persist across layers, no doubling the working set.
- Use more than one GPU. All experiments run on a single GPU.

**The goal is simple: get the lowest KL divergence as provided by eval_perplexity.py evaluation script.**

### Why `quantizer.py` is off-limits

The `Quantizer` class and `quantize_tensor` function are the **default primitives**
that define affine 2-bit symmetric group quantization.  The restriction means you
cannot *edit* these specific primitives — you can, however, write your own
quantization logic from scratch inside `quantize.py`.  A lookup-table quantizer,
a k-means-based centroid approach, or any other 2-bit scheme is fair game as long
as you implement it in `quantize.py` and the weights remain at exactly 2 bits per value.

What the restriction *prevents* is cheating the default primitives — e.g., changing
`maxq` to 15 so the existing `Quantizer` silently does 4-bit work.  You may replace
these primitives entirely with your own, but you may not "adjust" them.

---

### Calibration data rules

Quantization **may** use calibration data from a **non-test** split.
Do **not** use Wikitext-2 test split (`eval_perplexity.py --split test`) or the
eval reference logits (`cache/ref_logits.mmap`) during quantization.

**Allowed calibration statistics:**
- Per-layer input activation second moments `E[xⱼ²]` (one scalar per input channel).
- Per-channel activation scales.
- Small streaming diagonal Hessian approximations.
- Wikitext-2 **train** or **validation** split.

**Forbidden calibration state:**
- Cached full activation tensors across layers (violates VRAM limit and rules).
- Test-set activations.
- Reference logits from `eval_perplexity.py`.

Weighted reconstruction objectives using calibration statistics are strongly
encouraged.  For example:

```
weighted_MSE = Σⱼ E[xⱼ²] · (Wᵢⱼ - Wqᵢⱼ)²
```

This is much more aligned with KL than plain weight MSE and fits the VRAM rule
(storing only one scalar per input channel).

---

### Diagnostics

Diagnostic runs are **allowed** if:
- They are **not** written to `results.tsv`.
- They do **not** use the Wikitext-2 test split.
- They do **not** modify `eval_perplexity.py`.
- They are clearly logged under `runs/<exp_id>/diagnostics/`.

Allowed diagnostics:
- Synthetic tensor roundtrip tests (verify your quantize/dequantize is correct).
- Per-layer quantization error summaries (MSE, max abs error per layer).
- Calibration-split proxy KL or loss (use Wikitext-2 **train** split).
- Layer sensitivity experiments on non-test data.
- Size and VRAM estimates before full runs.

Use diagnostics to reject obviously bad ideas before paying for the full official eval.

---

### Performance guidance

**Ranked idea queue.**  Try these in order — early ideas are higher-probability
improvements:

1. **MSE-optimal per-group scale** — Replace `max(abs)/1.5` with an iterative
   least-squares scale.  For each group:
   ```
   q = clamp(round(w / s), qmin, qmax)
   s = sum(w * q) / sum(q * q)
   repeat 2–3 times
   ```
   Keeps the same 2-bit representation and nearly the same storage.

2. **Activation-weighted MSE scale** — Use calibration activations to estimate
   input-channel importance.  If `hⱼ = E[xⱼ²]`, then optimize:
   ```
   loss = Σⱼ hⱼ · (wⱼ - s·qⱼ)²
   s = Σ(h · w · q) / Σ(h · q · q)
   ```
   Much more useful than raw weight MSE.

3. **Layer-type policies** — Different linear layers should not necessarily use
   the same clipping or scale rule.  Classify layer names:
   - Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - MLP: `gate_proj`, `up_proj`, `down_proj`
   - Output: `lm_head`
   Then test one layer-policy change at a time.

4. **Sensitivity-aware groupsize** — If the format supports it, spend metadata
   budget where it matters: sensitive layers get smaller groupsize, less sensitive
   layers get larger groupsize.

5. **Error feedback within a layer** — Quantize groups sequentially and carry a
   bounded residual into the next group or block.  Useful at 2 bits, but should
   come after scale optimization and activation weighting.

---

### Performance rules

The quantization loop iterates over ~187 `nn.Linear` layers in a Python `for` loop.
Keep these rules in mind:

* **Use vectorised PyTorch operations** — `reshape`, broadcasting, `torch.clamp`,
  `torch.round`.  One GPU kernel handles millions of weights.
* **NEVER iterate over individual weights, rows, or columns in Python.**  A Python
  for-loop over 2048 columns × 187 layers = 383K iterations, each launching a
  tiny GPU kernel.  This is 100× slower than a single vectorised call.
* **Avoid `.item()` calls inside loops over layers.**  `.item()` synchronises the
  CUDA stream (blocks CPU until GPU finishes).  If you call it inside the layer
  loop, you force a GPU→CPU round-trip for every single layer, serialising work
  that could otherwise overlap.
* **Avoid trial-by-error grid searches** (e.g., trying 9 scale factors for each
  layer).  This multiplies per-layer work and adds 1683 extra CUDA syncs.
  Design a closed-form solution instead.
* **Avoid operations that allocate full-sized tensor copies.**  `torch.sort`
  returns a sorted copy plus indices — it doubles the memory footprint of the
  tensor being sorted.  Similarly, `torch.clone()`, `repeat_interleave` on large
  tensors, and any `.abs()` on a non-view all silently blow up VRAM.  Prefer
  views, in-place ops (`torch.abs` not `.abs()`, `amin`/`amax` which don't copy),
  and broadcasting over allocation.
* **`eval_perplexity.py` automatically prints `Tokens/sec`** in its output.
  You do not need to compute it manually — just read it from the eval result.

---

## Integrity Rules — what the agent MUST NOT do

These rules exist to prevent the agent from gaming the benchmark.  Violating any of
them invalidates the experiment.

### Evaluation integrity
- **Do not modify `eval_perplexity.py`.**  No exceptions.
- **Do not run eval with different parameters between experiments.**  `--context-length`,
  `--stride`, `--max-tokens`, `--reference-cache`, `--split`, and `--device` must be
  identical for every run in the same experiment branch.
- **Do not change the reference model.**  The `--reference` argument must always point
  to the same base model (e.g. `Qwen/Qwen3.5-0.8B`).  Running against a degraded or
  different reference makes results incomparable.
- **Do not corrupt or replace the reference cache.**  Once created, the `.mmap` cache
  must not be modified, truncated, or regenerated with different parameters.
- **Do not fabricate eval results.**  Every row in `results.tsv` must come from an
  actual `eval_perplexity.py` run whose output was captured verbatim.

### Quantization integrity
- **Do not skip quantization.**  Every experiment must run `quantize.py` with `--save`
  pointing to a _new_ `quantized_models/<tag>` directory, then run eval against that
  freshly saved model.
- **Do not re-use old quantized models.**  `rm -rf quantized_models/<tag>` after every
  eval.  The next experiment must produce a new quantization from scratch.
- **Do not partially quantize.**  All `nn.Linear` layers must be quantized (no skipping
  layers to cheat on KL).
- **Do not change the fixed CLI arguments.**  `--bits 2` and `--dtype bfloat16`
  must always be passed.  Groupsize may vary (via `--groupsize`, ≥ 16).
  Symmetric is hardcoded — there is no flag for it and it must not be added.
- **Do not exceed the compressed size limit of 1575 MB.**  `quantize.py` enforces
  this — if your experiment hits the limit, increase `--groupsize` to reduce
  scale/zero overhead.
- **Do not modify `quantizer.py`.**  The `Quantizer` class and `quantize_tensor`
  function are off-limits.
- **Do not increase VRAM.**  GPU memory usage must not exceed 8 GB.  Extra compute
  is fine; extra memory allocations that persist across the quantization loop
  are forbidden.
- **Do not consume the test set during quantization.**  Calibration data and evaluation
  data must be disjoint.  `eval_perplexity.py` uses Wikitext-2 test split — do not use
  that split for calibration.  The Wikitext-2 **train** split is available.
- **KL divergence regression is not acceptable by any means.**  Any change that
  increases KL divergence (even by small numerical margins) is a regression.
  The quantization must produce *identical* results regardless of the device
  (CPU/GPU) or caching strategy used.  If a change causes KL to increase, it
  must be fully reverted — no partial regressions are tolerated.

### Git and record-keeping integrity
- **Do not edit `results.tsv` directly** except to append a new row after a completed
  experiment.  Never change or delete past rows.
- **Do not skip recording bad results.**  Every experiment must be recorded, even
  (especially) the ones that regress.  Selective recording is cheating.
- **Do not `git commit --amend` or rewrite history** after the fact.
- **Do not delete or force-push branches.**  The git history is the experiment log.
- **Do not run concurrent experiments.**  One experiment at a time on one GPU.
- **Do not skip the cleanup step.**  After each eval, delete `quantized_models/<tag>`.
- **Do not change the Python environment** (no `pip install`, no version bumps) between
  experiments in the same branch.

---

## Logging results

Each experiment records into **three** files:

### results.tsv (tab-separated header + rows)

Header:
```
timestamp	exp_id	code_sha	parent_sha	description	status	kl_divergence	bits	groupsize	symmetric	format	context_length	max_tokens	size_mb	peak_vram_mb	tokens_per_sec
```

Fields:
| field | source |
|---|---|
| `timestamp` | ISO-8601 time of eval completion |
| `exp_id` | Unique experiment ID (e.g. `exp-20260701-001`) |
| `code_sha` | `git rev-parse HEAD` **of the code commit** (not the result commit) |
| `parent_sha` | `git rev-parse HEAD~1` from before the code change |
| `description` | One-line idea description (no tabs, no commas) |
| `status` | `success` or the error type |
| `kl_divergence` | Mean KL from eval output |
| `bits` | Always `2` |
| `groupsize` | Groupsize used |
| `symmetric` | Always `true` |
| `format` | Quantization scheme tag (e.g. `q2_k`, `err_diff`, `mse_opt`) |
| `context_length` | Always `1024` |
| `max_tokens` | Always `5000` |
| `size_mb` | `du -sm quantized_models/<tag>/compressed` |
| `peak_vram_mb` | Peak VRAM from `nvidia-smi` or `torch.cuda.max_memory_allocated()` |
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
Expected win: <lower KL / faster / same>
Outcome: <KL improved/regressed/failed>
```

### failures.tsv (experiments/)

Log crashes, OOMs, NaNs, size-limit failures, and invalid-format attempts here
(**not** in `results.tsv`).

Header:
```
timestamp	exp_id	code_sha	description	status	error_signature	groupsize	size_mb	peak_vram_mb	notes
```

---

## Synthesis checkpoints

**Every 5 completed official experiments**, pause and write a synthesis to
`experiments/synthesis.md`:

```markdown
# Synthesis checkpoint — <date>

- Current global best KL: <value>
- Best code sha: <sha>
- Best groupsize: <N>
- Ideas that improved: <list>
- Ideas that regressed: <list>
- Failure patterns: <list>
- Next three highest-priority hypotheses:
  1. <hypothesis>
  2. <hypothesis>
  3. <hypothesis>
- Current phase: <phase number and name>
```

This prevents random-walk drift and ensures the search stays goal-oriented.

---

## Concurrent agents

When multiple agents work in parallel (each in their own workspace clone),
they coordinate through a shared Git remote.  The rules below ensure zero
conflict and zero wasted work.

### Branch discipline

**Each agent works on its own experiment branch.**  Never modify `master`
directly.  Create a branch like `exp/20260702-layertype-diffusion` and do
all work there.

```bash
git checkout master && git pull origin master
git checkout -b exp/YYYYMMDD-<keyword>
```

### Claim protocol

1. Generate a unique `EXP_ID`: `exp-$(date +%Y%m%d)-$(printf '%03d' $(wc -l < results.tsv))`
2. Read `experiments/in_progress.md` — if any claim is live, choose a different
   idea or wait 10s and pull again
3. Write one line: `EXP_ID: <one-line idea description>`
4. `git add experiments/in_progress.md && git commit -m "claim: EXP_ID"
   && git pull origin master && git push -u origin HEAD`
5. If push fails (race), return to step 1

### Experiment protocol (no conflicts guaranteed)

1. **Commit code**: `git add quantize.py && git commit -m "exp: <description>"`
2. **Push code**: `git push`
3. **Quantize + eval** (safe — no one else touches this branch)
4. **Append results.tsv and idea_ledger.md**
5. **Commit results**: `git add results.tsv experiments/idea_ledger.md runs/
   && git commit -m "record: <description> (KL=<value>)"`
6. **Push results**: `git push`
7. **Clear claim**: remove your line from in_progress.md, commit, push
8. **Decide keep/revert** on YOUR branch — no impact on master
9. **If global best**: open a PR or merge to master.  If regressed: no action
   needed beyond recording.

### No-busy-wait variant

If you don't want to wait for another agent to finish:

- Skip the claim file entirely
- Create your branch, run the experiment, merge results into master with
  `git pull origin master && git merge --no-ff` only AFTER the other agent's
  experiment has merged to master
- Append to results.tsv on your branch, then merge to master

The claim-file approach is preferred — it avoids duplicate experiments.

---

## Operational notes

- **Never use `pkill`.**  It hangs the session.  If a process needs to be killed,
  use `kill <PID>` by finding the PID with `ps aux | grep <process>`.  Better
  yet, avoid killing processes — just delete the output directory
  (`rm -rf quantized_models/<tag>`) and the next run will overwrite cleanly.

- **Peak VRAM** can be read from the last line of quantize.py output
  (`Peak VRAM: <N> MB`), or via `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
  after the quantization completes.

---

## The experiment loop

The active `quantize.py` at the branch tip should **always** be the global best
KL implementation.  The search must not drift into worse code because of
groupsize-specific baselines.

### Pre-loop checks (do these ONCE at the start)

1. **Verify the branch**: `git branch --show-current` — must match `autoresearch/<tag>`.
2. **Read current global best**: `cut -f6 results.tsv | sort -n | head -1`
   (KL is column 6).  This is your target to beat.
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
   - Algorithm family tag (e.g. `scale_optimization`, `error_compensation`,
     `activation_weighted`, `codebook`, `layer_policy`).
   - Expected storage impact (none / +X MB).
   - Expected VRAM impact (none / +X MB for Y tensors).
   - Why it is novel versus prior runs (check idea_ledger.md).

3. **Modify only `quantize.py`**.

4. **Run smoke checks** (optional, recommended):
   ```
   python -m py_compile quantize.py
   ```
   If your change introduces new metadata or format, verify the save/load round-trip
   works with a minimal test (logged under `runs/<exp_id>/diagnostics/`).

5. **Commit the code BEFORE eval** (so `code_sha` points to the actual tested code):
   ```bash
   git add quantize.py
   git commit -m "exp: <description>"
   CODE_SHA=$(git rev-parse HEAD)
   PARENT_SHA=$(git rev-parse HEAD~1)
   EXP_ID="exp-$(date +%Y%m%d)-$(printf '%03d' $(wc -l < results.tsv))"
   mkdir -p runs/$EXP_ID
   ```

6. **Run quantization and eval** (from repo root):
   ```bash
   # Quantize
   HF_HUB_OFFLINE=1 .venv/bin/python quantize.py \
       --model Qwen/Qwen3.5-0.8B --bits 2 \
       --dtype bfloat16 --groupsize <N> \
       --save quantized_models/<tag> 2>&1 | tee runs/$EXP_ID/quantize.log

   # Record size BEFORE deleting
   du -sm quantized_models/<tag>/compressed | tee runs/$EXP_ID/size_mb.txt

   # Evaluate
   HF_HUB_OFFLINE=1 .venv/bin/python eval_perplexity.py \
       --model quantized_models/<tag> \
       --reference Qwen/Qwen3.5-0.8B \
       --context-length 1024 --max-tokens 5000 \
       --reference-cache cache/ref_logits_0.8B.mmap 2>&1 | tee runs/$EXP_ID/eval.log

   # Cleanup
   rm -rf quantized_models/<tag>
   ```
   Extract `SIZE_MB`, `KL`, and `TOK_PER_SEC` from the log files.

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
   <timestamp>	<EXP_ID>	<CODE_SHA>	<PARENT_SHA>	<description>	success	<KL>	2	<groupsize>	true	<format>	1024	5000	<SIZE_MB>	<VRAM_MB>	<TOK_PER_SEC>
   ```
   Tab-separated, no commas in description.

   Update `experiments/idea_ledger.md` with the structured experiment entry.

9. **Record the result permanently**:
   ```bash
   git add results.tsv experiments/idea_ledger.md runs/$EXP_ID/
   git commit -m "record: <description> (KL=<value>)"
   ```

10. **Decide whether to keep the code**:
    - **If this KL is the new global best** (lower than every other row in results.tsv):
      keep the code — the branch tip is now the best implementation.
    - **If KL is NOT a new global best**:
      ```bash
      git revert --no-edit $CODE_SHA
      ```
      The result remains recorded forever.  The branch tip returns to the
      previous-best code.  Do NOT reset — `git revert` preserves history.

    Exception: during **Phase 0 baseline sweeps**, no algorithm changes are made
    and results are only used to choose the default groupsize.  Code does not change.

11. **Every 5 valid official runs**, write a synthesis checkpoint to
    `experiments/synthesis.md` and commit it.

12. Go to step 1.
