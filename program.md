# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `eval_perplexity.py` — fixed evaluation, that you cannot modify.
   - `quantize.py` — the file containing the quantization algorithm.
   - `data_utils.py` — data preparation utilities.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The workflow is:

1. Run `quantize.py` to quantize the model and save it to `quantized_models/<tag>`
2. Load the quantized model from `quantized_models/<tag>` and run `eval_perplexity.py` to evaluate KL divergence against the reference model. Save the results to `results.tsv`.
3. Delete the quantized model from `quantized_models/<tag>`.

All experiments MUST be run with the following arguments for quantize.py:
* `--bits 2` — quantize to 2 bits (fixed, never change)
* `--dtype bfloat16` — load the base model in BF16 precision (fixed, never change)

Groupsize is **hardcoded to 32** in `quantize.py` — there is no `--groupsize` CLI
flag and it must not be added or changed.  Symmetric quantization is also hardcoded
(always on).

**What you CAN do:**
- Modify `quantize.py` — this is the only file you edit.
- Modify the quantization algorithm: change how weights are quantized (rounding strategy,
  group partitioning, scale computation, error compensation, etc.) as long as it produces
  a valid quantized model via `Quantizer` or equivalent logic applied to `nn.Linear` weights.
- Add new functions, classes, or imports within `quantize.py` (no external packages).
- Choose calibration data or design the quantization to not require it.

**What you CANNOT do:**
- Modify `eval_perplexity.py`. It is read-only. It contains the fixed evaluation.
- Modify `quantizer.py` (the `Quantizer` class and `quantize_tensor` function).
- Modify `main()` in `quantize.py` — the entry point structure is fixed.
- Modify `data_utils.py`.
- Install new packages or add dependencies beyond those already in the environment.
- Add modifications that increase the size of the compressed model (e.g., low-rank
  corrections, extra stored tensors, or storing weights at > 2 bits per value).
- **Increase VRAM usage.**  GPU memory consumption must stay at or below the current
  baseline.  Extra computation (FLOPs) is acceptable, but VRAM is strictly capped.
  No caching intermediate activations, no allocating auxiliary tensors that persist
  across layers, no doubling the working set.  If it makes `nvidia-smi` climb, it's
  forbidden.
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

### Performance guidance

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

## Integrity Rules — what the agent MUST NOT do

These rules exist to prevent the agent from gaming the benchmark.  Violating any of
them invalidates the experiment.

### Evaluation integrity
- **Do not modify `eval_perplexity.py`.**  No exceptions.
- **Do not run eval with different parameters between experiments.**  `--context-length`,
  `--stride`, `--max-tokens`, `--reference-cache`, `--split`, and `--device` must be
  identical for every run in the same experiment branch.
- **Do not change the reference model.**  The `--reference` argument must always point
  to the same base model (e.g. `Qwen/Qwen3.5-2B`).  Running against a degraded or
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
  must always be passed.  Groupsize and symmetric are hardcoded — there are no
  flags for them and they must not be added.
- **Do not modify `quantizer.py`.**  The `Quantizer` class and `quantize_tensor`
  function are off-limits.
- **Do not increase VRAM.**  GPU memory usage must not exceed the current baseline.
  Extra compute is fine; extra memory allocations that persist across the
  quantization loop are forbidden.

### Git and record-keeping integrity
- **Do not edit `results.tsv` directly** except to append a new row after a completed
  experiment.  Never change or delete past rows.
- **Do not skip recording bad results.**  Every experiment must be recorded, even
  (especially) the ones that regress.  Selective recording is cheating.
- **Do not `git commit --amend` or rewrite history** after the fact.
- **Do not `git reset` to a commit that was not an experiment boundary.**  The loop is:
  commit → eval → keep-or-reset.  You may only reset to the commit you started from.
- **Do not delete or force-push branches.**  The git history is the experiment log.

### Process integrity
- **Do not run concurrent experiments.**  One experiment at a time on one GPU.
- **Do not skip the cleanup step.**  After each eval, delete `quantized_models/<tag>`.
- **Do not change the Python environment** (no `pip install`, no version bumps) between
  experiments in the same branch.
- **Do not consume the test set during quantization.**  Calibration data and evaluation
  data must be disjoint.  `eval_perplexity.py` uses Wikitext-2 test split — do not use
  that split for calibration.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

## Operational notes

- **Never use `pkill`.**  It hangs the session.  If a process needs to be killed,
  use `kill <PID>` by finding the PID with `ps aux | grep <process>`.  Better
  yet, avoid killing processes — just delete the output directory
  (`rm -rf quantized_models/<tag>`) and the next run will overwrite cleanly.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

### Pre-loop checks (do these ONCE at the start)

1. **Verify the branch**: `git branch --show-current` — must match `autoresearch/<tag>`.
2. **Read current best KL**: `tail -1 results.tsv` and note the `kl_divergence` column.
   This is your target to beat.
3. **Confirm reference cache exists**: `ls cache/ref_logits.mmap` — must be present.
   Do NOT delete or regenerate it.
4. **Confirm environment**: `HF_HUB_OFFLINE=1` is set in the shell for all commands.

### Loop iteration — repeat forever

1. **Read state**: `git log --oneline -1` and `tail -1 results.tsv` to know where you stand.
2. **Read all past descriptions** in `results.tsv` (`cut -f3 results.tsv`).  Use these
   to avoid repeating ideas.  Your next idea must be novel — check that no prior row
   describes the same approach.  If an idea was already tried (even with a worse KL),
   skip it and think of something else.
3. **Come up with ONE experimental idea** to improve the quantization algorithm.
   This is entirely your choice — do not wait for human suggestions.  The idea should
   modify how weights are quantized in `_quantize_one_layer()` or `quantize_model()`.
4. **Modify `quantize.py`** with that single change.  Avoid shotgun diffs — change
   one thing so the effect is clearly attributable.
5. **Run the experiment** (all commands from the repo root):
    ```
    HF_HUB_OFFLINE=1 .venv/bin/python quantize.py \
        --model Qwen/Qwen3.5-2B --bits 2 \
        --dtype bfloat16 --save quantized_models/<tag>
    HF_HUB_OFFLINE=1 .venv/bin/python eval_perplexity.py \
        --model quantized_models/<tag> \
        --reference Qwen/Qwen3.5-2B \
        --context-length 1024 --max-tokens 5000 \
        --reference-cache cache/ref_logits.mmap
    rm -rf quantized_models/<tag>
    ```
    All args are fixed.  Groupsize (32) and symmetric are hardcoded.
6. **On failure** (non-zero exit, OOM, crash, NaN KL): `git checkout -- quantize.py`
   to revert.  Do NOT record the result.  Go back to step 2.
7. **On success**: append one TSV row to `results.tsv`:
   ```
    <ISO-timestamp>\t<git rev-parse HEAD>\t<description of the idea>\t<KL value>\t2\t32\ttrue\tq2_k\t0\t0\t1024\t5000\t<size_mb>\t<tokens_per_sec>
   ```
   Use actual values from the eval output and `ls -l` on the compressed directory.
   Tab-separated, no commas in the description.
8. **Git decision**:
    - `git add quantize.py results.tsv`
    - `git commit -m "<description> (KL=<value>)"`
    - Find the **best KL for the current groupsize** in results.tsv (lowest value
      in the `kl_divergence` column where `groupsize` matches the hardcoded 32).
    - **If no matching baseline exists**: this run IS the baseline.  Keep the commit
      and continue — you now have a target to beat.
    - **Lower KL than the previous best**: advance — keep the commit.  This is now
      the new best.
    - **Equal or higher KL**: `git reset --hard HEAD~1` — revert quantize.py to
      the previous best.  Never reset further back than one commit.
9. Go to step 2.
