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
* `--bits 2` — quantize to 2 bits
* `--groupsize 128` — quantize weights in a groups of 128
* `--symmetric` — use symmetric quantization (zero-point = 0)
* `--dtype bfloat16` — load the base model in BF16 precision

**What you CAN do:**
- Modify `quantize.py` — this is the only file you edit. 
- You can modify the quantization algorithm (`GPTQLayer`), replace it with a different one, but the one applying quantizer `Quantizer` to the weights of linear layers.
- Apply different ideas from linear algebra and information theory to the quantization algorithm.
- Choose a different calibration dataset.
- Choose a different number of calibration sequences.

**What you CANNOT do:**
- Modify `eval_perplexity.py`. It is read-only. It contains the fixed evaluation.
- Install new packages or add dependencies.
- Modify `Quantizer` and `quantize_tensor` in `quantizer.py`.
- Modify `main` in `quantize.py`.
- Add any modifications that increase the size of the compressed model. For instance, low-rank corrections incur additional memory and compute overhead.

**The goal is simple: get the lowest KL divergence as provided by eval_perplexity.py evaluation script.**

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:
1. Look at the git state: the current branch/commit we're on
2. Tune `quantize.py` with an experimental idea by directly hacking the code.
3. Run an experiment.
4. git commit
5. Record the experiment result in `results.tsv`
6. If KL divergence improved (lower), you "advance" the branch, keeping the git commit
7. If KL divergence is equal or worse, you git reset back to where you started
