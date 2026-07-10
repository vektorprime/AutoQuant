# Validation results

The available sandbox did not contain the Transformers package, the source model checkpoint, or a CUDA GPU, so a full Qwen KL/top-token benchmark could not be rerun here. The codec and loader paths were tested with PyTorch synthetic models by stubbing the unavailable Transformers entry points.

Completed checks:

- All Python files pass `py_compile`.
- Canonical decoder versus tiled `QuantizedLinear` forward: maximum absolute error below `7.2e-7` in the tested FP32 cases.
- End-to-end synthetic packed checkpoint load, including residual tensors, meta construction, and a biased quantized layer: maximum logit error `5.36e-7` versus a manually dequantized reference model.
- The encoder does not mutate source model weights.
- Bias metadata and residual bias loading were verified.
- On a centered synthetic weight matrix, restoring three least-squares iterations reduced normalized weight MSE from `0.0061668` to `0.0058972`, about a 4.4% relative reduction.
- On a shifted synthetic weight matrix, refinement reduced normalized weight MSE from `0.0049622` to `0.0047501`, about a 4.3% relative reduction.
- In those synthetic tests, changing refined `d`/`dmin` storage from FP16 to FP32 changed normalized MSE by only about `1e-7` to `3e-7`. This does not prove FP16 scales are harmless for every real layer, but it shows that scale conversion alone is unlikely to explain the full observed KL/top-token regression.

A new model-level benchmark must be run after re-quantizing with the patched encoder. Existing packed files do not contain the restored least-squares solution and may contain a residual embedding modified by the previous in-place quantizer.
