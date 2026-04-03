#!/usr/bin/env python3
"""
GPTQ post-training quantization for AutoModelForCausalLM.

Algorithm reference:
  Frantar et al., "GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers", ICLR 2023.
  https://arxiv.org/abs/2210.17323
"""

import argparse
import logging
import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import get_calibration_data
from quantizer import Quantizer, quantize_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer GPTQ
# ---------------------------------------------------------------------------

class GPTQLayer:
    """
    Runs the GPTQ algorithm on a single nn.Linear layer.

    Usage:
        gptq = GPTQLayer(linear_layer)
        for x in calibration_inputs:
            gptq.add_batch(x)              # accumulate Hessian
        Q, scale, zero = gptq.quantize(quantizer)
        gptq.free()
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        W = layer.weight.data.clone().float()
        self.rows = W.shape[0]   # out_features
        self.cols = W.shape[1]   # in_features
        self.H = torch.zeros((self.cols, self.cols), device=W.device)
        self.n_samples = 0

    # ------------------------------------------------------------------
    def add_batch(self, inp: torch.Tensor) -> None:
        """
        Accumulate Hessian H = 2/n * X X^T from a batch of inputs.

        inp: (..., in_features)  — raw input to the Linear layer.
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])     # (tokens, in_features)
        inp = inp.t().float()                        # (in_features, tokens)

        n = inp.shape[1]
        self.H = self.H * (self.n_samples / (self.n_samples + n))
        self.n_samples += n
        inp = math.sqrt(2 / self.n_samples) * inp
        self.H += inp.matmul(inp.t())

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def quantize(
        self,
        quantizer: Quantizer,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run GPTQ and return (Q, scale, zero).

        Q     : quantised–dequantised weight  (out, in)  float32
        scale : (out, n_groups)  float32
        zero  : (out, n_groups)  float32
        """
        W = self.layer.weight.data.clone().float()      # (out, in)
        H = self.H.clone()

        # ---- Hessian regularisation -----------------------------------
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        damp = percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(self.cols, device=W.device)
        H[idx, idx] += damp

        # ---- Cholesky of H^{-1} (upper triangular) --------------------
        # More numerically stable than direct inversion.
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except torch.linalg.LinAlgError:
            logger.warning(
                "Cholesky failed — adding extra damping and retrying."
            )
            H = self.H.clone()
            H[idx, idx] += 10 * damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)

        Hinv = H   # upper-triangular Cholesky factor of H^{-1}

        # ---- Compute per-group quant params ---------------------------
        quantizer.find_params(W)

        # ---- Block-wise GPTQ update -----------------------------------
        Q = torch.zeros_like(W)

        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            bs = i2 - i1

            W_blk   = W[:, i1:i2].clone()          # (out, bs)
            Q_blk   = torch.zeros_like(W_blk)
            Err_blk = torch.zeros_like(W_blk)
            Hinv_blk = Hinv[i1:i2, i1:i2]          # (bs, bs)

            g = self.cols if quantizer.groupsize == -1 else quantizer.groupsize
            # Corrected zero for symmetric quantization: Quantizer sets zero=0 which
            # clips negative weights to 0. Use zero=maxq//2 to center the range at 0,
            # mapping [-xmax, +xmax] to [0, 2*(maxq//2)] with 0 → maxq//2.
            sym_zero = quantizer.maxq // 2  # = 7 for 4-bit

            for i in range(bs):
                col = i1 + i
                w = W_blk[:, i]                     # (out,)
                d = Hinv_blk[i, i]

                # Quantize this column
                if quantizer.groupsize != -1 and col % quantizer.groupsize == 0:
                    grp_start = col
                    grp_end   = min(col + quantizer.groupsize, self.cols)
                    quantizer.find_params(W[:, grp_start:grp_end])
                    _s = quantizer.scale.squeeze(1)  # (out,)
                else:
                    group_idx = col // g
                    _s = quantizer.scale[:, min(group_idx, quantizer.scale.shape[1] - 1)]

                # Use corrected zero (sym_zero) so negative weights are preserved:
                # q = clamp(round(w/s) + sym_zero, 0, maxq), dequant = s*(q - sym_zero)
                _z = _s.new_full(_s.shape, sym_zero)
                q = quantize_tensor(w.unsqueeze(1), _s.unsqueeze(1), _z.unsqueeze(1), quantizer.maxq).squeeze(1)
                Q_blk[:, i] = q

                # Propagate error to remaining columns in the block
                err = (w - q) / d                   # (out,)
                Err_blk[:, i] = err
                W_blk[:, i:] -= err.unsqueeze(1) * Hinv_blk[i, i:].unsqueeze(0)

            Q[:, i1:i2] = Q_blk

            # Propagate block error to all remaining columns
            if i2 < self.cols:
                W[:, i2:] -= Err_blk.matmul(Hinv[i1:i2, i2:])

        # Undo act-order permutation so weights are in original column order
        Q = Q[:, invperm]

        # Final params from fully quantised W
        quantizer.find_params(self.layer.weight.data.float())

        return Q, quantizer.scale, quantizer.zero

    # ------------------------------------------------------------------
    def free(self) -> None:
        del self.H
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model traversal utilities
# ---------------------------------------------------------------------------

def find_linear_layers(
    module: nn.Module,
    prefix: str = "",
) -> dict[str, nn.Linear]:
    """Recursively collect all nn.Linear layers with their dotted names."""
    result = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            result[full] = child
        else:
            result.update(find_linear_layers(child, full))
    return result


_LAYER_ATTRS = [
    "model.layers",            # LLaMA, Mistral, Qwen, Gemma …
    "transformer.h",           # GPT-2, GPT-J, GPT-NeoX
    "model.decoder.layers",    # OPT
    "transformer.blocks",      # MPT
]


def get_sequential_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the list of transformer decoder blocks."""
    for attr in _LAYER_ATTRS:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, list)):
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(
        "Cannot locate transformer blocks automatically. "
        "Tried: " + ", ".join(_LAYER_ATTRS)
    )


def get_embeddings(model: nn.Module) -> list[nn.Module]:
    """Return modules that run before the first transformer block."""
    _EMBED_ATTRS = [
        ["model.embed_tokens"],
        ["transformer.wte", "transformer.wpe"],
        ["model.decoder.embed_tokens", "model.decoder.embed_positions",
         "model.decoder.project_in"],
        ["transformer.wte"],
    ]
    for attrs in _EMBED_ATTRS:
        try:
            mods = []
            for a in attrs:
                obj = model
                for part in a.split("."):
                    obj = getattr(obj, part)
                mods.append(obj)
            return mods
        except AttributeError:
            continue
    return []


# ---------------------------------------------------------------------------
# Main quantization routine
# ---------------------------------------------------------------------------

@contextmanager
def _capture_input(layer: nn.Module, store: list):
    """Hook that saves the first positional argument of a forward call."""
    def hook(mod, args, kwargs):
        store.append(args[0].detach().cpu())
    h = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def quantize_model(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    bits: int = 4,
    groupsize: int = 128,
    symmetric: bool = True,
    blocksize: int = 128,
    percdamp: float = 0.01,
    device: str = "cpu",
) -> dict[str, dict]:
    """
    Quantize all Linear layers in *model* using GPTQ.

    Returns a dict mapping layer name → {"scale": ..., "zero": ...} so the
    caller can store quantization metadata alongside the model.

    The model's weight tensors are updated in-place with the quantised values.
    """
    model.eval()
    blocks = get_sequential_blocks(model)
    embeddings = get_embeddings(model)
    quant_meta: dict[str, dict] = {}

    # Move embeddings to device so the first pass produces activations there
    for emb in embeddings:
        emb.to(device)

    # ------------------------------------------------------------------ #
    # Step 1: collect inputs to the first transformer block               #
    # ------------------------------------------------------------------ #
    block_inputs: list[torch.Tensor] = []
    extra_kwargs: list[dict] = []  # attention_mask, position_ids, …

    def _input_hook(mod, args, kwargs):
        block_inputs.append(args[0].detach().cpu())
        extra_kwargs.append({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                              for k, v in kwargs.items()})

    hook = blocks[0].register_forward_pre_hook(_input_hook, with_kwargs=True)
    with torch.inference_mode():
        for seq in calibration_data:
            try:
                model(seq.to(device))
            except Exception:
                pass   # OK — we only need the first-block inputs
    hook.remove()

    logger.info(f"Collected {len(block_inputs)} calibration inputs.")

    # ------------------------------------------------------------------ #
    # Step 2: quantize block by block                                      #
    # ------------------------------------------------------------------ #
    for b_idx, block in enumerate(blocks):
        logger.info(f"Quantizing block {b_idx + 1}/{len(blocks)} …")
        block = block.to(device)

        linears = find_linear_layers(block)
        gptq_layers = {name: GPTQLayer(lin) for name, lin in linears.items()}

        # Hooks to feed calibration inputs through the block and collect
        # per-linear activations.
        handles = []
        for name, lin in linears.items():
            def _make_hook(n):
                def _h(mod, args, kwargs):
                    gptq_layers[n].add_batch(args[0].detach())
                return _h
            handles.append(
                lin.register_forward_pre_hook(_make_hook(name), with_kwargs=True)
            )

        with torch.inference_mode():
            for inp, kw in zip(block_inputs, extra_kwargs):
                kw_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in kw.items()}
                block(inp.to(device), **kw_dev)

        for h in handles:
            h.remove()

        # Run GPTQ on each linear
        for name, gptq in gptq_layers.items():
            quantizer = Quantizer(bits=bits, symmetric=symmetric, groupsize=groupsize)
            Q, scale, zero = gptq.quantize(quantizer, blocksize=blocksize, percdamp=percdamp)
            linears[name].weight.data = Q.to(linears[name].weight.dtype)
            quant_meta[f"blocks.{b_idx}.{name}"] = {"scale": scale.cpu(), "zero": zero.cpu()}
            gptq.free()

        # Collect outputs of this block to use as inputs for the next
        new_inputs: list[torch.Tensor] = []

        def _out_hook(mod, args, output):
            new_inputs.append(
                (output[0] if isinstance(output, tuple) else output).detach().cpu()
            )

        out_h = block.register_forward_hook(_out_hook)
        with torch.inference_mode():
            for inp, kw in zip(block_inputs, extra_kwargs):
                kw_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in kw.items()}
                block(inp.to(device), **kw_dev)
        out_h.remove()

        block.cpu()
        torch.cuda.empty_cache()
        block_inputs = new_inputs

    return quant_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPTQ post-training quantization for causal LMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bit-width")
    parser.add_argument("--groupsize", type=int, default=128,
                        help="Group size for quantization (-1 = per-row)")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use symmetric quantization (zero-point = 0)")
    parser.add_argument("--dataset", default="wikitext2",
                        choices=["wikitext2", "c4", "ptb"],
                        help="Calibration dataset")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration sequences")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Calibration sequence length (tokens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Hessian damping factor (fraction of mean diagonal)")
    parser.add_argument("--blocksize", type=int, default=128,
                        help="GPTQ column block size")
    parser.add_argument("--device", default=None,
                        help="Device (default: cuda if available, else cpu)")
    parser.add_argument("--save", default=None,
                        help="Directory to save the quantized model")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()

    logger.info(
        f"Preparing {args.nsamples} calibration sequences "
        f"(dataset={args.dataset}, seqlen={args.seqlen})"
    )
    calib_data = get_calibration_data(
        dataset=args.dataset,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
        seed=args.seed,
    )

    logger.info(
        f"Starting GPTQ  bits={args.bits}  groupsize={args.groupsize}  "
        f"symmetric={args.symmetric}  device={device}"
    )
    quant_meta = quantize_model(
        model=model,
        calibration_data=calib_data,
        bits=args.bits,
        groupsize=args.groupsize,
        symmetric=args.symmetric,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        device=device,
    )

    logger.info(f"Quantized {len(quant_meta)} linear layers.")

    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        logger.info(f"Saved quantized model to '{args.save}'")


if __name__ == "__main__":
    main()
