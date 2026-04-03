from typing import Optional

import torch


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
) -> torch.Tensor:
    """Round-to-nearest quantize then dequantize (simulated quantization)."""
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer:
    """
    Computes per-column (or per-group) affine quantization parameters and
    applies simulated quantization to a weight tensor.

    Scale/zero are shaped (out_features, n_groups) after find_params().
    """

    def __init__(self, bits: int, symmetric: bool = True, groupsize: int = -1):
        assert 1 <= bits <= 8, "bits must be in [1, 8]"
        self.bits = bits
        self.symmetric = symmetric
        self.groupsize = groupsize          # -1 → whole row as one group
        self.maxq = 2**bits - 1
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None

    def find_params(self, W: torch.Tensor) -> None:
        """
        Compute per-group scale and zero for weight matrix W (out, in).
        After this call, self.scale and self.zero are (out, n_groups).
        """
        dev = W.device
        out, inp = W.shape
        g = inp if self.groupsize == -1 else self.groupsize
        assert inp % g == 0, (
            f"in_features ({inp}) must be divisible by groupsize ({g})"
        )
        n_groups = inp // g

        # Reshape to (out * n_groups, g) for vectorised min/max
        W_grouped = W.reshape(out * n_groups, g)

        xmin = W_grouped.min(dim=1).values          # (out * n_groups,)
        xmax = W_grouped.max(dim=1).values

        if self.symmetric:
            xmax = torch.maximum(xmin.abs(), xmax.abs())
            scale = xmax / (self.maxq / 2)
            scale[scale == 0] = 1.0
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            scale = (xmax - xmin) / self.maxq
            scale[scale == 0] = 1.0
            zero = torch.round(-xmin / scale)

        self.scale = scale.reshape(out, n_groups).to(dev)
        self.zero  = zero.reshape(out, n_groups).to(dev)

    def quantize(self, W: torch.Tensor) -> torch.Tensor:
        """
        Simulated quantization of W (out, in).
        Uses the scale/zero computed by find_params().
        """
        assert self.scale is not None, "Call find_params() first."
        out, inp = W.shape
        g = inp if self.groupsize == -1 else self.groupsize
        n_groups = inp // g

        # Broadcast scale/zero: (out, n_groups) → (out, in)
        scale = self.scale.repeat_interleave(g, dim=1)   # (out, in)
        zero  = self.zero.repeat_interleave(g, dim=1)    # (out, in)

        return quantize_tensor(W, scale, zero, self.maxq)
