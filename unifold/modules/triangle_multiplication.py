from functools import partialmethod
from typing import Optional, List
import torch
import torch.nn as nn
from .common import Linear
from unicore.utils import (
    permute_final_dims,
)
from unicore.modules import (
    LayerNorm,
)


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid, outgoing=True):
        super(TriangleMultiplication, self).__init__()
        self.outgoing = outgoing

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")

        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")

        self.layer_norm_in = LayerNorm(d_pair)
        self.layer_norm_out = LayerNorm(d_hid)

        self._alphafold_original_mode = False

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        if not self._alphafold_original_mode:
            # divided by 1/sqrt(dim) for numerical stability
            mask = mask * (mask.shape[-2] ** -0.5)

        z = self.layer_norm_in(z)
        g = nn.functional.linear(z, self.linear_g.weight)
        ab = self.linear_ab_p(z) * mask
        ab *= torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)
        del z, ab

        if self.outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = b.transpose(-1, -3)
        else:
            b = permute_final_dims(b, (2, 0, 1))
            a = a.transpose(-1, -3)
        x = torch.matmul(a, b)
        del a, b

        x = permute_final_dims(x, (1, 2, 0))

        x = self.layer_norm_out(x)
        x = nn.functional.linear(x, self.linear_z.weight)
        return x, g

    def get_output_bias(self):
        return self.linear_z.bias, self.linear_g.bias


class TriangleMultiplicationOutgoing(TriangleMultiplication):
    __init__ = partialmethod(TriangleMultiplication.__init__, outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplication):
    __init__ = partialmethod(TriangleMultiplication.__init__, outgoing=False)
