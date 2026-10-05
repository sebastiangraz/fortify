"""Weight-free surrogates for tests, selftest and consumer smoke runs.

- toy: a fixed random conv net standing in for a remover. PGD must be able to move it.
- null: δ ≡ 0. vaccinate() sees `zero` and skips the attack, including the noise start.
  A consumer can check that a zero δ leaves its output byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..attack import LossFn
from .base import Context


@dataclass
class ToySurrogate:
    name: str = "toy"

    def bind(self, ctx: Context) -> LossFn:
        g = torch.Generator().manual_seed(0)
        net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Tanh(), nn.Conv2d(8, 3, 3, padding=1))
        for p in net.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3
            p.requires_grad_(False)
        net = net.to(ctx.clean)
        hole = ctx.logo_mask(4)
        with torch.no_grad():
            ref = net(ctx.clean)

        def loss(x: Tensor) -> Tensor:
            return -(((net(x) - ref) * hole) ** 2).mean()

        return loss


@dataclass
class NullSurrogate:
    name: str = "null"
    zero: bool = True

    def bind(self, ctx: Context) -> LossFn:
        return lambda x: x.sum() * 0
