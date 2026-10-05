"""Surrogate = a differentiable stand-in for one stage of a removal pipeline.

`bind(ctx)` sees the clean (watermarked, unperturbed) group once and returns a loss
function over perturbed batches. Anything expensive that only depends on the clean
frames (reference outputs, prompts, token targets) is computed in bind.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor

from ..attack import LossFn
from ..region import Rect, box_mask


@dataclass
class Context:
    clean: Tensor  # (N, 3, H, W) watermarked frames (crop around the mark), in [0, 1]
    logo: Rect  # where the mark sits inside the crop

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.clean.shape[-2:]
        return h, w

    def logo_mask(self, dilate: int = 0) -> Tensor:
        return box_mask(self.logo, *self.size, dilate=dilate).to(self.clean)


class Surrogate(Protocol):
    name: str

    def bind(self, ctx: Context) -> LossFn: ...


@dataclass
class Ensemble:
    """Weighted sum of surrogate losses. `last` holds each term of the latest call, for logging."""

    members: Sequence[tuple[Surrogate, float]]
    name: str = "ensemble"
    last: dict[str, float] = field(default_factory=dict)

    @property
    def zero(self) -> bool:
        return all(getattr(s, "zero", False) for s, _ in self.members)

    def bind(self, ctx: Context) -> LossFn:
        bound = [(s.name, s.bind(ctx), w) for s, w in self.members]

        def loss(x: Tensor) -> Tensor:
            total = x.new_zeros(())
            for name, fn, weight in bound:
                term = fn(x)
                self.last[name] = float(term.detach())
                total = total + weight * term
            return total

        return loss
