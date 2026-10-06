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

from ..attack import LossFn, SumLoss
from ..region import Rect, box_mask


@dataclass(frozen=True)
class View:
    """Where the crop sits in its full frame.

    Some removers run on the whole frame (Florence-2 sees it shrunk to 768²), so the logo
    reaches them at a different scale, and with different neighbours, than in the crop.
    A surrogate given a View rebuilds that picture: the background, with the crop
    pasted in at its place.
    """

    frame: tuple[int, int]  # (W, H) of the full frame
    at: tuple[int, int]  # (x, y) of the crop's top-left corner in the frame
    background: Tensor | None = None  # (1, 3, h, w) the whole frame at any size; None = unknown


@dataclass
class Context:
    clean: Tensor  # (N, 3, H, W) watermarked frames (crop around the mark), in [0, 1]
    logo: Rect  # where the mark sits inside the crop
    view: View | None = None

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
    """Weighted sum of surrogate losses. `last` holds each term of the latest call, for logging.

    The bound loss is a balanced SumLoss: pgd backpropagates the surrogates one at a time,
    and each one's gradient is normalised first, so a weight is that surrogate's share of
    the step rather than a fudge factor for its loss scale.
    """

    members: Sequence[tuple[Surrogate, float]]
    name: str = "ensemble"
    last: dict[str, float] = field(default_factory=dict)

    @property
    def zero(self) -> bool:
        return all(getattr(s, "zero", False) for s, _ in self.members)

    def bind(self, ctx: Context) -> LossFn:
        loss = SumLoss(
            [s.bind(ctx) for s, _ in self.members],
            weights=[w for _, w in self.members],
            names=[s.name for s, _ in self.members],
            balance=True,
        )
        self.last = loss.last
        return loss
