"""Inpainting stage: LaMa.

Threat: the attacker masks the logo (box, dilated) and LaMa fills it from the
surrounding context. Inside the hole LaMa never sees our pixels, so δ works through
the context ring around it.

Loss (DWV-style, "disrupting"): push LaMa's fill inside the hole away from the fill it
produces on the unperturbed frame, i.e. away from the plausible clean plate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..attack import LossFn
from ..models import lama_inpaint, load_lama
from .base import Context


@dataclass
class LamaSurrogate:
    dilate: int = 8  # px the attacker's mask grows past the logo box
    name: str = "lama"

    def bind(self, ctx: Context) -> LossFn:
        model = load_lama(str(ctx.clean.device))
        hole = ctx.logo_mask(self.dilate)
        with torch.no_grad():
            ref = lama_inpaint(model, ctx.clean, hole)
        area = hole.sum().clamp(min=1) * 3

        def loss(x: Tensor) -> Tensor:
            fill = lama_inpaint(model, x, hole)
            err = (((fill - ref) * hole) ** 2).sum(dim=(1, 2, 3)) / area
            return -err.mean()

        return loss
