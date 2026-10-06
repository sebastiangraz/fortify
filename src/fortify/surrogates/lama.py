"""Inpainting stage: LaMa.

Threat: the attacker masks the logo and LaMa fills it from the surrounding context.
Inside the hole LaMa never sees our pixels, so δ works through the context ring around it.

Loss (DWV-style, "disrupting"): push LaMa's fill inside the hole away from the fill it
produces on the unperturbed frame, i.e. away from the plausible clean plate.

Mask EOT: attackers draw different masks: a hand-drawn box at some dilation (IOPaint brush,
`oracle-lama`), a SAM mask that hugs the strokes (IOPaint's click-to-mask, `sam-lama`),
or WatermarkRemover-AI's undilated Florence-2 box. bind() builds a bank of masks and each
call of the loss uses the next one in turn. What Phase 2 measured (eval/README, Phase 2):
- The disruption is specific to the mask's edge. A δ trained on one box leaves a clean fill
  for a box 4 px off from it; the δ that breaks LaMa sits right at the hole's edge.
- Masks a few px apart (4/8/16) fight over the same pixels and break nothing; masks 16 px
  apart (8 and 24) coexist, and each breaks its own attacker. Hence the default.
- Cycling needs momentum in the attack (AttackConfig.momentum): plain sign steps chase the
  latest mask and oscillate. With momentum it costs the same as one mask.
- Summing every mask on every call (`cycle=False`) is K× the cost and was no better.
- A SAM mask in the bank (`sam:<px>`) broke SAM-masked fills only weakly and diluted the
  boxes, so it is available but not default.
Mask specs:
- `box:<px>`: the logo box dilated by px.
- `sam:<px>`: the real SAM's mask for the logo box (no grad, once per group), dilated by px.
  Skipped with a warning if SAM can't load or finds nothing.

LaMa holds ~4 GiB of activations per 976×418 frame for the backward, so the loss is a
SumLoss over chunks of `chunk` frames and pgd backpropagates one chunk at a time.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..attack import LossFn, SumLoss
from ..models import lama_inpaint, load_lama
from .base import Context

MASKS = ("box:8", "box:24")


@dataclass
class LamaSurrogate:
    masks: tuple[str, ...] = MASKS  # the attacker masks to train against, see module doc
    cycle: bool = True  # one mask per call, in turn; False = every mask on every call
    chunk: int = 2  # frames per backward
    name: str = "lama"

    def bind(self, ctx: Context) -> LossFn:
        model = load_lama(str(ctx.clean.device))
        holes = [h for spec in self.masks if (h := self._mask(spec, ctx)) is not None]
        if not holes:
            raise ValueError(f"lama: no usable mask in {self.masks}")
        with torch.no_grad():
            bank = [(hole, lama_inpaint(model, ctx.clean, hole), hole.sum() * 3) for hole in holes]

        n = ctx.clean.shape[0]

        def chunk_loss(lo: int, hi: int) -> LossFn:
            turn = itertools.cycle(bank)  # per chunk, so every chunk sees every mask

            def loss(x: Tensor) -> Tensor:
                total = x.new_zeros(())
                for hole, ref, area in [next(turn)] if self.cycle else bank:
                    fill = lama_inpaint(model, x[lo:hi], hole)
                    err = (((fill - ref[lo:hi]) * hole) ** 2).sum(dim=(1, 2, 3)) / area
                    total = total - err.sum() / n
                return total if self.cycle else total / len(bank)

            return loss

        return SumLoss(chunk_loss(k, min(n, k + self.chunk)) for k in range(0, n, self.chunk))

    def _mask(self, spec: str, ctx: Context) -> Tensor | None:
        kind, _, px = spec.partition(":")
        px = int(px or 0)
        if kind == "box":
            return ctx.logo_mask(px)
        if kind == "sam":
            return _sam_mask(ctx, px)
        raise ValueError(f"lama: unknown mask {spec!r} (box:<px> | sam:<px>)")


def _sam_mask(ctx: Context, px: int) -> Tensor | None:
    """What IOPaint's click-to-mask gives the attacker: SAM's mask for the logo box, dilated."""
    from .sam import SamSurrogate

    try:
        sam = SamSurrogate()
        sam.bind(ctx)
        with torch.no_grad():
            mask = (sam.logits_in_crop(ctx.clean[:1]) > 0).to(ctx.clean)
    except Exception as e:  # noqa: BLE001 - a missing optional model only drops this mask
        warnings.warn(f"lama: no SAM mask ({e}); training on boxes only")
        return None
    if px:
        mask = F.max_pool2d(mask, 2 * px + 1, stride=1, padding=px)
    if float(mask.sum()) < 16:
        warnings.warn("lama: SAM found nothing under the logo box; mask skipped")
        return None
    return mask
