"""Segmentation stage: SAM / SAM2 prompted with a box around the logo.

This is the IOPaint "click to mask" flow, and the first step of video removers
(SAM2 propagates the mask, ProPainter fills it).

Loss (Attack-SAM "ClipMSE"): push the mask logits under the logo below -tau, so the
predicted mask comes back empty. Logits that are already below -tau stop counting.

Preprocessing is redone in torch so gradients reach the pixels. SAM pads the longest
side to 1024 and SAM2 stretches to 1024², so bind() tries both against the real
processor and keeps the one that matches.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from ..attack import LossFn
from ..imageio import to_image
from ..models import load_sam, normalize
from .base import Context

SIDE = 1024


@dataclass
class SamSurrogate:
    tau: float = 2.0
    dilate: int = 4
    name: str = "sam"

    def bind(self, ctx: Context) -> LossFn:
        dev = ctx.clean.device
        model, processor = load_sam(dev=str(dev))
        h, w = ctx.size
        box = [float(v) for v in ctx.logo.xyxy()]
        inputs = processor(images=to_image(ctx.clean[:1]), input_boxes=[[box]], return_tensors="pt")
        theirs = inputs["pixel_values"].to(ctx.clean)
        input_boxes = inputs["input_boxes"].to(dev)

        scale = SIDE / max(h, w)
        fit = (int(h * scale + 0.5), int(w * scale + 0.5))
        modes = {"pad": fit, "stretch": (SIDE, SIDE)}

        def pixels(x: Tensor, mode: str) -> Tensor:
            rh, rw = modes[mode]
            x = normalize(
                F.interpolate(
                    x.clamp(0, 1),
                    size=(rh, rw),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            )
            return F.pad(x, (0, SIDE - rw, 0, SIDE - rh)) if mode == "pad" else x

        errors = {
            m: float((pixels(ctx.clean[:1], m) - theirs).abs().max())
            for m in modes
            if theirs.shape[-2:] == (SIDE, SIDE)
        }
        mode = min(errors, key=errors.get) if errors else "pad"
        if errors and errors[mode] > 0.1:
            warnings.warn(f"sam preprocessing drifts from the processor: {errors}")

        target = ctx.logo_mask(self.dilate)
        area = target.sum().clamp(min=1)

        def logits_in_crop(x: Tensor) -> Tensor:
            out = model(
                pixel_values=pixels(x, mode),
                input_boxes=input_boxes.expand(x.shape[0], -1, -1),
                multimask_output=False,
            )
            low = out.pred_masks.flatten(1, -3)[:, :1]  # (N, 1, 256, 256)
            full = F.interpolate(low, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
            rh, rw = modes[mode]
            return F.interpolate(
                full[..., :rh, :rw], size=(h, w), mode="bilinear", align_corners=False
            )

        def loss(x: Tensor) -> Tensor:
            logits = logits_in_crop(x)
            return ((F.relu(logits + self.tau) ** 2) * target).sum(dim=(1, 2, 3)).div(area).mean()

        self.logits_in_crop = logits_in_crop  # eval/ reuses it to measure mask area
        return loss
