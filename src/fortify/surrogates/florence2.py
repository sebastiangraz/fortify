"""Detection stage: Florence-2 open-vocabulary detection ("watermark").

This is how WatermarkRemover-AI builds its mask. If the detector stops boxing the logo,
the inpainter is never told to remove it.

Loss: at bind time, run the real detector on each clean frame and keep the token
sequence it emits (boxes come out as <loc_*> tokens). The loss is the NEGATIVE
cross-entropy of that sequence's location tokens, so PGD makes the detector unlikely
to repeat the boxes. It is capped so one frame can't dominate.
Untargeted ascent can move a box rather than remove it. If eval shows that, switch to
a targeted variant (descend towards an empty or far-away answer); see README → Roadmap.

Preprocessing is redone in torch so gradients reach the pixels. bind() checks it against
the real processor and warns on a mismatch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..attack import LossFn
from ..imageio import to_image
from ..models import load_florence2, normalize
from .base import Context

TASK = "<OPEN_VOCABULARY_DETECTION>"


@dataclass
class Florence2Surrogate:
    prompts: tuple[str, ...] = ("watermark", "logo")
    cap: float = 12.0
    name: str = "florence2"

    def bind(self, ctx: Context) -> LossFn:
        dev = ctx.clean.device
        model, processor = load_florence2(dev=str(dev))
        side = processor.image_processor.size
        size = (side["height"], side["width"]) if isinstance(side, dict) else (768, 768)

        def pixels(x: Tensor) -> Tensor:
            x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
            return normalize(x.clamp(0, 1))

        targets: list[
            tuple[int, Tensor, Tensor, Tensor]
        ] = []  # frame, input_ids, labels, loc weight
        loc_ids = _loc_token_ids(processor.tokenizer)
        for i in range(ctx.clean.shape[0]):
            for prompt in self.prompts:
                inputs = processor(
                    text=TASK + prompt, images=to_image(ctx.clean[i : i + 1]), return_tensors="pt"
                )
                ours = pixels(ctx.clean[i : i + 1])
                theirs = inputs["pixel_values"].to(ours)
                if theirs.shape == ours.shape and (ours - theirs).abs().max() > 0.1:
                    warnings.warn(
                        f"florence2 preprocessing drifts from the processor: {(ours - theirs).abs().max():.3f}"
                    )
                input_ids = inputs["input_ids"].to(dev)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        pixel_values=theirs,
                        max_new_tokens=256,
                        num_beams=3,
                        do_sample=False,
                    )
                labels = gen[:, 1:]  # drop decoder_start; the model shifts labels right itself
                weight = torch.isin(labels, loc_ids.to(dev)).float()
                if weight.sum() > 0:  # nothing detected → nothing to suppress for this prompt
                    targets.append((i, input_ids, labels, weight))

        def loss(x: Tensor) -> Tensor:
            if not targets:
                return x.sum() * 0
            total = x.new_zeros(())
            pv = pixels(x)
            for i, input_ids, labels, weight in targets:
                logits = model(
                    input_ids=input_ids, pixel_values=pv[i : i + 1], labels=labels
                ).logits
                ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
                total = total + ((ce * weight).sum() / weight.sum()).clamp(max=self.cap)
            return -total / len(targets)

        return loss


def _loc_token_ids(tokenizer) -> Tensor:
    ids = [tokenizer.convert_tokens_to_ids(f"<loc_{i}>") for i in range(1000)]
    return torch.tensor([i for i in ids if i is not None and i != tokenizer.unk_token_id])
