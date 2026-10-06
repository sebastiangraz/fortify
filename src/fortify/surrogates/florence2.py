"""Detection stage: Florence-2 open-vocabulary detection ("watermark").

This is how WatermarkRemover-AI builds its mask. If the detector stops boxing the logo,
the inpainter is never told to remove it.

Loss (targeted, a decoy): at bind time, run the real detector on each clean frame. Its
answer is a token sequence, `<s>watermark<loc_x0><loc_y0><loc_x1><loc_y1>...</s>`. Every
box that covers most of the logo gets its four <loc_*> tokens swapped for a decoy box in the
context ring beside the logo. The loss is the cross-entropy of that edited answer's
location tokens, so PGD steers the detector towards boxing the decoy. The remover then
inpaints a strip of background and leaves the mark.
- Why not "no box": Florence-2 OVD always answers with a box. On frames without a mark
  it boxes some other object, so an empty answer is off-distribution.
- Why not untargeted (ascend the clean answer's CE): tried in Phase 1. It moved boxes
  rather than removing them, and on one case it grew a box from the "S" symbol to the
  whole wordmark, which helps the attacker.
- Partial boxes are kept, not decoyed. On large wordmarks the clean detector often boxes
  only the symbol (coverage ~0.25). Swapping that box for the decoy made the detector
  find the whole mark on three Phase 1 cases. So only boxes that cover at least `whole` of
  the logo get the decoy; a partial box stays the target, which holds the miss.
- And a hinge on the whole-logo answer: the same answer with its logo box (or its first
  box) set to the whole logo. Its CE may rise but never drop below its clean value, so no
  frame's δ makes "box the whole mark" likelier than it was.

View: the real tool runs on the whole frame, shrunk to 768². With a View in the context
the surrogate builds that picture (background thumbnail, crop pasted at its place)
so the logo is seen at the attacker's scale, and decoy coordinates are in the
attacker's frame. Without one it falls back to the crop stretched to 768², which in
Phase 1 did not transfer to full-frame detection.

Cost: the vision tower (DaViT at 768²) holds ~3.5 GiB of activations per frame for the
backward, the language model far less. So the vision tower runs once per frame and its
features are shared by all prompts, frames are batched per prompt, and the loss is a
SumLoss over chunks of `chunk` frames, so pgd backpropagates one chunk at a time.
bf16 autocast was tried: no faster on the 5090 at 1-3 frames, and its input gradient
agreed with fp32 in sign on only ~75% of pixels, so it stays fp32.

Preprocessing is redone in torch so gradients reach the pixels. bind() checks it against
the real processor and warns on a mismatch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..attack import LossFn, SumLoss
from ..imageio import to_image
from ..models import load_florence2, normalize
from ..region import Rect
from .base import Context

TASK = "<OPEN_VOCABULARY_DETECTION>"
BINS = 1000  # Florence-2 quantises box coordinates to 1000 bins per axis


@dataclass
class Florence2Surrogate:
    prompts: tuple[str, ...] = ("watermark", "logo")
    chunk: int = 2  # frames per backward
    covers: float = 0.1  # a box "covers" the logo if it overlaps this share of the logo's area
    whole: float = 0.5  # ...and only boxes covering this share are swapped for the decoy
    gap: int = 16  # px between the (attacker-dilated) logo and the decoy
    name: str = "florence2"

    def bind(self, ctx: Context) -> LossFn:
        dev = ctx.clean.device
        model, processor = load_florence2(dev=str(dev))
        side = processor.image_processor.size
        size = (side["height"], side["width"]) if isinstance(side, dict) else (768, 768)
        cfg = model.config
        start, pad = cfg.text_config.decoder_start_token_id, cfg.text_config.pad_token_id
        embed = model.get_input_embeddings()
        loc_ids = _loc_token_ids(processor.tokenizer).to(dev)
        pixels, to_bins = _viewer(ctx, size)

        if ctx.view is None:
            ours = pixels(ctx.clean[:1])
            theirs = processor(images=to_image(ctx.clean[:1]), text=TASK, return_tensors="pt")
            theirs = theirs["pixel_values"].to(ours)
            # Mean, not max: bicubic in torch and PIL differ by up to ~0.15 on sharp edges
            # (mean ~0.001); a wrong size or normalisation shows up as ≫ 0.01.
            if theirs.shape == ours.shape and (ours - theirs).abs().mean() > 0.01:
                warnings.warn(
                    f"florence2 preprocessing drifts from the processor: {(ours - theirs).abs().mean():.3f}"
                )

        logo = to_bins(ctx.logo)
        decoy = _decoy(ctx, self.gap)
        if decoy is None:
            warnings.warn("florence2: no room for a decoy box in the crop's ring; term disabled")
            return lambda x: x.sum() * 0
        decoy_ids = loc_ids[torch.tensor(to_bins(decoy), device=dev)]  # all 1000 bins present
        logo_ids = loc_ids[torch.tensor(logo, device=dev)]

        def loc_ce(feats: Tensor, embeds: Tensor, slots: Tensor, seqs: list[Tensor]) -> Tensor:
            """Mean CE of each answer's <loc_*> tokens, teacher-forced on feats[k] for seqs[k]."""
            labels = torch.full((len(seqs), max(len(s) for s in seqs)), pad, device=dev)
            for k, s in enumerate(seqs):
                labels[k, : len(s)] = s
            weight = torch.isin(labels, loc_ids).float()  # pad and text tokens weigh 0
            decoder_in = torch.cat([torch.full_like(labels[:, :1], start), labels[:, :-1]], 1)
            e = embeds.expand(len(seqs), -1, -1).clone()
            e[:, slots] = feats
            logits = model(inputs_embeds=e, decoder_input_ids=decoder_in).logits
            ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
            return (ce * weight).sum(1) / weight.sum(1).clamp(min=1)

        # Per prompt: the text+image-placeholder embedding (same for every frame) and, per
        # frame, the edited answer to descend towards plus the whole-logo answer to hold off.
        # Frames where no box covers the logo (the detector already misses it), or only part
        # of it, keep their own answer, so δ doesn't undo that.
        prompts = []  # (embeds, image slots, {frame: target}, {frame: whole}, {frame: whole CE})
        with torch.no_grad():
            pv = pixels(ctx.clean)
            feats_clean = model.get_image_features(pv).pooler_output
            for prompt in self.prompts:
                input_ids = processor(
                    text=TASK + prompt, images=to_image(ctx.clean[:1]), return_tensors="pt"
                )["input_ids"].to(dev)
                embeds, slots = embed(input_ids), input_ids[0] == cfg.image_token_id
                targets: dict[int, Tensor] = {}
                wholes: dict[int, Tensor] = {}
                for i in range(ctx.clean.shape[0]):
                    gen = model.generate(
                        input_ids=input_ids,
                        pixel_values=pv[i : i + 1],
                        max_new_tokens=256,
                        num_beams=3,
                        do_sample=False,
                    )
                    # gen starts with decoder_start; the labels are what follows. Teacher-forced
                    # on clean, the <loc_*> positions reproduce the generated boxes (checked
                    # token by token).
                    labels = gen[0, 1:].clone()
                    whole = labels.clone()
                    pos = torch.isin(labels, loc_ids).nonzero().flatten().tolist()
                    boxes = [pos[k : k + 4] for k in range(0, len(pos) - 3, 4)]
                    on_logo = None
                    for idx in boxes:
                        box = tuple(int((loc_ids == labels[j]).nonzero()) for j in idx)
                        cover = _overlap(box, logo)
                        if cover > self.covers and on_logo is None:
                            on_logo = idx
                        if cover >= self.whole:
                            labels[idx] = decoy_ids
                    if boxes:
                        whole[on_logo or boxes[0]] = logo_ids
                        wholes[i] = whole
                    targets[i] = labels
                floor = {
                    i: float(loc_ce(feats_clean[i : i + 1], embeds, slots, [s])[0])
                    for i, s in wholes.items()
                }
                prompts.append((embeds, slots, targets, wholes, floor))
        frames = list(range(ctx.clean.shape[0]))
        n_targets = len(frames) * len(prompts)

        def chunk_loss(chunk: list[int]) -> LossFn:
            def loss(x: Tensor) -> Tensor:
                feats = model.get_image_features(pixels(x[chunk])).pooler_output
                total = x.new_zeros(())
                for embeds, slots, targets, wholes, floor in prompts:
                    # One LM pass scores both answers: every frame's target, then the
                    # whole-logo answer of each frame that has one.
                    held = [i for i in chunk if i in wholes]
                    rows = list(range(len(chunk))) + [chunk.index(i) for i in held]
                    seqs = [targets[i] for i in chunk] + [wholes[i] for i in held]
                    ce = loc_ce(feats[rows], embeds, slots, seqs)
                    target_ce, whole_ce = ce[: len(chunk)], ce[len(chunk) :]
                    floors = torch.tensor([floor[i] for i in held], device=dev)
                    total = total + target_ce.sum() + F.relu(floors - whole_ce).sum()
                return total / n_targets

            return loss

        return SumLoss(
            chunk_loss(frames[k : k + self.chunk]) for k in range(0, len(frames), self.chunk)
        )


def _viewer(ctx: Context, size: tuple[int, int]):
    """(pixels, to_bins): crop batch → the model's normalised input, and crop Rect → loc bins."""
    h, w = ctx.size
    resize = lambda x, hw: F.interpolate(
        x, size=hw, mode="bicubic", align_corners=False, antialias=True
    )
    if ctx.view is None:

        def to_bins(r: Rect) -> tuple[int, int, int, int]:
            x0, y0, x1, y1 = r.xyxy()
            return _bin(x0 / w), _bin(y0 / h), _bin(x1 / w), _bin(y1 / h)

        return (lambda x: normalize(resize(x, size).clamp(0, 1))), to_bins

    (fw, fh), (ax, ay) = ctx.view.frame, ctx.view.at
    sy, sx = size[0] / fh, size[1] / fw
    x0, y0 = round(ax * sx), round(ay * sy)
    x1, y1 = max(x0 + 1, round((ax + w) * sx)), max(y0 + 1, round((ay + h) * sy))
    if ctx.view.background is not None:
        bg = resize(ctx.view.background.to(ctx.clean), size).clamp(0, 1)
    else:  # unknown surroundings: the crop's mean colour
        bg = ctx.clean.mean(dim=(0, 2, 3), keepdim=True).expand(1, 3, *size)

    def pixels(x: Tensor) -> Tensor:
        canvas = bg.expand(x.shape[0], -1, -1, -1).clone()
        canvas[..., y0:y1, x0:x1] = resize(x, (y1 - y0, x1 - x0)).clamp(0, 1)
        return normalize(canvas)

    def to_bins(r: Rect) -> tuple[int, int, int, int]:
        rx0, ry0, rx1, ry1 = r.xyxy()
        return (
            _bin((ax + rx0) / fw),
            _bin((ay + ry0) / fh),
            _bin((ax + rx1) / fw),
            _bin((ay + ry1) / fh),
        )

    return pixels, to_bins


def _decoy(ctx: Context, gap: int) -> Rect | None:
    """The biggest strip of the ring beside the logo, clear of it by `gap` and of the feathered
    crop border, spanning the logo along the other axis."""
    h, w = ctx.size
    lx0, ly0, lx1, ly1 = ctx.logo.xyxy()
    ring = min(lx0, ly0, w - lx1, h - ly1)
    m = max(2, ring // 4)  # δ ramps to 0 over the outer quarter of the ring
    strips = [
        Rect(lx0, m, lx1 - lx0, ly0 - gap - m),  # above
        Rect(lx0, ly1 + gap, lx1 - lx0, h - m - ly1 - gap),  # below
        Rect(m, ly0, lx0 - gap - m, ly1 - ly0),  # left
        Rect(lx1 + gap, ly0, w - m - lx1 - gap, ly1 - ly0),  # right
    ]
    strips = [s for s in strips if s.w >= 4 and s.h >= 4]
    return max(strips, key=lambda s: s.w * s.h, default=None)


def _bin(v: float) -> int:
    return min(BINS - 1, max(0, int(v * BINS)))


def _overlap(box: tuple[int, ...], logo: tuple[int, ...]) -> float:
    """Share of the logo's area (both in loc bins) that the box covers."""
    ix = max(0, min(box[2], logo[2]) - max(box[0], logo[0]))
    iy = max(0, min(box[3], logo[3]) - max(box[1], logo[1]))
    area = max(1, (logo[2] - logo[0]) * (logo[3] - logo[1]))
    return ix * iy / area


def _loc_token_ids(tokenizer) -> Tensor:
    ids = [tokenizer.convert_tokens_to_ids(f"<loc_{i}>") for i in range(BINS)]
    return torch.tensor([i for i in ids if i is not None and i != tokenizer.unk_token_id])
