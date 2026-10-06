"""Real removal pipelines, run in inference mode, for red-teaming.

Each remover takes a full frame (1, 3, H, W) and the true logo rect (some attackers get
to know it, e.g. by drawing the mask by hand). It returns (restored frame, info), where
info carries detector-side numbers.

- oracle-lama:   attacker draws the mask by hand (dilated box) → LaMa. Worst case for us.
- loose-lama:    the same with a sloppier box (dilated 24 px). Held out: the lama
                 surrogate trains on boxes dilated 4/8/16 and a SAM mask, never on this one.
- florence-lama: Florence-2 OVD "watermark" → boxes → LaMa. The WatermarkRemover-AI pipeline.
- sam-lama:      attacker drags a box → SAM mask (dilated) → LaMa. IOPaint-style.

Video-only removers (SAM2 propagation + ProPainter/E2FGVI) need clips, not frames, and
ProPainter is non-commercial. Run them by hand from their own repos for now; see eval/README.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fortify.imageio import to_image
from fortify.models import lama_inpaint, load_florence2, load_lama
from fortify.region import Rect, box_mask

DILATE = 8


def _dilate(mask: Tensor, px: int) -> Tensor:
    return F.max_pool2d(mask, 2 * px + 1, stride=1, padding=px) if px else mask


@torch.no_grad()
def oracle_lama(x: Tensor, logo: Rect, dilate: int = DILATE) -> tuple[Tensor, dict]:
    h, w = x.shape[-2:]
    mask = box_mask(logo, h, w, dilate).to(x)
    return lama_inpaint(load_lama(str(x.device)), x, mask), {}


def loose_lama(x: Tensor, logo: Rect) -> tuple[Tensor, dict]:
    return oracle_lama(x, logo, dilate=24)


@torch.no_grad()
def florence_lama(x: Tensor, logo: Rect, prompt: str = "watermark") -> tuple[Tensor, dict]:
    model, processor = load_florence2(dev=str(x.device))
    task = "<OPEN_VOCABULARY_DETECTION>"
    image = to_image(x)
    inputs = processor(text=task + prompt, images=image, return_tensors="pt").to(x.device)
    gen = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
        do_sample=False,
    )
    text = processor.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(text, task=task, image_size=image.size)[task]
    h, w = x.shape[-2:]
    mask = torch.zeros((1, 1, h, w), device=x.device)
    boxes = parsed.get("bboxes", [])
    for x0, y0, x1, y1 in boxes:
        mask[..., max(0, int(y0)) : int(y1) + 1, max(0, int(x0)) : int(x1) + 1] = 1
    mask = _dilate(mask, DILATE)
    info = {"boxes": len(boxes), "logo_covered": _covered(mask, logo)}
    if not boxes:
        return x, info
    return lama_inpaint(load_lama(str(x.device)), x, mask), info


@torch.no_grad()
def sam_lama(x: Tensor, logo: Rect) -> tuple[Tensor, dict]:
    from fortify.surrogates import Context
    from fortify.surrogates.sam import SamSurrogate

    sam = SamSurrogate()
    sam.bind(Context(clean=x, logo=logo))
    mask = (sam.logits_in_crop(x) > 0).float()
    mask = _dilate(mask, DILATE)
    info = {"mask_px": int(mask.sum()), "logo_covered": _covered(mask, logo)}
    return lama_inpaint(load_lama(str(x.device)), x, mask), info


def _covered(mask: Tensor, logo: Rect) -> float:
    x0, y0, x1, y1 = logo.xyxy()
    return float(mask[..., y0:y1, x0:x1].mean())


REMOVERS = {
    "oracle-lama": oracle_lama,
    "loose-lama": loose_lama,
    "florence-lama": florence_lama,
    "sam-lama": sam_lama,
}
