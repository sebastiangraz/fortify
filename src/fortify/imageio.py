"""PNG ⇄ tensor helpers and the δ wire format."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from . import DELTA_OFFSET


def load_png(data: bytes) -> Tensor:
    """PNG/JPEG bytes → (1, 3, H, W) float in [0, 1]. Alpha is dropped."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return to_tensor(img)


def to_tensor(img: Image.Image) -> Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].float() / 255


def to_image(x: Tensor) -> Image.Image:
    arr = (x[0].detach().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


def save_png(x: Tensor) -> bytes:
    buf = io.BytesIO()
    to_image(x).save(buf, format="PNG")
    return buf.getvalue()


def encode_delta(delta: Tensor) -> bytes:
    """(1, 3, H, W) δ on whole 8-bit levels → RGB PNG with uint8 = 128 + δ·255."""
    levels = torch.round(delta[0].detach() * 255).clamp(-DELTA_OFFSET, 255 - DELTA_OFFSET)
    arr = (levels + DELTA_OFFSET).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def decode_delta(data: bytes) -> Tensor:
    arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)
    return torch.from_numpy(arr - DELTA_OFFSET).permute(2, 0, 1)[None].float() / 255
