"""Where δ may go, and the boxes surrogates are told about."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def clipped(self, width: int, height: int) -> Rect:
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(width, self.x + self.w), min(height, self.y + self.h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"rect {self} lies outside a {width}x{height} crop")
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def crop_around(logo: Rect, width: int, height: int, ring: int | None = None) -> Rect:
    """Recommended crop for a mark: the logo plus a context ring, clipped to the frame, even-sized.

    The ring is the context an inpainter reads around its hole, so δ there is what reaches it.
    By default it is half the logo's long side, at least 32 px.
    """
    ring = ring if ring is not None else max(32, max(logo.w, logo.h) // 2)
    x0, y0 = max(0, logo.x - ring), max(0, logo.y - ring)
    x1, y1 = min(width, logo.x + logo.w + ring), min(height, logo.y + logo.h + ring)
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    return Rect(x0, y0, w, h)


def box_mask(rect: Rect, height: int, width: int, dilate: int = 0) -> Tensor:
    """(1, 1, H, W) binary mask of the rect grown by `dilate` px (removers dilate their masks)."""
    m = torch.zeros((1, 1, height, width))
    x0, y0, x1, y1 = rect.xyxy()
    m[
        ...,
        max(0, y0 - dilate) : min(height, y1 + dilate),
        max(0, x0 - dilate) : min(width, x1 + dilate),
    ] = 1
    return m


def feather(height: int, width: int, px: int) -> Tensor:
    """(1, 1, H, W) weight: 1 inside, linear ramp to 0 over `px` at the crop border.

    The consumer pastes δ back over the frame at the crop rect; the ramp hides the seam.
    """
    if px <= 0:
        return torch.ones((1, 1, height, width))
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    ry = torch.minimum(ys + 1, height - ys).clamp(max=px) / px
    rx = torch.minimum(xs + 1, width - xs).clamp(max=px) / px
    return (ry[:, None] * rx[None, :])[None, None]


def resize_to(x: Tensor, size: tuple[int, int]) -> Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
