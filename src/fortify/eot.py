"""Differentiable stand-ins for what happens to a frame after we perturb it.

The goal is Expectation over Transformation (EOT): average the attack gradient over
random draws of these so δ survives the real pipeline (yuv420p + H.264, rescaling,
light blur). Rounding uses a straight-through estimator (forward = round, backward = identity).

H.264 is approximated by an 8×8 DCT quantizer (JPEG tables) on YCbCr with 4:2:0 chroma.
That is not H.264's integer transform, but it removes the same high frequencies,
which is what kills naive perturbations. Calibrate the quality range against real
x264 output with eval/ (see README "Calibrating the codec proxy").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

_LUMA_Q = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
_CHROMA_Q = torch.full((8, 8), 99.0)
_CHROMA_Q[:4, :4] = torch.tensor(
    [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]], dtype=torch.float32
)


def _dct_matrix(n: int = 8) -> Tensor:
    k = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2 / n)
    d[0] /= math.sqrt(2)
    return d


_DCT = _dct_matrix()


def round_ste(x: Tensor) -> Tensor:
    return x + (torch.round(x) - x).detach()


def quality_table(base: Tensor, quality: float) -> Tensor:
    """libjpeg's quality scaling."""
    quality = min(max(quality, 1.0), 100.0)
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return torch.clamp(torch.floor((base * scale + 50) / 100), min=1)


def rgb_to_ycbcr(x: Tensor) -> Tensor:
    r, g, b = x.unbind(1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], 1)


def ycbcr_to_rgb(x: Tensor) -> Tensor:
    y, cb, cr = x.unbind(1)
    cb, cr = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], 1)


def _blockwise(plane: Tensor, table: Tensor) -> Tensor:
    """DCT → quantize (STE) → IDCT on one (N, 1, H, W) plane in [0, 255], H and W multiples of 8."""
    n, _, h, w = plane.shape
    d = _DCT.to(plane)
    blocks = (plane - 128).reshape(n, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
    coeffs = d @ blocks @ d.T
    q = table.to(plane)
    coeffs = round_ste(coeffs / q) * q
    blocks = d.T @ coeffs @ d
    return blocks.permute(0, 1, 3, 2, 4).reshape(n, 1, h, w) + 128


def jpeg_proxy(x: Tensor, quality: float, subsample: bool = True) -> Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    padded = F.pad(x, (0, pw, 0, ph), mode="replicate")
    ycc = rgb_to_ycbcr(padded) * 255
    y, cb, cr = ycc[:, 0:1], ycc[:, 1:2], ycc[:, 2:3]
    y = _blockwise(y, quality_table(_LUMA_Q, quality))
    chroma = []
    for c in (cb, cr):
        if subsample:
            c = F.avg_pool2d(c, 2)
        c = _blockwise(c, quality_table(_CHROMA_Q, quality))
        if subsample:
            c = F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
        chroma.append(c)
    out = ycbcr_to_rgb(torch.cat([y, *chroma], 1) / 255)
    return out[..., :h, :w].clamp(0, 1)


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return x
    radius = max(1, math.ceil(sigma * 3))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def resize_roundtrip(x: Tensor, factor: float) -> Tensor:
    if abs(factor - 1) < 1e-3:
        return x
    h, w = x.shape[-2:]
    small = F.interpolate(
        x,
        size=(max(8, round(h * factor)), max(8, round(w * factor))),
        mode="bilinear",
        align_corners=False,
        antialias=factor < 1,
    )
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def shift(x: Tensor, dx: int, dy: int) -> Tensor:
    if dx == 0 and dy == 0:
        return x
    h, w = x.shape[-2:]
    p = max(abs(dx), abs(dy))
    x = F.pad(x, (p, p, p, p), mode="replicate")
    return x[..., p - dy : p - dy + h, p - dx : p - dx + w]


@dataclass
class CodecProxy:
    """Random draw of the post-processing chain, called once per EOT sample."""

    quality: tuple[float, float] = (45, 85)
    p_jpeg: float = 0.9
    resize: tuple[float, float] = (0.6, 1.0)
    p_resize: float = 0.4
    blur_sigma: tuple[float, float] = (0.0, 0.8)
    p_blur: float = 0.3
    noise_std: float = 1.5 / 255
    p_noise: float = 0.3
    max_shift: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __call__(self, x: Tensor) -> Tensor:
        r = self._rng
        if self.max_shift and r.random() < 0.5:
            x = shift(
                x,
                r.randint(-self.max_shift, self.max_shift),
                r.randint(-self.max_shift, self.max_shift),
            )
        if r.random() < self.p_resize:
            x = resize_roundtrip(x, r.uniform(*self.resize))
        if r.random() < self.p_blur:
            x = gaussian_blur(x, r.uniform(*self.blur_sigma))
        if r.random() < self.p_jpeg:
            x = jpeg_proxy(x, r.uniform(*self.quality))
        if r.random() < self.p_noise:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0, 1)
        return x
