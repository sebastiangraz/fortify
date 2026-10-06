"""One group of crops in, one δ out. Used by the service, the CLI and eval/."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from .attack import AttackConfig, pgd, rfgsm_config
from .eot import CodecProxy
from .models import device
from .region import Rect, feather
from .surrogates import Context, Surrogate, View

Strength = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Preset:
    attack: AttackConfig
    eot: bool


# ε are whole 8-bit levels, so the quantized δ never exceeds them. All presets start
# from noise: the lama loss has zero gradient at δ = 0 (see attack.AttackConfig).
PRESETS: dict[Strength, Preset] = {
    "low": Preset(rfgsm_config(4 / 255), eot=False),
    "medium": Preset(
        AttackConfig(
            eps=8 / 255, alpha=1.5 / 255, steps=50, eot_samples=2, grid=2, random_start=0.25
        ),
        eot=True,
    ),
    "high": Preset(
        AttackConfig(
            eps=12 / 255, alpha=1.5 / 255, steps=100, eot_samples=4, grid=2, random_start=0.25
        ),
        eot=True,
    ),
}


@dataclass
class Group:
    frames: Tensor  # (N, 3, H, W) crops of the watermarked frames, all the same rect
    logo: Rect  # mark position inside the crop
    id: str = ""
    view: View | None = None  # where the crop sits in the full frame, if the consumer says


@dataclass
class Result:
    id: str
    delta: Tensor  # (1, 3, H, W), on whole 8-bit levels, |δ| ≤ eps
    eps: float
    ms: float
    trace: list[float] = field(default_factory=list)  # total loss per step
    terms: list[dict[str, float]] = field(default_factory=list)  # per surrogate, per step


def vaccinate(
    group: Group,
    surrogate: Surrogate,
    strength: Strength = "medium",
    feather_px: int | None = None,
    seed: int = 0,
) -> Result:
    preset = PRESETS[strength]
    start = time.perf_counter()
    x = group.frames.to(device())
    _, _, h, w = x.shape
    logo = group.logo.clipped(w, h)
    # Default ramp: a quarter of the ring between the logo and the crop edge.
    ring = min(logo.x, logo.y, w - logo.x - logo.w, h - logo.y - logo.h)
    region = feather(h, w, feather_px if feather_px is not None else max(0, ring // 4)).to(x)

    if getattr(surrogate, "zero", False):  # the null surrogate: δ ≡ 0, skip the noise start too
        return Result(group.id, torch.zeros((1, 3, h, w)), preset.attack.eps, 0.0)
    loss_fn = surrogate.bind(Context(clean=x, logo=logo, view=group.view))
    transform = CodecProxy(seed=seed) if preset.eot else None
    trace: list[float] = []
    terms: list[dict[str, float]] = []

    def on_step(_: int, value: float) -> None:
        trace.append(value)
        terms.append(dict(getattr(surrogate, "last", {})))

    gen = torch.Generator(device=x.device).manual_seed(seed)
    delta = pgd(x, loss_fn, region, preset.attack, transform, gen, on_step=on_step)
    if x.is_cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000
    return Result(group.id, delta.cpu(), preset.attack.eps, ms, trace, terms)
