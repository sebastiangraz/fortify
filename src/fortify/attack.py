"""FGSM / PGD with a region mask, a low-frequency grid and EOT.

Conventions (shared by every module):
- Images are float tensors in [0, 1], shape (N, 3, H, W).
- A loss function takes the perturbed batch and returns a scalar to MINIMISE.
  Surrogates pick the sign (e.g. "ascend the remover's error" = return -error).
- One δ of shape (1, 3, H, W) is shared by all N frames: a universal perturbation
  for a group of frames (in Mark: one rotation stay, where the mark sits still).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

LossFn = Callable[[Tensor], Tensor]
Transform = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class AttackConfig:
    eps: float = 8 / 255  # L∞ budget on δ
    alpha: float = 1.5 / 255  # step size per iteration
    steps: int = 50  # 1 step with alpha == eps is FGSM
    eot_samples: int = 1  # random transforms averaged per step
    grid: int = 1  # δ lives on an (H/grid, W/grid) grid, upsampled bilinearly: low-pass by design
    # Start from uniform noise in ±random_start·eps. Needed for "disrupting" losses
    # (distance to the clean output), whose gradient is exactly 0 at δ = 0.
    random_start: float = 0.0


def fgsm_config(eps: float) -> AttackConfig:
    """Plain FGSM: one full step from δ = 0."""
    return AttackConfig(eps=eps, alpha=eps, steps=1)


def rfgsm_config(eps: float) -> AttackConfig:
    """R+FGSM (Tramèr et al. 2018): a random half-ε start, then one half-ε gradient step."""
    return AttackConfig(eps=eps, alpha=eps / 2, steps=1, random_start=0.5)


def upsample(theta: Tensor, size: tuple[int, int]) -> Tensor:
    if theta.shape[-2:] == size:
        return theta
    # Bilinear interpolation is a convex combination, so |δ| ≤ eps still holds.
    return F.interpolate(theta, size=size, mode="bilinear", align_corners=False)


def quantize(delta: Tensor) -> Tensor:
    """Snap δ to whole 8-bit levels: the output video is 8-bit and the wire format is too."""
    return torch.round(delta * 255) / 255


def pgd(
    x: Tensor,
    loss_fn: LossFn,
    region: Tensor,
    cfg: AttackConfig,
    transform: Transform | None = None,
    generator: torch.Generator | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> Tensor:
    """Return a quantized δ (1, 3, H, W), already multiplied by `region`.

    x: (N, 3, H, W) frames sharing δ. region: (1, 1, H, W) weight in [0, 1]; 0 = never touch.
    transform: random differentiable transform applied after δ (EOT), e.g. a codec proxy.
    """
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"x must be (N, 3, H, W), got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    gh, gw = max(1, round(h / cfg.grid)), max(1, round(w / cfg.grid))
    x = x.detach()
    region = region.to(x)

    if cfg.random_start > 0:
        noise = torch.rand((1, 3, gh, gw), generator=generator, device=x.device) * 2 - 1
        theta = (noise * cfg.random_start * cfg.eps).to(x.dtype)
    else:
        theta = torch.zeros((1, 3, gh, gw), device=x.device, dtype=x.dtype)

    for step in range(cfg.steps):
        theta.requires_grad_(True)
        delta = upsample(theta, (h, w)) * region
        total = x.new_zeros(())
        for _ in range(cfg.eot_samples):
            adv = (x + delta).clamp(0, 1)
            if transform is not None:
                adv = transform(adv)
            total = total + loss_fn(adv)
        total = total / cfg.eot_samples
        (grad,) = torch.autograd.grad(total, theta)
        with torch.no_grad():
            theta = (theta - cfg.alpha * grad.sign()).clamp(-cfg.eps, cfg.eps)
        if on_step is not None:
            on_step(step, float(total.detach()))

    with torch.no_grad():
        delta = upsample(theta.detach(), (h, w)) * region
        return quantize(delta).clamp(-cfg.eps, cfg.eps)


def apply_delta(x: Tensor, delta: Tensor) -> Tensor:
    """What the consumer does on its side: add δ and clip to the valid range."""
    return (x + delta).clamp(0, 1)
