"""FGSM / PGD with a region mask, a low-frequency grid and EOT.

Conventions (shared by every module):
- Images are float tensors in [0, 1], shape (N, 3, H, W).
- A loss function takes the perturbed batch and returns a scalar to MINIMISE.
  Surrogates pick the sign (e.g. "ascend the remover's error" = return -error).
- One δ of shape (1, 3, H, W) is shared by all N frames: a universal perturbation
  for a group of frames (in Mark: one rotation stay, where the mark sits still).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
    # MI-FGSM (Dong et al. 2018): step on the sign of a running sum of L1-normalised
    # gradients, decayed by this factor. When each step sees a different EOT draw (a codec,
    # an attacker mask), plain sign steps chase the latest draw and oscillate; momentum
    # steps towards what most draws agree on. 0 = plain PGD.
    momentum: float = 0.0


class SumLoss:
    """A weighted sum of named parts, e.g. one per surrogate.

    pgd backpropagates each part on its own (value_and_grad), so peak memory is the largest
    part's graph instead of all of them at once. With `balance`, each part's gradient is
    scaled to unit mean |g| before weighting, so the weights set each part's share of the
    step whatever its loss scale (LaMa MSE ~0.01 vs Florence-2 CE ~10 vs SAM ClipMSE ~100).
    """

    def __init__(
        self,
        parts: Iterable[LossFn],
        weights: Iterable[float] | None = None,
        names: Iterable[str] | None = None,
        balance: bool = False,
    ):
        self.parts = list(parts)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.parts)
        self.names = list(names) if names is not None else [str(i) for i in range(len(self.parts))]
        self.balance = balance
        self.last: dict[str, float] = {}  # each part's value at the latest call, for logging

    def __call__(self, x: Tensor) -> Tensor:
        total = x.new_zeros(())
        for name, weight, part in zip(self.names, self.weights, self.parts):
            term = part(x)
            self.last[name] = float(term.detach())
            total = total + weight * term
        return total


def value_and_grad(loss_fn: LossFn, x: Tensor) -> tuple[float, Tensor]:
    """The loss at x and its gradient with respect to x, one SumLoss part at a time."""
    return _value_and_grad(loss_fn, x.detach().requires_grad_(True))


def _value_and_grad(fn: LossFn, leaf: Tensor) -> tuple[float, Tensor]:
    if not isinstance(fn, SumLoss):
        v = fn(leaf)
        g = torch.autograd.grad(v, leaf)[0] if v.requires_grad else torch.zeros_like(leaf)
        return float(v.detach()), g
    value, grad = 0.0, torch.zeros_like(leaf)
    for name, weight, part in zip(fn.names, fn.weights, fn.parts):
        v, g = _value_and_grad(part, leaf)
        fn.last[name] = v
        if fn.balance:
            g = g / g.abs().mean().clamp(min=1e-12)
        value += weight * v
        grad += weight * g
    return value, grad


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

    velocity = torch.zeros_like(theta)
    for step in range(cfg.steps):
        theta.requires_grad_(True)
        grad, total = torch.zeros_like(theta), 0.0
        # One backward per EOT sample (and per SumLoss part), accumulated: memory stays at
        # one sample's graph however many samples, frames or surrogates there are.
        for _ in range(cfg.eot_samples):
            adv = (x + upsample(theta, (h, w)) * region).clamp(0, 1)
            if transform is not None:
                adv = transform(adv)
            value, g = value_and_grad(loss_fn, adv)
            grad += torch.autograd.grad(adv, theta, g)[0]
            total += value / cfg.eot_samples
        with torch.no_grad():
            if cfg.momentum:
                velocity = cfg.momentum * velocity + grad / grad.abs().mean().clamp(min=1e-12)
                grad = velocity
            theta = (theta - cfg.alpha * grad.sign()).clamp(-cfg.eps, cfg.eps)
        if on_step is not None:
            on_step(step, total)

    with torch.no_grad():
        delta = upsample(theta.detach(), (h, w)) * region
        return quantize(delta).clamp(-cfg.eps, cfg.eps)


def apply_delta(x: Tensor, delta: Tensor) -> Tensor:
    """What the consumer does on its side: add δ and clip to the valid range."""
    return (x + delta).clamp(0, 1)
