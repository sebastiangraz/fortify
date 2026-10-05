import torch

from fortify.attack import AttackConfig, apply_delta, fgsm_config, pgd, rfgsm_config
from fortify.region import Rect, feather
from fortify.surrogates import Context, ensemble
from fortify.vaccinate import Group, vaccinate


def _frames(n=2, h=64, w=96):
    return torch.rand((n, 3, h, w), generator=torch.Generator().manual_seed(1))


def test_pgd_respects_eps_region_and_levels():
    x = _frames()
    region = torch.zeros((1, 1, 64, 96))
    region[..., 16:48, 24:72] = 1
    loss = ensemble("toy").bind(Context(x, Rect(32, 20, 32, 24)))
    cfg = AttackConfig(eps=6 / 255, alpha=2 / 255, steps=10, random_start=0.5)
    delta = pgd(x, loss, region, cfg)
    assert delta.shape == (1, 3, 64, 96)
    assert float(delta.abs().max()) <= 6 / 255 + 1e-6
    assert float(delta[..., :16, :].abs().max()) == 0  # outside region untouched
    levels = delta * 255
    assert torch.allclose(levels, levels.round(), atol=1e-4)  # whole 8-bit levels


def test_pgd_lowers_the_loss():
    x = _frames()
    loss = ensemble("toy").bind(Context(x, Rect(32, 20, 32, 24)))
    region = torch.ones((1, 1, 64, 96))
    before = float(loss(x))
    delta = pgd(
        x, loss, region, AttackConfig(eps=8 / 255, alpha=2 / 255, steps=20, random_start=0.25)
    )
    assert float(loss(apply_delta(x, delta))) < before


def test_fgsm_is_one_full_step():
    cfg = fgsm_config(4 / 255)
    assert cfg.steps == 1 and cfg.alpha == cfg.eps


def test_disrupting_loss_needs_a_random_start():
    # Distance-to-clean losses have zero gradient at δ = 0: a zero start never moves.
    x = _frames()
    loss = ensemble("toy").bind(Context(x, Rect(32, 20, 32, 24)))
    region = torch.ones((1, 1, 64, 96))
    assert float(pgd(x, loss, region, fgsm_config(4 / 255)).abs().max()) == 0
    moved = pgd(x, loss, region, rfgsm_config(4 / 255))
    assert float(loss(apply_delta(x, moved))) < float(loss(x))


def test_grid_makes_delta_smooth():
    x = _frames()
    loss = ensemble("toy").bind(Context(x, Rect(32, 20, 32, 24)))
    region = torch.ones((1, 1, 64, 96))
    coarse = pgd(x, loss, region, AttackConfig(eps=8 / 255, steps=5, grid=4, random_start=0.5))
    fine = pgd(x, loss, region, AttackConfig(eps=8 / 255, steps=5, grid=1, random_start=0.5))

    def roughness(d):
        return float((d[..., 1:] - d[..., :-1]).abs().mean())

    assert roughness(coarse) < roughness(fine)


def test_null_surrogate_gives_zero_delta():
    r = vaccinate(Group(_frames(), Rect(32, 20, 32, 24)), ensemble("null"), "high")
    assert float(r.delta.abs().max()) == 0


def test_feather_ramps_to_border():
    f = feather(10, 20, 4)
    assert float(f[0, 0, 5, 10]) == 1
    assert float(f[0, 0, 0, 0]) < 0.1
