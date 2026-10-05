import io

import numpy as np
import torch
from PIL import Image

from fortify.eot import CodecProxy, jpeg_proxy, rgb_to_ycbcr, ycbcr_to_rgb
from fortify.imageio import to_image, to_tensor


def _image(h=48, w=72):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return torch.stack([xx, yy, (xx + yy) / 2])[None]


def test_ycbcr_roundtrip():
    x = _image()
    assert torch.allclose(ycbcr_to_rgb(rgb_to_ycbcr(x)), x, atol=1e-5)


def test_jpeg_proxy_tracks_real_jpeg():
    x = (
        _image() * 0.8
        + torch.rand((1, 3, 48, 72), generator=torch.Generator().manual_seed(0)) * 0.2
    )
    buf = io.BytesIO()
    to_image(x).save(buf, format="JPEG", quality=60, subsampling=2)
    real = to_tensor(Image.open(buf))
    proxy = jpeg_proxy(x, 60)
    # Not bit-exact (libjpeg rounds and upsamples differently), but much closer to
    # real JPEG than the input is.
    assert float((proxy - real).abs().mean()) < float((x - real).abs().mean())


def test_proxy_is_differentiable():
    x = _image().requires_grad_(True)
    out = CodecProxy(seed=0, p_jpeg=1, p_resize=1, p_blur=1, p_noise=1)(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0


def test_proxy_keeps_shape():
    x = torch.rand((2, 3, 37, 53))
    for seed in range(5):
        assert CodecProxy(seed=seed)(x).shape == x.shape
    assert np.isfinite(jpeg_proxy(x, 30).numpy()).all()
