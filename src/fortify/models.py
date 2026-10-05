"""Model loaders shared by the surrogates (attack side) and eval/ (removal side).

All models are frozen: we only ever need gradients with respect to the input image.
"""

from __future__ import annotations

import os
import urllib.request
from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

# Apache-2.0. The TorchScript export IOPaint and WatermarkRemover-AI load.
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"

FLORENCE2_ID = os.environ.get("FORTIFY_FLORENCE2", "florence-community/Florence-2-large")
SAM_ID = os.environ.get("FORTIFY_SAM", "facebook/sam2.1-hiera-large")


def device() -> torch.device:
    return torch.device(
        os.environ.get("FORTIFY_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    )


def weights_dir() -> Path:
    d = Path(os.environ.get("FORTIFY_WEIGHTS", Path(__file__).resolve().parents[2] / "weights"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@cache
def load_lama(dev: str | None = None) -> torch.jit.ScriptModule:
    path = weights_dir() / "big-lama.pt"
    if not path.exists():
        print(f"downloading LaMa → {path}")
        urllib.request.urlretrieve(LAMA_URL, path)
    return _freeze(torch.jit.load(str(path), map_location=dev or device()))


def lama_inpaint(model: torch.jit.ScriptModule, image: Tensor, mask: Tensor) -> Tensor:
    """image (N, 3, H, W) in [0, 1], mask (N|1, 1, H, W) with 1 = fill. Differentiable in `image`."""
    n, _, h, w = image.shape
    mask = (mask > 0.5).to(image).expand(n, 1, h, w)
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        image = F.pad(image, (0, pw, 0, ph), mode="reflect")
        mask = F.pad(mask, (0, pw, 0, ph), mode="reflect")
    out = model(image, mask)
    return out[..., :h, :w].clamp(0, 1)


@cache
def load_florence2(model_id: str = FLORENCE2_ID, dev: str | None = None):
    # Native transformers port (5.x). microsoft/Florence-2-* remote code breaks on transformers 5.
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_id)
    return _freeze(model.to(dev or device())), processor


@cache
def load_sam(model_id: str = SAM_ID, dev: str | None = None):
    if "sam2" in model_id:
        from transformers import Sam2Model as Model
        from transformers import Sam2Processor as Processor
    else:
        from transformers import SamModel as Model
        from transformers import SamProcessor as Processor
    model = Model.from_pretrained(model_id)
    return _freeze(model.to(dev or device())), Processor.from_pretrained(model_id)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(x: Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s
