"""HTTP API. See README → "API contract" (the TS client in packages/client mirrors it).

Env:
  FORTIFY_TOKEN       bearer token required on /v1/* (unset = no auth, dev only)
  FORTIFY_SURROGATES  ensemble spec, default "lama,sam,florence2"
  FORTIFY_DEVICE      cuda / cpu override
"""

from __future__ import annotations

import base64
import hmac
import os
import threading
import time
from typing import Literal

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import DELTA_OFFSET, __version__
from .imageio import encode_delta, load_png
from .models import device
from .region import Rect
from .surrogates import View, ensemble
from .vaccinate import Group, vaccinate

API_VERSION = 1
MAX_GROUPS = 64
MAX_FRAMES = 8
MAX_PIXELS = 1024 * 1024

SURROGATES = os.environ.get("FORTIFY_SURROGATES", "lama,sam,florence2")


class RectIn(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class SizeIn(BaseModel):
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class PointIn(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class ViewIn(BaseModel):
    frame: SizeIn  # full frame size
    at: PointIn  # the crop's top-left corner in the frame
    background: str | None = None  # base64 PNG of the whole frame, any size (a thumbnail)


class GroupIn(BaseModel):
    id: str = Field(max_length=64)
    logo: RectIn
    frames: list[str] = Field(min_length=1, max_length=MAX_FRAMES)  # base64 PNG crops
    view: ViewIn | None = None


class VaccinateIn(BaseModel):
    version: Literal[1] = 1
    strength: Literal["low", "medium", "high"] = "medium"
    groups: list[GroupIn] = Field(min_length=1, max_length=MAX_GROUPS)


class GroupOut(BaseModel):
    id: str
    delta: str  # base64 RGB PNG, uint8 = offset + δ·255
    eps: int  # |δ| bound in 8-bit levels
    ms: float


class VaccinateOut(BaseModel):
    version: int = API_VERSION
    offset: int = DELTA_OFFSET
    surrogates: str
    groups: list[GroupOut]
    ms: float


app = FastAPI(title="fortify", version=__version__)
_gpu = threading.Lock()  # one GPU, one attack at a time
_ensemble = None


def _model():
    global _ensemble
    if _ensemble is None:
        _ensemble = ensemble(SURROGATES)
    return _ensemble


def _auth(authorization: str | None) -> None:
    token = os.environ.get("FORTIFY_TOKEN")
    if token and not hmac.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(401, "bad token")


def _png(group: GroupIn, data: str, what: str) -> torch.Tensor:
    try:
        return load_png(base64.b64decode(data, validate=True))
    except (ValueError, OSError) as e:  # bad base64 (binascii.Error) or not an image (PIL)
        raise HTTPException(400, f"group {group.id}: unreadable {what} ({e})") from None


def _decode(group: GroupIn) -> Group:
    frames = [_png(group, f, "frame") for f in group.frames]
    sizes = {tuple(f.shape[-2:]) for f in frames}
    if len(sizes) != 1:
        raise HTTPException(400, f"group {group.id}: frames differ in size {sizes}")
    (h, w) = sizes.pop()
    if h * w > MAX_PIXELS:
        raise HTTPException(413, f"group {group.id}: crop {w}x{h} is over {MAX_PIXELS} px")
    logo = Rect(**group.logo.model_dump())
    if logo.x + logo.w > w or logo.y + logo.h > h:
        raise HTTPException(400, f"group {group.id}: logo rect is outside the {w}x{h} crop")
    return Group(torch.cat(frames), logo, group.id, _view(group, w, h))


def _view(group: GroupIn, w: int, h: int) -> View | None:
    v = group.view
    if v is None:
        return None
    if v.at.x + w > v.frame.w or v.at.y + h > v.frame.h:
        raise HTTPException(400, f"group {group.id}: the crop is outside the view's frame")
    bg = None
    if v.background is not None:
        bg = _png(group, v.background, "view background")
        if bg.shape[-2] * bg.shape[-1] > MAX_PIXELS:
            raise HTTPException(413, f"group {group.id}: view background is over {MAX_PIXELS} px")
    return View((v.frame.w, v.frame.h), (v.at.x, v.at.y), bg)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "version": __version__,
        "api": API_VERSION,
        "device": str(device()),
        "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "surrogates": SURROGATES,
        "loaded": _ensemble is not None,
    }


@app.post("/v1/vaccinate")
def vaccinate_route(
    body: VaccinateIn, authorization: str | None = Header(default=None)
) -> VaccinateOut:
    _auth(authorization)
    groups = [_decode(g) for g in body.groups]
    start = time.perf_counter()
    out = []
    with _gpu:
        model = _model()
        for g in groups:
            r = vaccinate(g, model, body.strength)
            out.append(
                GroupOut(
                    id=r.id,
                    delta=base64.b64encode(encode_delta(r.delta)).decode(),
                    eps=round(r.eps * 255),
                    ms=r.ms,
                )
            )
    return VaccinateOut(surrogates=SURROGATES, groups=out, ms=(time.perf_counter() - start) * 1000)
