"""Calibrate fortify.eot's JPEG proxy against real x264: which quality range erases δ as much
as the CRFs Mark ships at?

Usage:
  uv run python eval/calibrate_codec.py [--data eval/data] [--crf 18,23,28] [--clips]

Survival of a δ through a codec c, on a frame x, inside the crop:
    s = <c(x + δ) − c(x), δ> / <δ, δ>
1 = δ arrives intact, 0 = erased. Measured for test δs shaped like the attack's: random ±8
levels on an (H/grid × W/grid) grid, upsampled bilinearly, on whole levels, feathered, at
each case's crop (`crop_around` the logo). PGD with sign steps saturates δ at ±ε much like
this. Codecs:
- x264 at each CRF with Mark's settings (`-preset fast`, yuv420p), on the single frame
  (an I-frame, as eval/run.py's `h264:<crf>` does);
- with --clips, x264 over 48 frames of each clean clip with the same δ on every frame,
  scored on all of them (P/B-frames predict δ from the frame before, so they may keep less
  or more of it than an I-frame);
- the proxy's JPEG step alone (`jpeg_proxy`, deterministic) at qualities 20..95;
- PIL JPEG at the same qualities, as a check that the proxy behaves like a real JPEG.
Then for each CRF, the proxy quality whose survival is closest, per grid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from run import codec

from fortify.attack import quantize
from fortify.eot import jpeg_proxy
from fortify.imageio import load_png
from fortify.region import Rect, crop_around, feather

EPS = 8 / 255
QUALITIES = (20, 30, 40, 50, 60, 70, 80, 90, 95)
GRIDS = (1, 2, 4)
CLIP_FRAMES = 48


def test_delta(h: int, w: int, grid: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    gh, gw = max(1, round(h / grid)), max(1, round(w / grid))
    theta = (torch.randint(0, 2, (1, 3, gh, gw), generator=g).float() * 2 - 1) * EPS
    up = F.interpolate(theta, size=(h, w), mode="bilinear", align_corners=False)
    return quantize(up * feather(h, w, max(8, min(h, w) // 16)))


def survival(before: torch.Tensor, after: torch.Tensor, delta: torch.Tensor) -> float:
    return float(((after - before) * delta).sum() / (delta * delta).sum())


def place(frame: torch.Tensor, delta: torch.Tensor, crop: Rect) -> torch.Tensor:
    out = frame.clone()
    x0, y0, x1, y1 = crop.xyxy()
    out[..., y0:y1, x0:x1] = (out[..., y0:y1, x0:x1] + delta).clamp(0, 1)
    return out


def cut(frame: torch.Tensor, crop: Rect) -> torch.Tensor:
    x0, y0, x1, y1 = crop.xyxy()
    return frame[..., y0:y1, x0:x1]


def ffmpeg() -> str:
    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ff:
        raise SystemExit(
            "needs ffmpeg: set FFMPEG to videotools' api/_bin/ffmpeg/win32-x64/ffmpeg.exe"
        )
    return ff


def read_clip(path: Path, n: int) -> torch.Tensor:
    """First n frames as (n, 3, H, W) in [0, 1]."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(
            [ffmpeg(), "-v", "error", "-i", str(path), "-frames:v", str(n), f"{d}/%03d.png"],
            check=True,
        )
        return torch.cat([load_png(p.read_bytes()) for p in sorted(Path(d).glob("*.png"))])


def x264_clip(frames: torch.Tensor, crf: int) -> torch.Tensor:
    """Encode (n, 3, H, W) as one x264 stream the way Mark does, decode it back."""
    n, _, h, w = frames.shape
    raw = (frames.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).contiguous().numpy()
    with tempfile.TemporaryDirectory() as d:
        mp4 = Path(d, "x.mp4")
        subprocess.run(
            [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-preset", "fast",
             "-crf", str(crf), "-pix_fmt", "yuv420p", str(mp4)],
            input=raw.tobytes(), check=True,
        )  # fmt: skip
        out = subprocess.run(
            [ffmpeg(), "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
    arr = np.frombuffer(out, dtype=np.uint8).reshape(n, h, w, 3)
    return torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float() / 255


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="eval/data")
    p.add_argument("--crf", default="18,23,28")
    p.add_argument("--clips", action="store_true", help="also measure on clips (P/B-frames)")
    args = p.parse_args()
    crfs = [int(c) for c in args.crf.split(",")]

    cases = sorted(d for d in Path(args.data).iterdir() if (d / "marked.png").exists())
    # results[codec][grid] = list of survivals over cases
    results: dict[str, dict[int, list[float]]] = {}

    def add(name: str, grid: int, value: float) -> None:
        results.setdefault(name, {}).setdefault(grid, []).append(value)

    for i, case in enumerate(cases):
        x = load_png((case / "marked.png").read_bytes())
        logo = Rect(**json.loads((case / "logo.json").read_text()))
        crop = crop_around(logo, x.shape[-1], x.shape[-2])
        for grid in GRIDS:
            delta = test_delta(crop.h, crop.w, grid, seed=i)
            xd = place(x, delta, crop)
            for crf in crfs:
                before, after = codec(x, f"h264:{crf}"), codec(xd, f"h264:{crf}")
                add(
                    f"x264 crf {crf} (frame)",
                    grid,
                    survival(cut(before, crop), cut(after, crop), delta),
                )
            xc, xdc = cut(x, crop), cut(xd, crop)
            for q in QUALITIES:
                add(f"proxy q{q}", grid, survival(jpeg_proxy(xc, q), jpeg_proxy(xdc, q), delta))
                add(
                    f"pil q{q}",
                    grid,
                    survival(codec(xc, f"jpeg:{q}"), codec(xdc, f"jpeg:{q}"), delta),
                )
        print(f"{case.name}: done", flush=True)

    if args.clips:
        for clip in sorted(Path(args.data, "_clips").glob("*-clean.mp4")):
            frames = read_clip(clip, CLIP_FRAMES)
            _, _, h, w = frames.shape
            logo = Rect(w - w // 5 - 40, h - h // 8 - 40, w // 5, h // 8)  # a bottom-right mark
            crop = crop_around(logo, w, h)
            for grid in GRIDS:
                delta = test_delta(crop.h, crop.w, grid, seed=100 + grid)
                shielded = torch.cat([place(f[None], delta, crop) for f in frames])
                for crf in crfs:
                    before, after = x264_clip(frames, crf), x264_clip(shielded, crf)
                    s = [
                        survival(cut(b[None], crop), cut(a[None], crop), delta)
                        for b, a in zip(before, after)
                    ]
                    for v in s:
                        add(f"x264 crf {crf} (clip)", grid, v)
            print(f"{clip.name}: done", flush=True)

    mean = {name: {g: sum(v) / len(v) for g, v in by.items()} for name, by in results.items()}
    print(f"\nsurvival of a ±8-level δ (mean over cases), by grid {GRIDS}")
    for name, by in mean.items():
        print(f"  {name:<24} " + "  ".join(f"g{g} {by[g]:.3f}" for g in GRIDS))
    print("\nclosest proxy quality per x264 setting:")
    for name, by in mean.items():
        if not name.startswith("x264"):
            continue
        best = []
        for g in GRIDS:
            q = min(QUALITIES, key=lambda q: abs(mean[f"proxy q{q}"][g] - by[g]))
            best.append(f"g{g} q{q}")
        print(f"  {name:<24} " + "  ".join(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
