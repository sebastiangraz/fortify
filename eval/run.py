"""Red-team a set of frames: run real removers on marked (and shielded) frames after a codec.

Usage:
  uv run python eval/run.py --data eval/data [--shield medium] [--codec h264:28] [--removers oracle-lama,florence-lama]

Data layout (one directory per case; frames exported from videotools, see eval/README.md):
  eval/data/<case>/clean.png    the frame without the mark
  eval/data/<case>/marked.png   the same frame from Mark
  eval/data/<case>/logo.json    {"x":..,"y":..,"w":..,"h":..} mark rect in frame px

Writes runs/eval-<time>/results.csv plus the restored images for a look.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from removers import REMOVERS

from fortify.attack import apply_delta
from fortify.imageio import load_png, save_png, to_image, to_tensor
from fortify.models import device
from fortify.region import Rect, crop_around
from fortify.surrogates import ensemble
from fortify.vaccinate import Group, vaccinate


def codec(x: torch.Tensor, spec: str) -> torch.Tensor:
    """none | jpeg:<q> | h264:<crf> (real x264 via ffmpeg, yuv420p, like Mark's mp4 output)."""
    kind, _, arg = spec.partition(":")
    if kind == "none":
        return x
    if kind == "jpeg":
        buf = io.BytesIO()
        to_image(x).save(buf, format="JPEG", quality=int(arg or 75), subsampling=2)
        return to_tensor(Image.open(buf)).to(x)
    if kind == "h264":
        ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
        if not ff:
            raise SystemExit(
                "h264 codec needs ffmpeg: set FFMPEG (e.g. videotools/api/_bin/ffmpeg/win32-x64/ffmpeg.exe)"
            )
        with tempfile.TemporaryDirectory() as d:
            src, mp4, out = Path(d, "in.png"), Path(d, "x.mp4"), Path(d, "out.png")
            src.write_bytes(save_png(x))
            run = lambda *a: subprocess.run([ff, "-v", "error", "-y", *a], check=True)
            run(
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                arg or "23",
                "-pix_fmt",
                "yuv420p",
                str(mp4),
            )
            run("-i", str(mp4), "-frames:v", "1", str(out))
            return load_png(out.read_bytes()).to(x)
    raise SystemExit(f"unknown codec {spec}")


def region_mse(a: torch.Tensor, b: torch.Tensor, r: Rect) -> float:
    x0, y0, x1, y1 = r.xyxy()
    return float(((a[..., y0:y1, x0:x1] - b[..., y0:y1, x0:x1]) ** 2).mean())


def mark_residual(
    restored: torch.Tensor, marked: torch.Tensor, clean: torch.Tensor, r: Rect
) -> float:
    """Share of the mark's own signal left in `restored`: its projection on (marked − clean).

    1 = mark untouched, 0 = gone. Unlike removal_score it ignores inpainting error that is
    uncorrelated with the mark, so a faint mark over texture that a remover wipes out
    reads ≈ 0 even though the fill doesn't match clean pixel for pixel.
    """
    h, w = clean.shape[-2:]
    x0, y0, x1, y1 = Rect(r.x - 8, r.y - 8, r.w + 16, r.h + 16).clipped(w, h).xyxy()
    mark = (marked - clean)[..., y0:y1, x0:x1]
    left = (restored - clean)[..., y0:y1, x0:x1]
    energy = float((mark * mark).sum())
    return float((left * mark).sum()) / energy if energy > 0 else 0.0


def psnr(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1 / mse)


def shield(
    marked: torch.Tensor, logo: Rect, strength: str, surrogates: str
) -> tuple[torch.Tensor, float]:
    h, w = marked.shape[-2:]
    crop = crop_around(logo, w, h)
    x0, y0, x1, y1 = crop.xyxy()
    inner = Rect(logo.x - crop.x, logo.y - crop.y, logo.w, logo.h)
    r = vaccinate(
        Group(marked[..., y0:y1, x0:x1].cpu(), inner, "eval"), ensemble(surrogates), strength
    )
    full = torch.zeros_like(marked)
    full[..., y0:y1, x0:x1] = r.delta.to(marked)
    return apply_delta(marked, full), r.ms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="eval/data")
    p.add_argument("--shield", choices=("low", "medium", "high"), help="also test shielded frames")
    p.add_argument("--surrogates", default="lama,sam,florence2", help="ensemble used for shielding")
    p.add_argument(
        "--codec", default="h264:23", help="none | jpeg:<q> | h264:<crf>, applied before removal"
    )
    p.add_argument("--removers", default=",".join(REMOVERS))
    p.add_argument("--out", default=f"runs/eval-{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args()

    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = sorted(d for d in Path(args.data).iterdir() if (d / "marked.png").exists())
    if not cases:
        raise SystemExit(f"no cases under {args.data} (see eval/README.md)")
    for case in cases:
        clean = load_png((case / "clean.png").read_bytes()).to(dev)
        marked = load_png((case / "marked.png").read_bytes()).to(dev)
        logo = Rect(**json.loads((case / "logo.json").read_text()))
        variants = {"marked": (marked, 0.0)}
        if args.shield:
            variants[f"shield-{args.shield}"] = shield(marked, logo, args.shield, args.surrogates)
        for vname, (frame, ms) in variants.items():
            sent = codec(frame, args.codec)
            base = region_mse(
                sent, clean, logo
            )  # how far the delivered frame is from clean under the logo
            for rname in args.removers.split(","):
                restored, info = REMOVERS[rname](sent, logo)
                err = region_mse(restored, clean, logo)
                row = {
                    "case": case.name,
                    "variant": vname,
                    "codec": args.codec,
                    "remover": rname,
                    # 1 = perfect removal, 0 = no better than leaving the mark, < 0 = made it worse.
                    "removal_score": round(1 - err / base, 4) if base > 0 else 0.0,
                    "mark_residual": round(mark_residual(restored, marked, clean, logo), 4),
                    "removed_psnr_logo": round(psnr(err), 2),
                    "shield_cost_psnr": round(psnr(float(((frame - marked) ** 2).mean())), 2),
                    "shield_ms": round(ms),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()},
                }
                rows.append(row)
                (out / f"{case.name}_{vname}_{rname}.png").write_bytes(save_png(restored))
                print(row)
    keys = sorted(
        {k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99
    )
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"→ {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
