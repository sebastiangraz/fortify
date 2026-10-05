"""Score a clip restored by a video remover against the folders video_masks.py wrote.

Usage:
  uv run python eval/score_clip.py --dir runs/p0-video/smoke-plain-rotating \
      --restored runs/p0-video/wrai/smoke-plain-rotating.mp4 --label wrai

`--restored` is a video or a folder of frames in order. Per frame, mark_residual is the
mark's share left (see run.py), taken over the truth mask. One row per clip is appended to
`<dir>/../scores.csv`.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def frames(path: Path, tmp: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.png"))
    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("needs ffmpeg: set FFMPEG")
    subprocess.run([ff, "-v", "error", "-y", "-i", str(path), str(tmp / "%05d.png")], check=True)
    return sorted(tmp.glob("*.png"))


def load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--restored", required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    d = Path(args.dir)
    clean = sorted((d / "clean").glob("*.png"))
    truth = sorted((d / "truth").glob("*.png"))
    oracle = sorted((d / "oracle").glob("*.png"))
    with tempfile.TemporaryDirectory() as tmp:
        restored = frames(Path(args.restored), Path(tmp))
        if len(restored) != len(clean):
            print(
                f"note: {len(restored)} restored frames vs {len(clean)} clean; scoring the overlap"
            )
        res = []
        for c, t, o, r in zip(clean, truth, oracle, restored):
            c, t, r = load(c), load(t), load(r)
            if r.shape != c.shape:
                raise SystemExit(f"{r.shape} ≠ {c.shape}: restore at the clip's own size")
            m = np.asarray(Image.open(o)) > 127
            mark, left = (t - c)[m], (r - c)[m]
            energy = float((mark * mark).sum())
            res.append(float((left * mark).sum()) / energy if energy > 0 else 0.0)
    res = np.array(res)
    row = {
        "clip": d.name,
        "remover": args.label,
        "frames": len(res),
        "residual_median": round(float(np.median(res)), 4),
        "residual_p90": round(float(np.quantile(res, 0.9)), 4),
        # Frames where at least a fifth of the mark survived: it is still visible there.
        "frames_visible": round(float((res > 0.2).mean()), 4),
    }
    print(row)
    out = d.parent / "scores.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
