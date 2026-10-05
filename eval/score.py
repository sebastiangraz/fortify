"""Score frames restored by hand-run tools (WatermarkRemover-AI, IOPaint, ProPainter...).

Usage:
  uv run python eval/score.py --data eval/data --restored runs/p0-wild/wrai-box [--label wrai-box]

`--restored` holds one image per case, named `<case>.png` (anything PIL reads, same size as the
case). Rows have the same metric columns as run.py's; they are appended to
`<restored>/../scores.csv` so several tools end up in one table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from run import mark_residual, psnr, region_mse

from fortify.imageio import load_png, to_tensor
from fortify.region import Rect


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="eval/data")
    p.add_argument("--restored", required=True)
    p.add_argument("--sent", help="frames the tool was given (default: <restored>/../in)")
    p.add_argument("--label", help="remover name in the table (default: the folder name)")
    args = p.parse_args()

    restored_dir = Path(args.restored)
    sent_dir = Path(args.sent) if args.sent else restored_dir.parent / "in"
    label = args.label or restored_dir.name
    rows = []
    for case in sorted(d for d in Path(args.data).iterdir() if (d / "marked.png").exists()):
        found = sorted(restored_dir.glob(f"{case.name}.*"))
        if not found:
            print(f"skip {case.name}: no restored frame")
            continue
        clean = load_png((case / "clean.png").read_bytes())
        marked = load_png((case / "marked.png").read_bytes())
        restored = to_tensor(Image.open(found[0]).convert("RGB"))
        sent = to_tensor(Image.open(sent_dir / f"{case.name}.png").convert("RGB"))
        logo = Rect(**json.loads((case / "logo.json").read_text()))
        if restored.shape != clean.shape:
            raise SystemExit(f"{found[0]}: {tuple(restored.shape)} ≠ case {tuple(clean.shape)}")
        base, err = region_mse(sent, clean, logo), region_mse(restored, clean, logo)
        changed = (restored - sent).abs().amax(1, keepdim=True) > 4 / 255
        x0, y0, x1, y1 = logo.xyxy()
        row = {
            "case": case.name,
            "remover": label,
            "removal_score": round(1 - err / base, 4) if base > 0 else 0.0,
            "mark_residual": round(mark_residual(restored, marked, clean, logo), 4),
            "removed_psnr_logo": round(psnr(err), 2),
            # Share of the logo rect the tool touched: a proxy for its mask's coverage.
            "logo_covered": round(float(changed[..., y0:y1, x0:x1].float().mean()), 4),
            "changed_px": int(changed.sum()),
        }
        rows.append(row)
        print(row)
    out = restored_dir.parent / "scores.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
