"""Frames and masks for the video removers (ProPainter etc.), from an exported clip pair.

Usage:
  uv run python eval/video_masks.py --marked eval/data/_clips/smoke-plain-rotating.mp4 \
      --clean eval/data/_clips/smoke-clean.mp4 --out runs/p0-video/smoke-plain-rotating

Writes under --out:
  frames/   the marked clip as PNGs (what the attacker downloaded)
  clean/    the clean clip as PNGs (for scoring)
  truth/    the lossless marked twin as PNGs (for scoring)
  oracle/   per-frame truth mask, |truth − clean| > 6 levels, dilated: a hand-drawn mask
  sam2/     the "click once" attacker: SAM2 video, one box on frame 0, propagated
  masks.json  per-frame share of the truth mask that sam2/ covers

--truth defaults to <marked>.truth.mp4, the lossless twin export_cases.ts writes. The
delivered CRF 23 clip differs from clean everywhere by codec noise, so the truth mask comes
from the twin; it only exists for exported clips.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DILATE = 8


def dump(ff: str, video: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-v", "error", "-y", "-i", str(video), str(out / "%05d.png")], check=True)
    return sorted(out.glob("*.png"))


def dilate(m: np.ndarray, px: int) -> np.ndarray:
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.max_pool2d(t, 2 * px + 1, stride=1, padding=px)[0, 0].numpy() > 0.5


def save(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def sam2_propagate(frames: list[Image.Image], box: list[float]) -> list[np.ndarray]:
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    from fortify.models import SAM_ID, device

    dev = device()
    model = Sam2VideoModel.from_pretrained(SAM_ID).to(dev, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(SAM_ID)
    session = processor.init_video_session(video=frames, inference_device=dev, dtype=torch.bfloat16)
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=0, obj_ids=1, input_boxes=[[box]]
    )
    model(inference_session=session, frame_idx=0)
    size = [[session.video_height, session.video_width]]
    masks: dict[int, np.ndarray] = {}
    for out in model.propagate_in_video_iterator(session):
        m = processor.post_process_masks([out.pred_masks], original_sizes=size, binarize=True)[0]
        masks[out.frame_idx] = m[0, 0].cpu().numpy().astype(bool)
    return [masks[i] for i in range(len(frames))]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--marked", required=True)
    p.add_argument("--clean", required=True)
    p.add_argument("--truth", help="lossless marked clip (default: <marked>.truth.mp4)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("needs ffmpeg: set FFMPEG")
    out = Path(args.out)
    marked = dump(ff, Path(args.marked), out / "frames")
    clean = dump(ff, Path(args.clean), out / "clean")
    lossless = dump(
        ff, Path(args.truth or Path(args.marked).with_suffix(".truth.mp4")), out / "truth"
    )
    (out / "oracle").mkdir(exist_ok=True)
    (out / "sam2").mkdir(exist_ok=True)

    truth = []
    for m, c, t in zip(marked, clean, lossless):
        a = np.asarray(Image.open(t).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(c).convert("RGB"), dtype=np.int16)
        mask = dilate(np.abs(a - b).max(-1) > 6, DILATE)
        truth.append(mask)
        save(mask, out / "oracle" / m.name)

    ys, xs = np.nonzero(truth[0])
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    frames = [Image.open(m).convert("RGB") for m in marked]
    covered = []
    for m, t, path in zip(sam2_propagate(frames, box), truth, marked):
        m = dilate(m, DILATE)
        save(m, out / "sam2" / path.name)
        covered.append(round(float((m & t).sum() / max(1, t.sum())), 3))
    (out / "masks.json").write_text(json.dumps({"box": box, "sam2_covers_truth": covered}))
    print(
        f"{out.name}: sam2 covers truth, per frame: min {min(covered)}, mean {np.mean(covered):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
