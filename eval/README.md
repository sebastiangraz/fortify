# eval/: red-team harness

The question this answers: **does a real removal pipeline still get rid of the mark?**
Ask it of plain Mark output first (Phase 0, the baseline), then of shielded output.

## Getting cases out of videotools

Each case is one frame, exported twice from videotools: with Mark and without. Keep the
mark's rect in frame pixels.

```
eval/data/<case>/clean.png     source frame
eval/data/<case>/marked.png    same frame through Mark (filter: plain | glass | blur)
eval/data/<case>/logo.json     {"x": 1580, "y": 900, "w": 260, "h": 120}
```

`export_cases.ts` does this for the Phase 0 set (24 cases, plus clips for the video removers).
It imports videotools' `watermarkGraph`/`watermarkLayout` directly and runs its pinned ffmpeg:

```
bun eval/export_cases.ts --clips          # needs ../videotools (or --videotools <dir>)
```

- **Sources:** videotools' smoke assets; the Sintel trailer (CC-BY, download.blender.org/durian/trailer,
  letterbox cropped to 1920×816 on first run); and Kodak PhotoCD images kodim04/05/08/13/15/19/20/23
  (r0k.us/graphics/kodak). Put the downloads in `eval/data/_sources/`.
- **One format chain for both:** clean and marked go through the same chain (yuv420p for video,
  RGBA for stills, as Mark does), then to RGB with one explicit matrix, so they differ only where
  Mark draws. The script checks this; the max outside the cell is 0.
- **`meta.json`** adds the glass/blur cell, filter, size, position and source.
- **Clips:** `eval/data/_clips/` holds `<clip>-clean.mp4`, a static and a rotating marked version
  (CRF 23, as delivered), and a lossless `.truth.mp4` twin of each for the truth masks.
- **Plain on video:** overlay onto yuv420p floors x/y to even, so an odd plain corner lands 1 px up
  and left of `watermarkLayout`'s. `logo.json` records where it actually is.

Cover a spread: each of plain, glass and blur; each of the size presets; flat vs textured
backgrounds (sky, grass, faces, text); a few positions. Names are `<clip>-<filter>-<size>-<pos>`.

## Running

```
uv run python eval/run.py --data eval/data --codec h264:23                    # Phase 0 baseline
uv run python eval/run.py --data eval/data --codec h264:23 --shield medium    # with fortify
uv run python eval/run.py --codec jpeg:75 --removers oracle-lama              # quick loop
```

`--codec` is applied *before* removal. It stands for what the attacker downloads (our
encode, maybe re-encoded by a platform). `h264:<crf>` uses real x264 and needs ffmpeg:
set `FFMPEG` to videotools' `api/_bin/ffmpeg/win32-x64/ffmpeg.exe`.

## Metrics (per case × variant × remover)

| column | meaning | want (shielded) |
|---|---|---|
| `removal_score` | 1 − MSE(restored, clean) / MSE(delivered, clean) under the logo. 1 = perfect removal, 0 = no better than leaving the mark, < 0 = made it worse | ≤ 0.3 |
| `mark_residual` | share of the mark's own signal left: projection of (restored − clean) on (marked − clean), logo + 8 px. 1 = untouched, 0 = gone | ≥ 0.5 |
| `removed_psnr_logo` | PSNR of the restored logo area vs clean | low |
| `logo_covered` | share of the logo rect the remover's mask covers (detection stages) | low |
| `boxes` / `mask_px` | detector output size | 0 / small |
| `shield_cost_psnr` | PSNR of shielded vs marked, whole frame (visual cost of δ) | ≥ 38 dB |

Always compare the `marked` and `shield-*` rows for the same case and remover: the
shield has worked when the score drops.

Read the two scores together, and look at the images:
- **`removal_score` under-reads faint marks.** On a blur or glass mark over texture, an inpainter
  that wipes the mark completely still misses clean pixel for pixel by about as much as the mark
  itself, so the score lands near 0–0.5. Phase 0 had a desert case at 0.23 with the mark visibly gone.
- **`mark_residual` ignores error that has nothing to do with the mark**, so it reads ≈ 0 there.
- **`mark_residual` over-reads blur marks over sharp texture.** The blur mark *is* (blurred
  − sharp) in the logo's shape, and any smooth fill correlates with that. Values ≈ 1 or above on
  `kodim08`/`kodim15` blur came from fills where no logo shape is left. Check the restored PNGs
  before trusting a blur row.

## Removers covered here vs by hand

In `removers.py` (frame-level, Apache/MIT models):
`oracle-lama` (hand-drawn mask, worst case), `florence-lama` (WatermarkRemover-AI style),
`sam-lama` (IOPaint style).

By hand, from their own repos, on short clips (keep out of the shipped service):
- WatermarkRemover-AI (https://github.com/D-Ogi/WatermarkRemover-AI): the actual tool, MIT.
- IOPaint (https://github.com/Sanster/IOPaint): try the MAT and SD inpainting models too. These are the held-out removers for checking transfer.
- SAM2 + ProPainter (https://github.com/sczhou/ProPainter): video inpainting, non-commercial licence, eval only.

Clone each into `eval/tools/` (gitignored) with its own uv venv, then reinstall
torch/torchvision from the cu128 index last, because their requirements pull CPU or old wheels.
- **IOPaint** also needs `huggingface_hub<0.26`, `transformers<4.46` and `numpy<2` (diffusers 0.27).
  `runwayml/stable-diffusion-inpainting` still resolves through HF's redirect.
- **Frames:** feed each tool the frames as delivered. Encode `marked.png` through x264 CRF 23 into
  `runs/<run>/in/<case>.png`, exactly what `--codec h264:23` does.
- **Masks:** for IOPaint, write the dilated logo box as `runs/<run>/mask/<case>.png`; that's the
  same mask as `oracle-lama`.
- **Scoring frames:**
  `uv run python eval/score.py --restored runs/<run>/<tool>` appends to `runs/<run>/scores.csv`.
- **Clips:** `video_masks.py` writes frames, a truth mask and a SAM2 "box once on frame 0,
  propagate" mask per clip. `score_clip.py` scores a restored clip (video or frames folder)
  frame by frame.

## Calibrating the codec proxy

`fortify.eot.CodecProxy` approximates x264 with JPEG-style quantization. To calibrate:
1. Pick a δ that was shielded without EOT (strength `low`).
2. Compare how much of it survives `--codec h264:23` vs `--codec jpeg:<q>`. Measure the PSNR between shielded and marked after the codec.
3. Set `CodecProxy.quality` to the JPEG range that matches the x264 CRFs you ship at.
