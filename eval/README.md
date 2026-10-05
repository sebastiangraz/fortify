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

Ways to produce them:
- Render one frame with ffmpeg, then run videotools' `renderWatermarkFrame` (api/_lib/tools/mark.ts)
  through a Bun scratch script. `watermarkLayout` (mark-graph.ts) gives the rect.
- Or export a frame with the Mark UI preview and the matching source frame at the same timestamp.
  Measure the rect by eye or from the layout code.

Cover a spread: each of plain, glass and blur; each of the size presets; flat vs textured
backgrounds (sky, grass, faces, text); a few rotation positions. About 20 cases is enough to start.
Name them `<clip>-<filter>-<size>-<pos>`.

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
| `removed_psnr_logo` | PSNR of the restored logo area vs clean | low |
| `logo_covered` | share of the logo rect the remover's mask covers (detection stages) | low |
| `boxes` / `mask_px` | detector output size | 0 / small |
| `shield_cost_psnr` | PSNR of shielded vs marked, whole frame (visual cost of δ) | ≥ 38 dB |

Always compare the `marked` and `shield-*` rows for the same case and remover: the
shield has worked when the score drops.

## Removers covered here vs by hand

In `removers.py` (frame-level, Apache/MIT models):
`oracle-lama` (hand-drawn mask, worst case), `florence-lama` (WatermarkRemover-AI style),
`sam-lama` (IOPaint style).

By hand, from their own repos, on short clips (keep out of the shipped service):
- WatermarkRemover-AI (https://github.com/D-Ogi/WatermarkRemover-AI): the actual tool, MIT.
- IOPaint (https://github.com/Sanster/IOPaint): try the MAT and SD inpainting models too. These are the held-out removers for checking transfer.
- SAM2 + ProPainter (https://github.com/sczhou/ProPainter): video inpainting, non-commercial licence, eval only.

## Calibrating the codec proxy

`fortify.eot.CodecProxy` approximates x264 with JPEG-style quantization. To calibrate:
1. Pick a δ that was shielded without EOT (strength `low`).
2. Compare how much of it survives `--codec h264:23` vs `--codec jpeg:<q>`. Measure the PSNR between shielded and marked after the codec.
3. Set `CodecProxy.quality` to the JPEG range that matches the x264 CRFs you ship at.
