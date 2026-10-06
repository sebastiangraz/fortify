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
uv run python eval/run.py --shield medium --codec h264:23,none --cases 'sintel*'  # one δ, two codecs
uv run python eval/run.py --shield medium --tune grid=1 --cases 'sintel03*,kodim05*'  # budget test
uv run python eval/summary.py runs/p1-medium runs/p2-medium     # the README's tables, per run
```

- Each δ is computed once per case and scored under every codec in the `--codec` list.
- Shielding passes surrogates a View: the frame size, the crop's place in it, and a
  1024 px thumbnail of the marked frame, as a consumer would send. `--no-view` turns it off.
- Rows carry `loss_<surrogate>`, each surrogate's loss at the last PGD step.
- `--tune` overrides preset fields for budget experiments (`eps`/`alpha` in 8-bit levels).
  The service never does this.
- `--cases` takes comma-separated globs (`fnmatch`, so no `{a,b}` braces).
- `summary.py` counts removals (`removal_score` > 0.5) and mask moves per remover and codec,
  and the shield's cost, the way the README's tables do.
- **Single-frame `h264:<crf>` is harsher than delivery.** One frame encoded alone is an
  I-frame. Inside a clip, P/B-frames predict a static δ from the frame before and keep about
  twice as much of it at CRF 23 (see "Calibrating the codec proxy"). Eval numbers under
  `h264:23` are a pessimistic bound.

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
| `shield_cost_ssim_ring` | SSIM (luma) of shielded vs marked over the context ring: the crop minus the logo + 8 px. Sees structured δ, like the decoy's glyph-like ghost on a flat background, that PSNR averages away | ≥ 0.95 |

Always compare the `marked` and `shield-*` rows for the same case and remover: the
shield has worked when the score drops.

Read the two scores together, and look at the images:
- **`removal_score` under-reads faint marks.** On a blur or glass mark over texture, an inpainter
  that wipes the mark completely still misses clean pixel for pixel by about as much as the mark
  itself, so the score lands near 0–0.5. Phase 0 had a desert case at 0.23 with the mark visibly gone.
- **`mark_residual` ignores error that has nothing to do with the mark**, so it reads ≈ 0 there.
- **A broken fill on a shielded frame** (the `lama` term's goal) shows up as a large negative
  `removal_score` and a low `removed_psnr_logo`. On plain marks `removal_score` can still read
  0.8 there, because the white logo makes the baseline error huge. Read `removed_psnr_logo`, or
  the images. `mark_residual` is meaningless for broken fills (values of −2 or +1.7 just mean
  "far from clean").
- **`mark_residual` over-reads blur marks over sharp texture.** The blur mark *is* (blurred
  − sharp) in the logo's shape, and any smooth fill correlates with that. Values ≈ 1 or above on
  `kodim08`/`kodim15` blur came from fills where no logo shape is left. Check the restored PNGs
  before trusting a blur row.

## Removers covered here vs by hand

In `removers.py` (frame-level, Apache/MIT models):
`oracle-lama` (hand-drawn mask, worst case), `loose-lama` (the same box dilated 24 px instead
of 8), `florence-lama` (WatermarkRemover-AI style), `sam-lama` (IOPaint style).
- All three hand-mask-like removers dilate by 8 px, except `loose-lama`. The real tools differ:
  WatermarkRemover-AI's box mode inpaints the Florence-2 box **undilated** (`remwm.py`
  `get_watermark_mask`), and IOPaint's click-to-mask dilates SAM's mask by ~4 px (9 px kernel
  in `gen_frontend_mask`). Since the `lama` term is specific to the mask edge (README Phase 2),
  these few px matter. Run the real tools by hand before trusting a harness number.

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

`fortify.eot.CodecProxy` approximates x264 with JPEG-style quantization.
`calibrate_codec.py` measures how much of a δ survives each codec, inside each case's crop:
`s = <c(x + δ) − c(x), δ> / <δ, δ>`, 1 = intact, 0 = erased. The test δs look like the
attack's: random ±8 levels on a grid (1, 2 or 4 px), bilinear, feathered.

```
FFMPEG=... uv run python eval/calibrate_codec.py --clips      # ~5 min, CPU; log in runs/p2-calibrate.log
```

Phase 2 result (24 cases; clips = 48 frames of each clean clip, δ static on every frame):

| codec | grid 1 | grid 2 | grid 4 | ≈ proxy quality |
|---|---|---|---|---|
| x264 CRF 18, one frame | 0.27 | 0.46 | 0.65 | q70–90 |
| x264 CRF 23, one frame (eval's `h264:23`) | 0.09 | 0.25 | 0.50 | q30–50 |
| x264 CRF 28, one frame | 0.02 | 0.13 | 0.34 | ≤ q20 |
| x264 CRF 18, clip | 0.39 | 0.71 | 0.82 | q95 |
| x264 CRF 23, clip | 0.25 | 0.45 | 0.66 | q70–90 |
| x264 CRF 28, clip | 0.07 | 0.21 | 0.48 | q20–40 |
| proxy / PIL JPEG q50 | 0.09 / 0.09 | 0.34 / 0.34 | 0.56 / 0.57 | |

- The proxy's JPEG matches PIL's JPEG to within 0.005 at every quality: the DCT model is right.
- A static δ survives about twice as well inside a clip as in a lone frame at the same CRF.
- `CodecProxy.quality` is now q30–90 (was q45–85): it spans eval's I-frames at CRF 23 and
  delivered clips at CRF 18–28.
- Grid 1 keeps a third as much as grid 2 through one frame at CRF 23 (0.09 vs 0.25). This is
  random δ, though; PGD through the proxy can find grid-1 patterns that survive better.
