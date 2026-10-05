# fortify

Adversarial "vaccine" perturbations that make AI watermark-removal tools fail on
visible watermarks. Built for **Mark**, the watermark tool in
[videotools](../videotools) (`api/_lib/tools/mark.ts`). It lives in its own repo and
runs as a GPU service. The consumer sends crops of watermarked frames and gets back a small signed
residual (δ) to add before its final encode. Nothing in here knows about Mark's
filters, layout or encoders.

> **Status: first draft (2026-10-05).** The core attack, codec proxy, service, CLI, TS
> client and eval harness are written. What has been run so far is in
> [Status](#status). Next is **Phase 0** (baseline red-team), in [Roadmap](#roadmap).

---

## 1. The problem

Mark burns a logo into video: plain, a "glass" refraction look, or blur. That
works, but off-the-shelf removers take it off. They almost all run the same
two-stage pipeline:

| stage | what real tools use | how it finds or fills the mark |
|---|---|---|
| **detect / segment** | Florence-2 open-vocabulary detection with the prompt "watermark" (WatermarkRemover-AI); SAM / SAM2 from a click or box (IOPaint; video tools propagate SAM2 masks through the clip); YOLO-style detectors | outputs a box or mask over the logo |
| **inpaint** | LaMa (most common), MAT, Stable Diffusion inpainting; for video, ProPainter or E2FGVI | fills the mask from the **context ring** around it |

fortify adds a near-invisible perturbation (|δ| ≤ 4–12 / 255) around the mark. It is
optimized so that:
- detectors and segmenters stop finding the logo, and the mask misses it; and
- the inpainter's fill, driven by the perturbed context ring, comes out visibly broken
  instead of a clean plate.

### Threat model and honest limits
- **In scope:** casual to moderately skilled actors who run off-the-shelf tools on our
  published (H.264) output.
- **Not fully in scope:** an attacker who draws the mask by hand, uses a strong
  pixel-space inpainter the shield never trained against, or heavily rescales or
  re-encodes first. That raises their cost but does not stop them. Adversarial
  protection is an arms race. Pair it with an invisible forensic watermark (Roadmap,
  Phase 5) so ownership can still be proven after a successful removal.
- **Compression is the main enemy.** H.264 at normal bitrates erases naive
  high-frequency perturbations. That is why the attack optimizes through a codec
  proxy (EOT) and keeps δ low-frequency.

## 2. Why not Watermark Vaccine directly

[Watermark Vaccine](https://github.com/thinwayliu/Watermark-Vaccine) (Liu et al.,
ECCV 2022, [arXiv 2207.08178](https://arxiv.org/abs/2207.08178)) is the starting idea.
We **do not use its code**:
- **No licence:** the repo has none (GitHub reports `null`), so we have no right to reuse it.
- **Research-script constraints:** hard-coded `.cuda()`, fixed 256×256 input, white-box against three
  research removers only (WDNet, BVMR, SplitNet). It doesn't cover the Florence-2/SAM + LaMa
  pipelines people actually use.
- **Weak robustness:** the paper reports it survives only mild JPEG (quality ≥ 80) and blur.
  It was not tested against H.264.

Ideas we take from it (published, reimplemented from scratch here):
- **Plain L∞ PGD** (sign-gradient steps, projection to an ε ball). Vaccine used 50 steps,
  α = 2/255, ε = 8/255.
- **DWV, the "disrupting" vaccine:** gradient *ascent* on the remover's reconstruction
  error, so its output gets wrecked. → our `lama` surrogate.
- **IWV, the "inerasable" vaccine:** gradient *descent* so the remover thinks there is no
  watermark (pred ≈ input, predicted mask ≈ 0). → our `florence2` and `sam` surrogates
  (suppress the detection or mask).
- **Ensembling** surrogates improves transfer to unseen removers. The paper's "stacked
  vaccine" finding.

Other references behind the design:
- FGSM, Goodfellow et al. 2014: one step, `x' = x + ε·sign(∇L)`.
- PGD, Madry et al. 2017: iterated FGSM with projection.
- EOT, Athalye et al. 2018: average gradients over random transforms.
- Attack-SAM ([2305.00866](https://arxiv.org/abs/2305.00866)): the ClipMSE mask-suppression loss.
- PhotoGuard, Mist and DiffusionGuard: anti-inpainting for diffusion models.
- AWD-AGP (ACM MM 2023): anti-removal against inpainting-based removers.
- Universal Watermark Vaccine (CVPRW 2023): one image-agnostic δ.

## 3. How it works

### 3.1 Groups: one δ per stretch of frames
The unit of work is a **group**: N ≤ 8 crops of consecutive frames in which the mark
sits still at the same crop rect. fortify computes **one universal δ per group**,
optimized jointly over its frames. A full-frame per-frame attack would cost tens of
seconds per 1080p frame; one δ per group around the mark only is what makes video
affordable.

Mark rotates the logo between positions at a fixed cadence ("stays", about 2 s each). So
in Mark, one stay = one group. The consumer samples a few frames per stay (e.g. 3:
start, middle, end) and later applies the same δ to every frame of the stay. Frames the
optimizer didn't see are covered by the EOT and by the fact that the mark itself doesn't
move.

### 3.2 Crop, ring and feather
- **The consumer chooses the crop.** It is the logo rect plus a context ring.
  `fortify.region.crop_around(logo, W, H)` is the recommended rule: ring = max(32 px,
  half the logo's long side), clipped to the frame, even-sized. The ring matters because
  inpainters fill the hole from it, so δ in the ring is what reaches them. Inside the hole
  LaMa never sees our pixels.
- `logo` in the request is the mark's rect **inside the crop**. Surrogates use it as the
  attacker's mask or box prompt (dilated, as removers do).
- δ is multiplied by a **feather** (linear ramp to 0 at the crop border, width = ring/4)
  so pasting it back leaves no seam.

### 3.3 The attack (`src/fortify/attack.py`)
```
θ ← 0 on an (H/grid × W/grid) grid               # grid > 1 ⇒ δ is low-frequency by construction
repeat steps:
    δ = upsample_bilinear(θ) · feather
    L = mean over eot_samples of  loss(T(clip(x + δ)))   # x: (N,3,H,W) group, T: random codec proxy
    θ ← clip(θ − α·sign(∇θ L), −ε, ε)
δ = round(δ·255)/255                              # whole 8-bit levels (the output is 8-bit anyway)
```
- `loss` returns something to **minimise**; each surrogate picks its own sign.
- FGSM = one step with α = ε (the `low` preset).
- Bilinear upsampling is a convex combination, so |δ| ≤ ε still holds after it.
- Optimizing on a coarse grid (grid = 2) puts δ's energy in low and mid frequencies,
  which survive 4:2:0 chroma and DCT quantization far better.

### 3.4 Codec proxy, i.e. EOT (`src/fortify/eot.py`)
Each EOT sample draws a random chain, all differentiable:
1. ±1 px shift;
2. down/up-scale (×0.6–1.0, p = 0.4);
3. Gaussian blur (σ ≤ 0.8, p = 0.3);
4. JPEG-style 8×8 DCT quantization on YCbCr with 4:2:0 chroma (quality 45–85, p = 0.9),
   rounding through a straight-through estimator;
5. light noise.

This approximates what x264 at Mark's CRFs and later platform re-encodes do. The quality
range is a guess until calibrated with real x264 (`eval/README.md` → "Calibrating the
codec proxy").

### 3.5 Surrogates (`src/fortify/surrogates/`)
Each surrogate's `bind(ctx)` sees the clean group once (to compute references and
targets) and returns `loss(x_adv)`.

| name | stands in for | loss (minimised) | default weight |
|---|---|---|---|
| `lama` | LaMa inpainting with a dilated box mask (8 px) | −MSE between LaMa's fill on x_adv and its fill on clean, inside the hole (DWV-style: push the fill away from the plausible clean plate) | 1.0 |
| `sam` | SAM / SAM2 with the logo box as prompt | Attack-SAM ClipMSE: Σ relu(logit + τ)² over the (dilated) logo, τ = 2, so the mask comes back empty | 0.5 |
| `florence2` | Florence-2 `<OPEN_VOCABULARY_DETECTION>` with "watermark" / "logo" | −CE of the clean output's `<loc_*>` tokens, capped at 12, so it stops repeating its boxes. Prompts with no detection on clean are skipped | 0.05 |
| `toy` | a fixed random conv net | for tests and `selftest`; no weights | — |
| `null` | nothing | constant loss ⇒ δ ≡ 0 exactly; for consumer smoke tests | — |

- The ensemble is a weighted sum: `FORTIFY_SURROGATES="lama:1,sam:0.5,florence2:0.05"`.
- The weights are **untuned first guesses**: the terms have very different scales (LaMa
  MSE ~0.01, CE ~10, ClipMSE up to ~100). Tune them with eval/.
- **Preprocessing is redone in torch** (resize + ImageNet normalisation) so gradients
  reach the pixels.
  - `bind()` compares it against the real Hugging Face processor and warns if the max
    difference is > 0.1.
  - The SAM wrapper tries both "pad longest side to 1024" (SAM) and "stretch to 1024²"
    (SAM2) and keeps whichever matches.
- **Models:**
  - LaMa: the TorchScript `big-lama.pt` (the one IOPaint and WatermarkRemover-AI load),
    downloaded to `weights/` on first use.
  - Florence-2 and SAM: from the Hugging Face hub. Override with `FORTIFY_FLORENCE2` /
    `FORTIFY_SAM`; the defaults are `florence-community/Florence-2-large` (the native transformers 5 port) and
    `facebook/sam2.1-hiera-large`.

### 3.6 Strength presets (`src/fortify/vaccinate.py`)
| strength | method | ε | steps | α | EOT samples / step | grid |
|---|---|---|---|---|---|---|
| `low` | FGSM | 4/255 | 1 | 4/255 | — | 1 |
| `medium` | PGD + EOT | 8/255 | 50 | 1.5/255 | 2 | 2 |
| `high` | PGD + EOT | 12/255 | 100 | 1.5/255 | 4 | 2 |

Cost scales with steps × eot_samples × frames per group × surrogates. Measure it (Status)
before promising a time budget to the consumer.

## 4. API contract

The service (`src/fortify/service.py`) and the TS client (`packages/client`) both
implement this. Change all three together and bump `version` on breaking changes.

### `POST /v1/vaccinate`
Headers: `Authorization: Bearer $FORTIFY_TOKEN` (required when the server has
`FORTIFY_TOKEN` set) and `Content-Type: application/json`.
```jsonc
{
  "version": 1,
  "strength": "medium",                  // low | medium | high
  "groups": [                            // 1..64
    {
      "id": "stay-0",                    // echoed back, ≤ 64 chars
      "logo": { "x": 48, "y": 40, "w": 260, "h": 120 },  // mark rect inside the crop
      "frames": ["<base64 PNG>", "..."]  // 1..8 crops, same size, ≤ 1024×1024 px
    }
  ]
}
```
Response `200`:
```jsonc
{
  "version": 1,
  "offset": 128,
  "surrogates": "lama,sam,florence2",
  "groups": [ { "id": "stay-0", "delta": "<base64 PNG>", "eps": 8, "ms": 2140.5 } ],
  "ms": 2210.3
}
```
Errors:
- `400`: unreadable frame, frames differ in size, or logo rect outside the crop.
- `401`: bad token.
- `413`: crop too large.
- `422`: schema violation (FastAPI).
- `5xx`: server fault.

The TS client maps these to `FortifyError` codes: `rejected` (4xx), `unavailable` (5xx
or network), `timeout`, `bad_response`.

Groups run one after another on one GPU, guarded by a lock. Requests queue.

### `GET /health`
Returns `{ ok, version, api, device, cuda, surrogates, loaded }`. `loaded` turns true
after the first vaccinate call; models load lazily.

### δ encoding
- δ is an **RGB PNG the size of the crop**, with `pixel = 128 + δ·255`, so δ is in
  whole 8-bit levels. 128 means unchanged; `eps` bounds |pixel − 128|.
- To apply it, add `(pixel − 128)` to the watermarked frame's RGB at the crop rect and
  clip to 0–255.
- Apply in **RGB**, after the mark is composited and **before** the final encode.
  ffmpeg sketch:
  ```
  [frame]split[a][b];
  [a]crop=W:H:X:Y,format=gbrp[c];
  [delta]format=gbrp[d];
  [c][d]blend=all_expr='A+B-128'[cs];      # blend clips to 0..255
  [b][cs]overlay=X:Y:enable='between(t,T0,T1)'[out]
  ```
  That is one δ per group, enabled for its time range. With many stays, the consumer
  may prefer to build the δ inputs into a single stream or sprite instead; that's its
  call.

### TS client
```ts
import { createClient } from "@fortify/client";
const fortify = createClient({ url: process.env.FORTIFY_URL!, token: process.env.FORTIFY_TOKEN, timeoutMs: 120_000 });
const { deltas } = await fortify.vaccinate({ strength: "medium", groups, signal });
// deltas[i].png: Uint8Array (PNG), deltas[i].eps, deltas[i].id
```
- Zero dependencies; built with `tsc`.
- Not published yet. Consume it via a `file:`/git dependency, or publish it to a private
  registry later.

## 5. Consumer integration (videotools / Mark)

videotools has a comment at the seam in `api/_lib/tools/mark.ts` (`addWatermark`, just
before it returns the `Render`). The intended flow, kept deliberately loose because both
repos will change:
1. **Gate.** Only when the job asks for protection *and* `FORTIFY_URL` is configured.
   Otherwise this is a no-op.
2. **Rects.** For each rotation stay (or the whole clip if the mark doesn't move), take
   the mark rect from Mark's layout code, then `crop_around` it (port the formula above).
3. **Sample.** With one ffmpeg call over the watermarked graph, export ~3 frames per stay,
   cropped to that rect, as PNG.
4. **Call** `vaccinate` with one group per stay, under a hard time budget. Mark has 300 s
   per request with ~240 s of encode budget; give fortify a slice, e.g. ≤ 120 s.
5. **Apply.** Write the δ PNGs to the job's workDir as extra ffmpeg inputs and blend them
   into the existing graph at the same rects and time ranges, so the video is still
   encoded **once**.
6. **Fall back.** On any `FortifyError` or timeout, ship the unshielded output and flag
   it in the response. Never fail the job because of the shield.
7. **Smoke.** With the server on `FORTIFY_SURROGATES=null`, the shielded output must be
   byte-identical to the unshielded one, which proves the plumbing adds nothing by itself.

## 6. Repo layout
```
src/fortify/
  __init__.py        version, DELTA_OFFSET
  attack.py          FGSM/PGD core (region, grid, EOT hook, quantization)
  eot.py             differentiable codec proxy (DCT/JPEG 4:2:0, resize, blur, shift, noise)
  region.py          Rect, crop_around, box_mask, feather
  imageio.py         PNG ⇄ tensor, δ wire encoding
  models.py          frozen model loaders (LaMa TorchScript, Florence-2, SAM/SAM2), normalisation
  surrogates/        base (Context, Ensemble), lama, sam, florence2, toy/null, registry
  vaccinate.py       presets + one group → one δ
  service.py         FastAPI app
  cli.py             fortify selftest | vaccinate | serve
tests/               pytest: attack invariants, codec proxy, wire format, service contract (toy surrogate)
eval/                red-team harness: run.py, removers.py, README.md (protocol and metrics)
packages/client/     @fortify/client (TS, zero deps, bun test)
deploy/              Dockerfile (uv, cu128) and modal_app.py (serverless GPU, draft)
weights/             downloaded model files (gitignored)
runs/                CLI/eval outputs (gitignored)
```

## 7. Setup and commands

Requirements: Windows or Linux, an NVIDIA GPU (dev box: **RTX 5090 FE, 32 GB, Blackwell
sm_120**, which is why torch comes from the **cu128** index), [uv](https://docs.astral.sh/uv/)
and [Bun](https://bun.sh) for the client.

```sh
uv sync --extra models --extra serve        # Python 3.13 venv, torch cu128
uv run fortify selftest                     # torch/CUDA check + PGD on the toy surrogate (no downloads)
uv run pytest                               # unit tests (CPU fine)
uv run ruff check .

# one group by hand: crops of the same rect from a few frames, logo rect inside the crop
uv run fortify vaccinate --frames a.png b.png c.png --logo 48,40,260,120 --strength medium \
  --surrogates lama,sam,florence2 --out runs/try1
#   → runs/try1/delta.png (wire format), delta_x8.png (amplified view), shielded_*.png

uv run fortify serve --port 8765            # HTTP API; set FORTIFY_TOKEN for auth
uv run python eval/run.py --data eval/data --codec h264:23 [--shield medium]

cd packages/client && bun install && bun test && bun run build
```
- The first run with real surrogates downloads LaMa (~200 MB), Florence-2-large (~1.5 GB)
  and SAM2.1-large (~900 MB). Set `HF_HOME` / `FORTIFY_WEIGHTS` to move the caches.
- For eval with real x264, set `FFMPEG` to videotools' pinned build
  (`../videotools/api/_bin/ffmpeg/win32-x64/ffmpeg.exe`).

Environment variables:

| var | default | meaning |
|---|---|---|
| `FORTIFY_TOKEN` | unset (no auth) | bearer token for `/v1/*` |
| `FORTIFY_SURROGATES` | `lama,sam,florence2` | ensemble spec (`name[:weight],...`) |
| `FORTIFY_DEVICE` | cuda if available | force `cpu` / `cuda:1` |
| `FORTIFY_WEIGHTS` | `./weights` | LaMa download dir |
| `FORTIFY_FLORENCE2` / `FORTIFY_SAM` | florence-community/Florence-2-large / facebook/sam2.1-hiera-large | HF model ids (SAM v1 ids like `facebook/sam-vit-huge` also work) |
| `FFMPEG` | `ffmpeg` on PATH | eval's real H.264 codec |

**Deploying:**
- Docker: `deploy/Dockerfile`, run with `docker run --gpus all ...`.
- Modal: `deploy/modal_app.py`. L4 GPU, scale to zero, weights cached on a volume. This is a draft and
  has not been deployed yet.
- The service is stateless apart from its model cache.

## 8. Licensing

| component | licence | use |
|---|---|---|
| Watermark Vaccine code | **none** | idea only; no code copied |
| LaMa (big-lama) | Apache-2.0 | surrogate + eval |
| Florence-2 | MIT | surrogate + eval |
| SAM / SAM2 | Apache-2.0 | surrogate + eval |
| WDNet / SLBR weights | unverified | not used yet; check before adding as a `blind_remover` surrogate |
| ProPainter | NTU S-Lab, **non-commercial** | eval by hand only, never in the service |
| WatermarkRemover-AI / IOPaint | MIT / Apache-2.0 | eval by hand |

## Status

What has actually been run is recorded here. Keep it current.

- [x] Repo scaffolded; core attack, codec proxy, surrogates, service, CLI, eval harness,
      TS client and deploy files written (2026-10-05).
- [x] `uv run fortify selftest` on the 5090. torch 2.11.0+cu128, capability (12, 0).
      Toy surrogate: medium ≈ 0.7 s, high ≈ 3 s per group; `null` gives exactly 0.
- [x] `uv run pytest`: 14 passed. `ruff check` and `ruff format` clean.
- [x] `bun test` (3 pass) and `bun run typecheck` in packages/client.
- [x] **Real surrogates load and backpropagate (2026-10-05).** Probe: a synthetic
      384×256 crop with a 160×80 "WATERMARK" box, 2 frames, ε = 8, 10 PGD steps,
      2 EOT samples with the codec proxy. Each surrogate ran in its own process.

      | surrogate | result | ms / step | peak VRAM |
      |---|---|---|---|
      | `lama` | fill MSE in the hole 0 → 0.020 (≈ 36/255 RMS), so the fill is pushed well off the clean plate | ~350 | 4.1 GiB |
      | `sam` (sam2.1-hiera-large) | ClipMSE 70 → 33; no preprocessing-drift warning, so stretch/pad detection matched | ~750 | 24.5 GiB |
      | `florence2` (florence-community/Florence-2-large) | detected the box on clean; loc-token CE 0.95 → 3.77 | **~41 000** | **34.3 GiB**, over the 32 GB card, spilling to shared memory |
- [ ] Phase 0 baseline numbers
- [ ] Timing per group per preset with the full ensemble

Findings and known issues:
- **"Disrupting" losses need a random start.** The `lama`/`toy` losses compare against
  the surrogate's own clean output, so their gradient at δ = 0 is exactly 0. A
  zero-start PGD or FGSM never moves (found and fixed: every preset has `random_start`,
  and `low` is R+FGSM). Keep this in mind for any new DWV-style surrogate.
- **Florence-2 is the bottleneck. Fix this first in Phase 1.** Its cost comes from 768²
  input × 2 frames × 2 prompts × 2 EOT samples, all summed before a single backward.
  Options, roughly in order:
  - backward per target (accumulate grads instead of summing losses);
  - bf16 autocast;
  - one prompt instead of two;
  - Florence-2-base instead of large;
  - gradient checkpointing on the vision tower;
  - fewer steps for this term only, i.e. alternate which surrogates get a step.
- **Florence-2 must come from the native transformers 5 port.** The original
  `microsoft/Florence-2-*` remote code crashes on transformers 5.x
  (`Florence2LanguageConfig ... forced_bos_token_id`), so `models.py` loads
  `Florence2ForConditionalGeneration` from `florence-community/Florence-2-large`.
- **SAM2 memory:** 24.5 GiB with 2 frames × 2 EOT samples at 1024². Running it with the
  full ensemble needs per-surrogate backward, or bf16.
- **Download stalls:** Hugging Face's Xet CDN failed once mid-download. Setting
  `HF_HUB_DISABLE_XET=1` fixed it.
- SAM2 load prints a harmless "`sam2_video` to instantiate `sam2`" notice.

Still unverified:
- `eval/removers.py` `florence_lama`: `processor.post_process_generation` on the native
  processor (written against the remote-code API).
- Whether `labels = generated[:, 1:]` is exactly aligned with the native Florence-2 decoder.
  The loss behaves as expected, but check token-by-token.
- The DCT-JPEG proxy's quality range vs real x264.
- Weight balance of the ensemble (term scales differ by about 10³).

## Roadmap
- **Phase 0, baseline red-team (do first; it decides scope).**
  - Export about 20 cases from videotools (plain, glass and blur × sizes × backgrounds)
    and run `eval/run.py` without `--shield`.
  - Also run WatermarkRemover-AI, IOPaint (LaMa/MAT/SD) and SAM2 + ProPainter by hand.
  - Question: how much do Mark's rotation and glass refraction already resist removal?
    The answer may change what fortify needs to target.
- **Phase 1, single-frame shield.** Get `lama` + `florence2` working end to end through
  the CLI. Compare ε = 8 before and after `jpeg:75` / `h264:23`.
- **Phase 2, robustness.**
  - Calibrate the codec proxy.
  - Add and tune the `sam` surrogate.
  - Tune the ensemble weights (normalise each term by its value at step 0?).
  - Per-stay universal δ over sampled frames.
  - A targeted Florence-2 loss (descend towards "no box") if untargeted ascent only moves boxes.
  - Optionally a `blind_remover` surrogate (SLBR) if its licence allows.
  - Hold out MAT/SD inpainting to measure transfer.
- **Phase 3, service.** Deploy (Modal or self-hosted 5090), latency measurements, a
  timing-aware strength choice.
- **Phase 4, videotools integration** per §5, behind a toggle, with the `null`
  byte-identical smoke case.
- **Phase 5, invisible forensic watermark.** Meta VideoSeal or Adobe TrustMark
  (licences to verify) as a second endpoint, for provenance when the visible mark is
  removed anyway.
- **Later:** an async job API for long videos (the 300 s request limit); temporal
  consistency of δ across stays; a universal δ per logo, precomputed, for instant "low"
  protection.

## Conventions
- Python: uv only (no pip/conda), Python 3.13, ruff, pytest. Config lives in `pyproject.toml`.
  Add no extra config files unless defaults don't do.
- TS: Bun for install, tests and scripts; `tsc` only for the build and `.d.ts`. No transpile shims (tsx/jiti).
- Line endings: LF (`git config core.autocrlf false` is set in this repo).
- Image tensors are float in [0, 1], shape (N, 3, H, W). Losses are minimised. δ is (1, 3, H, W).
