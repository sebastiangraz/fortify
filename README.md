# fortify

Adversarial "vaccine" perturbations that make AI watermark-removal tools fail on
visible watermarks. Built for **Mark**, the watermark tool in
[videotools](../videotools) (`api/_lib/tools/mark.ts`). It lives in its own repo and
runs as a GPU service. The consumer sends crops of watermarked frames and gets back a small signed
residual (δ) to add before its final encode. Nothing in here knows about Mark's
filters, layout or encoders.

> **Status: first draft (2026-10-05).** The core attack, codec proxy, service, CLI, TS
> client and eval harness are written. What has been run so far is in
> [Status](#status). Phases 0 (baseline red-team) and 1 (single-frame shield) are done.
> **Phase 2** (robustness) is partly done; what's left is in [Roadmap](#roadmap).

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
v ← 0
repeat steps:
    δ = upsample_bilinear(θ) · feather
    L = mean over eot_samples of  loss(T(clip(x + δ)))   # x: (N,3,H,W) group, T: random codec proxy
    g = ∇θ L;  v ← μ·v + g / mean|g|                      # momentum (MI-FGSM), μ = 0.9; μ = 0 is plain PGD
    θ ← clip(θ − α·sign(v), −ε, ε)
δ = round(δ·255)/255                              # whole 8-bit levels (the output is 8-bit anyway)
```
- `loss` returns something to **minimise**; each surrogate picks its own sign.
- FGSM = one step with α = ε (the `low` preset).
- **Momentum** (Dong et al. 2018) since Phase 2. Each step sees a different EOT draw: a
  codec, and for `lama` a different attacker mask. Plain sign steps chase the latest draw and
  oscillate; momentum steps where most draws agree. Without it, cycling `lama` over masks
  broke nothing at all.
- Bilinear upsampling is a convex combination, so |δ| ≤ ε still holds after it.
- Optimizing on a coarse grid (grid = 2) puts δ's energy in low and mid frequencies,
  which survive 4:2:0 chroma and DCT quantization far better.

### 3.4 Codec proxy, i.e. EOT (`src/fortify/eot.py`)
Each EOT sample draws a random chain, all differentiable:
1. ±1 px shift;
2. down/up-scale (×0.6–1.0, p = 0.4);
3. Gaussian blur (σ ≤ 0.8, p = 0.3);
4. JPEG-style 8×8 DCT quantization on YCbCr with 4:2:0 chroma (quality 30–90, p = 0.9),
   rounding through a straight-through estimator;
5. light noise.

This approximates what x264 at Mark's CRFs and later platform re-encodes do. The quality
range was calibrated against real x264 in Phase 2 (`eval/calibrate_codec.py`; table in
`eval/README.md` → "Calibrating the codec proxy"):
- One frame encoded alone at CRF 23 (eval's `h264:23`) erases a δ like JPEG q30–50.
- Inside a clip, CRF 23 is like q70–90 and CRF 28 like q20–40, because P/B-frames predict a
  static δ from the frame before.
- q30–90 spans both. The proxy's JPEG itself matches PIL's to within 0.005.

### 3.5 Surrogates (`src/fortify/surrogates/`)
Each surrogate's `bind(ctx)` sees the clean group once (to compute references and
targets) and returns `loss(x_adv)`.

| name | stands in for | loss (minimised) | default weight |
|---|---|---|---|
| `lama` | LaMa inpainting with the logo box as mask, dilated 8 and 24 px (one per call, in turn) | −MSE between LaMa's fill on x_adv and its fill on clean, inside the hole (DWV-style: push the fill away from the plausible clean plate) | 1.0 |
| `sam` | SAM / SAM2 with the logo box as prompt | Attack-SAM ClipMSE: Σ relu(logit + τ)² over the (dilated) logo, τ = 2, so the mask comes back empty | 1.0 |
| `florence2` | Florence-2 `<OPEN_VOCABULARY_DETECTION>` with "watermark" / "logo", on the whole frame | **targeted decoy**: CE of the clean answer with every box that covers most of the logo swapped for a decoy box in the ring beside it, plus a hinge that keeps the whole-logo box from getting likelier (see below) | 1.0 |
| `toy` | a fixed random conv net | for tests and `selftest`; no weights | — |
| `null` | nothing | constant loss ⇒ δ ≡ 0 exactly; for consumer smoke tests | — |

- **The ensemble is a weighted sum with balanced gradients**:
  `FORTIFY_SURROGATES="lama:1,florence2:1"`.
  - pgd backpropagates each surrogate on its own (`attack.SumLoss`). That keeps memory at
    one surrogate's graph and one EOT sample at a time.
  - Each surrogate's gradient is scaled to unit mean |g| before weighting. A weight is then
    that surrogate's share of the step, whatever its loss scale (LaMa MSE ~0.01, CE ~10,
    ClipMSE ~100).
  - Without balancing, Florence-2's gradient drowned LaMa's: its loss stayed at −0.0002, against
    −0.54 when run alone. The weights themselves (1:1) are still untuned.
- **`florence2` targets a decoy.**
  - Florence-2 OVD always answers with a box. Even on frames without a mark it boxes some
    other object, so "no box" is off-distribution.
  - Untargeted ascent on the clean answer (the first draft) moved boxes instead. On sintel03
    it grew the box from the "S" alone to the whole wordmark.
  - So bind() swaps each logo-covering box's four `<loc_*>` tokens for a decoy: the biggest ring
    strip beside the logo, 16 px clear of it and clear of the feathered border. PGD then
    descends the CE of the location tokens. WatermarkRemover-style tools inpaint the decoy
    strip and leave the mark.
- **`lama` trains on a bank of attacker masks** (`LamaSurrogate.masks`, default `box:8,box:24`;
  `sam:<px>` adds SAM's own mask). Its effect is specific to the mask's edge, so a mask the
  bank doesn't contain gets a clean fill. See [Phase 2 results](#phase-2-results-2026-10-06).
- **`florence2` keeps partial boxes.** Only a box covering ≥ 50% of the logo gets the decoy.
  On large wordmarks the clean detector often boxes just the "S" (coverage ~0.25), and
  decoying that box made it find the whole mark in Phase 1. A partial box stays the target,
  which holds the miss. A hinge on the same answer with the whole logo boxed stops its CE
  from dropping below its clean value.
- **`florence2` needs the View** (`surrogates.View`: frame size, the crop's place in it, and a
  thumbnail of the frame; API field `view`, CLI `--view`/`--background`).
  - The real tool runs on the whole frame shrunk to 768². With a View, the surrogate builds that
    picture: the thumbnail, with the crop pasted in at its place.
  - On a 1920×816 frame that's a 2.5× downscale of the logo. The crop stretched to 768² is a
    different picture at a different scale. In Phase 1, δ optimized on the crop view did
    nothing to full-frame detection, even before any codec.
  - Without a View, florence2 falls back to the crop view.
- **Florence-2 cost:**
  - The vision tower runs once per frame, and its features are shared by both prompts.
  - Frames are batched per prompt, and backpropagated in chunks of 2 frames.
  - One frame with 2 prompts costs ~180 ms forward + backward on the 5090; peak VRAM is
    7.4 GiB with lama.
  - bf16 autocast was no faster at 1–3 frames, and its gradient agreed with fp32 in sign on only
    ~75% of pixels, so it stays fp32.
- **Preprocessing is redone in torch** (resize + ImageNet normalisation) so gradients
  reach the pixels.
  - `bind()` compares it against the real Hugging Face processor and warns on a mismatch:
    max difference > 0.1 for SAM, mean > 0.01 for Florence-2. Torch and PIL bicubic differ
    by up to 0.15 on sharp edges of real frames, with a mean of ~0.001.
  - The SAM wrapper tries both "pad longest side to 1024" (SAM) and "stretch to 1024²"
    (SAM2) and keeps whichever matches.
- **Models:**
  - LaMa: the TorchScript `big-lama.pt` (the one IOPaint and WatermarkRemover-AI load),
    downloaded to `weights/` on first use.
  - Florence-2 and SAM: from the Hugging Face hub. Override with `FORTIFY_FLORENCE2` /
    `FORTIFY_SAM`; the defaults are `florence-community/Florence-2-large` (the native transformers 5 port) and
    `facebook/sam2.1-hiera-large`.

### 3.6 Strength presets (`src/fortify/vaccinate.py`)
| strength | method | ε | steps | α | EOT samples / step | grid | momentum |
|---|---|---|---|---|---|---|---|
| `low` | R+FGSM | 4/255 | 1 | 2/255 | — | 1 | — |
| `medium` | PGD + EOT | 8/255 | 50 | 1.5/255 | 2 | 2 | 0.9 |
| `high` | PGD + EOT | 12/255 | 100 | 1.5/255 | 4 | 2 | 0.9 |

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
      "frames": ["<base64 PNG>", "..."], // 1..8 crops, same size, ≤ 1024×1024 px
      "view": {                          // optional; florence2 needs it (§3.5)
        "frame": { "w": 1920, "h": 1080 },   // full frame size
        "at": { "x": 1490, "y": 576 },       // the crop's top-left in the frame
        "background": "<base64 PNG>"         // optional: the whole frame, any size ≤ 1024×1024 px
      }
    }
  ]
}
```
`view` was added in Phase 1 as an optional field, so `version` stays 1. Without
`background`, the surrogate fills the rest of the frame with the crop's mean colour.
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
- `400`: unreadable frame or background, frames differ in size, logo rect outside the
  crop, or crop outside the view's frame.
- `401`: bad token.
- `413`: crop or background too large.
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
// groups[i].view = { frame: { w, h }, at: { x, y }, background?: Uint8Array /* PNG */ }
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
   cropped to that rect, as PNG. Also export one whole frame per stay, scaled to ≤ 1024 px
   on its long side, as the `view.background`. Send `view.frame` and `view.at` too.
   Without them, the Florence-2 term attacks the wrong picture (§3.5).
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
  surrogates/        base (Context, View, Ensemble), lama, sam, florence2, toy/null, registry
  vaccinate.py       presets + one group → one δ
  service.py         FastAPI app
  cli.py             fortify selftest | vaccinate | serve
tests/               pytest: attack invariants, codec proxy, wire format, service contract (toy surrogate)
eval/                red-team harness: run.py, removers.py, export_cases.ts (cases from Mark), score.py /
                     score_clip.py (hand-run tools), video_masks.py, README.md (protocol and metrics)
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

# one group by hand: crops of the same rect from a few frames, logo rect inside the crop,
# plus where the crop sits in the full frame (W,H,X,Y) and a thumbnail of that frame
uv run fortify vaccinate --frames a.png b.png c.png --logo 48,40,260,120 --strength medium \
  --view 1920,816,1490,576 --background thumb.png --surrogates lama,florence2 --out runs/try1
#   → runs/try1/delta.png (wire format), delta_x8.png (amplified view), shielded_*.png;
#     prints each surrogate's loss (first → last step) and peak VRAM

uv run fortify serve --port 8765            # HTTP API; set FORTIFY_TOKEN for auth
uv run python eval/run.py --data eval/data --codec h264:23[,none] [--shield medium] [--cases 'sintel*']

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
| WDNet weights | unverified | not used |
| SLBR code and weights | **none** (GitHub `null`, none in the README; checked 2026-10-06) | not usable as a `blind_remover` surrogate |
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
- [x] **Phase 0 baseline red-team (2026-10-06).** Results are in
      [Phase 0 results](#phase-0-results-2026-10-06) below.
- [x] **Phase 1 single-frame shield, `lama` + `florence2` (2026-10-06).** Results are in
      [Phase 1 results](#phase-1-results-2026-10-06) below. `pytest` 15 passed, `bun test` 4 pass,
      ruff and tsc clean.
- [x] **Phase 2 robustness, first pass (2026-10-06).** Results are in
      [Phase 2 results](#phase-2-results-2026-10-06) below. `pytest` 17 passed, ruff clean.
- [ ] Timing per group per preset with the full ensemble (`sam` included)

### Phase 2 results (2026-10-06)

**What changed**
- `lama` trains on a bank of attacker masks (box dilated 8 and 24 px), one per call, in turn.
- The PGD presets use momentum (μ = 0.9).
- `florence2` keeps partial boxes as the target and holds off the whole-logo box (§3.5).
- The codec proxy's quality range is calibrated: q30–90, was q45–85 (§3.4).
- `sam` backpropagates frame by frame.
- Eval gained a held-out attacker, `loose-lama` (box dilated 24 px), and a cost metric, ring
  SSIM. `eval/summary.py` prints the tables below; on `runs/p1-medium` it reproduces Phase 1's
  table exactly.

**Eval:** `eval/run.py --shield medium --surrogates lama,florence2 --codec h264:23,none`, all 24
cases, output in `runs/p2-medium`. Cases removed out of 24 (marked → shielded), same counting as
Phase 1:

| attacker | h264:23, Phase 1 | h264:23, Phase 2 | no codec, Phase 2 | mask off / onto (h264:23) |
|---|---|---|---|---|
| `florence-lama` (WatermarkRemover-style) | 17 → 15 | 17 → **7** | 17 → 7 | **10 off / 0 onto** (Phase 1: 4 / 3) |
| `oracle-lama` (box + 8 px) | 19 → 8 | 19 → 11 | 21 → 10 | — |
| `loose-lama` (box + 24 px) | not run | 18 → 12 | 18 → 9 | — |
| `sam-lama` (SAM mask + 8 px) | 21 → 19 | 21 → 20 | 21 → 20 | 0 / 0 |

- Shield cost: PSNR 44.7 dB mean, 38.7 dB worst (Phase 1: 46.2 / 40.2). Ring SSIM 0.90 mean;
  worst sintel22-glass-medium 0.78, sintel12-blur 0.83, sintel22-plain-large 0.84. That misses
  the ≥ 0.95 target in eval/README: δ is visible on flat backgrounds.
- Time: 30 s per single-frame group on average, 42 s at most, the same as Phase 1.

What the numbers say:
- **The detector term is now the strong one, and it survives H.264.**
  - Florence-2 misses the logo on 10 of 24 cases after x264 CRF 23, against 4 in Phase 1.
  - **No mask moves onto the logo any more.** On kodim13, sintel19 and sintel22-plain-large
    (Phase 1's backfires) it keeps boxing only the "S" (coverage ~0.25, was 1.0).
  - Momentum is probably most of the H.264 gain. The decoy fix alone (probe, before momentum)
    stopped the backfires but didn't add wins.
- **The `lama` term is specific to the mask's edge.** This is the main Phase 2 finding. A lama-only
  probe on 3–4 Sintel/smoke cases (`removed_psnr_logo`, lower = more broken fill):
  - A δ trained on the 8 px box breaks that box (5–10 dB). The 24 px box and the SAM mask on the
    same frame get clean fills (25–35 dB).
  - Training on boxes 4/20/36 px broke nothing, including the 8 and 24 px boxes. **A mask edge
    4 px off is enough to escape.** The δ that breaks LaMa sits right at the hole's edge.
  - Masks a few px apart (4/8/16 + SAM) fight over the same pixels and break nothing, even
    summed on every call at 4× the cost. Masks 16 px apart (8 and 24) coexist. Hence the default.
  - Cycling masks without momentum broke nothing; with momentum it costs the same as one mask.
  - No setup broke the 40 px box, and a SAM mask in the bank broke SAM-masked fills only weakly
    (24 dB at best) while diluting the boxes.
  - In the full eval: `oracle-lama` is defended a bit less than in Phase 1 (11 vs 8 removed),
    and `loose-lama` gains the same (18 → 12). `sam-lama` is still untouched.
- **So `lama` defends against masks we can predict, not against a hand-drawn one.** The real
  tools' masks are predictable: WatermarkRemover-AI inpaints Florence-2's box *undilated*, and
  IOPaint's click-to-mask dilates SAM's mask by ~4 px. Our harness dilates both by 8 px, so
  its `florence-lama`/`sam-lama` lama numbers don't transfer one to one (eval/README).
- **Grid 1 for medium doesn't pay** (`runs/p2-medium-grid1`): `florence-lama` 17 → 9 under H.264
  (grid 2: 7), `oracle-lama` 19 → 13 (11), and it costs more: PSNR 42.9 dB, ring SSIM 0.80
  (grid 2: 0.90). Medium stays at grid 2. Phase 1's sweep, where grid 1 flipped the decoy on
  sintel03, predates momentum.
- **Codec calibration** (eval/README → "Calibrating the codec proxy"): a lone frame at CRF 23
  (eval's `h264:23`) erases about twice as much of a static δ as a clip at CRF 23. Eval's H.264
  numbers are a pessimistic bound.
- **`sam` with frame chunking:** 3 frames of a 430×240 crop take ~1 s per step at 6.9 GiB peak
  (1 frame: 6.8 GiB). Phase 0 had 24.5 GiB for 2 frames. Not yet run in a full eval.
- **SLBR has no licence**, so the optional `blind_remover` surrogate is off the table (§8).

### Phase 1 results (2026-10-06)

**Florence-2 is affordable now.** It took four changes:
- It loses the 41 s/step: that was VRAM spill, not compute. One fwd+bwd is ~150 ms on the 5090,
  but all 8 frame × prompt × EOT graphs were held at once, ~3.5 GiB each.
- pgd backpropagates per EOT sample and per surrogate (`attack.SumLoss`).
- The vision tower runs once per frame and is shared by both prompts.
- `florence2` and `lama` both backpropagate in chunks of 2 frames.

Measured on the largest Phase 0 crop (976×418), `lama,florence2`, medium preset:

| frames per group | s / PGD step | ≈ medium (50 steps) | peak VRAM (alloc / reserved) |
|---|---|---|---|
| 1 | 0.70 | 35 s | 7.5 / 8.9 GiB |
| 3 | 1.34 | 67 s | 11.5 / 14.5 GiB |
| 8 | 2.88 | 144 s | 11.9 / 15.1 GiB |

- Memory is bounded at any group size. Time is not: 3 frames × several stays won't fit in
  the ~120 s slice §5 gives fortify. That's Phase 3's problem (fewer steps, alternating
  surrogates, Florence-2-base, or async jobs).
- Over the 24 single-frame eval groups (smaller crops), shielding took 30 s on average and
  43 s at most.

**What made `florence2` work at all.** These were fixed before the eval; see §3.5 for the
mechanism:
1. **Untargeted ascent (the first draft) moves boxes and can make them better for the
   attacker.** On sintel03's crop view, Florence-2 boxed only the "S" on the clean frame; on the
   shielded one, it boxed the whole wordmark. → targeted decoy loss.
2. **The crop view doesn't transfer.** A florence2-only δ that wrecked the CE on the crop
   left full-frame detection untouched (same box, coverage 1.0), even without a codec. → the
   View: the attacker's whole-frame picture, rebuilt from a thumbnail.
3. **Without balancing, Florence-2's gradient drowned LaMa's.** In a joint run, lama reached
   −0.0002, against −0.86 alone. → gradients balanced per surrogate.

A budget sweep for the decoy (sintel03 and kodim05, real detector, `florence2` alone):
- **Sintel:** ε = 8, grid 2 (medium) gets the decoy CE down but doesn't flip the beam-search
  answer.
  - ε = 8 at grid 1 with 100 steps, ε = 12 with EOT, and ε = 16 all flip it to the decoy, and the
    flip survives x264 CRF 23.
  - At 1920 px, Florence-2's own 2.5× downscale already low-passes δ, so grid 2 is an extra
    handicap there.
- **kodim05:** nothing at ε = 8 flipped it. The crop is small (214×120) and isn't downscaled.

**Eval:** `eval/run.py --shield medium --surrogates lama,florence2 --codec h264:23,none`.
- All 24 Phase 0 cases, single frame, with a View (1024 px thumbnail).
- Output in `runs/p1-medium`; `_sheet.png` there is a contact sheet of restored crops.
- Shield cost: PSNR 46.2 dB on average, 40.2 dB at worst (large logos, whose crops are big).

Cases out of 24 (marked → shielded). Removal counts as successful when `removal_score` > 0.5,
because a broken fill reads as a large negative score. Mask moves are cases where the
remover's mask went from covering the logo (≥ 0.5) to missing it, or back.

| attacker | h264:23 removed | no codec removed | mask moved off / onto the logo (h264:23) |
|---|---|---|---|
| `oracle-lama` (hand box) | 19 → **8** | 21 → 6 | — |
| `florence-lama` (WatermarkRemover-style) | 17 → 15 | 17 → 11 | **4 off / 3 onto** (no codec: 8 / 2) |
| `sam-lama` (box prompt → SAM mask) | 21 → 19 | 21 → 19 | 0 / 0 |

What the numbers and images say:
- **`lama` is the strong term, and it survives H.264.**
  - With the attacker's own hand-drawn box, LaMa's fill breaks into a flat bright or dark slab
    where the logo was: sintel03 ×2, sintel12-blur, sintel26-glass, sintel37-glass, sintel40.
  - That's the first thing in this project that hurts the hand-mask attacker.
  - It works well on Sintel and smoke, and barely at all on Kodak photos (kodim05, kodim19,
    kodim20: fill still clean). The lama loss there stays at −0.03 to −0.2, against −0.4 to −0.8 on
    Sintel.
  - On plain marks `removal_score` stays high even when the fill is broken, because the white
    logo makes the baseline error huge. Check `removed_psnr_logo` (e.g. sintel22-plain-large
    38 → 13 dB) or the sheet.
- **The lama δ is mask-specific.** `sam-lama` runs the same LaMa with a stroke-shaped SAM mask
  (dilated 8 px) instead of the box, and is untouched. The surrogate only ever trained on
  the dilated box. → EOT over mask shapes (Phase 2).
- **`florence2` works in both directions.**
  - Where it works, Florence-2 boxes the decoy strip and the mark survives: gradient-blur,
    sintel26 ×2 and sintel40 under H.264; 8 cases without a codec.
  - On three large logos (kodim13, sintel19, sintel22-plain-large) it backfires. Florence-2 on
    the clean frame boxed only the "S" (coverage 0.24 to 0.27), the decoy target replaced that
    box, and the attack landed on the whole wordmark instead (coverage 1.0). The same failure as
    untargeted ascent.
  - H.264 halves its effect: 8 wins without a codec, 4 with.
  - At ε = 8, medium is below the budget the decoy needs (see the sweep above).
- **The decoy is visible.** It paints a faint ghost of the wordmark into the decoy strip. It shows in
  the delivered frame on flat or dark backgrounds (smoke4, sintel12-glass, sintel26-glass, the
  sintel03 CLI run), despite 46 dB PSNR. PSNR is the wrong cost metric for a structured δ.

### Phase 0 results (2026-10-06)

**Setup**
- **Cases:** 24 frames from `eval/export_cases.ts`, through videotools' real Mark graph.
  - Filters: plain ×7, glass ×10, blur ×7.
  - Sizes: small ×6, medium ×11, large ×7.
  - Backgrounds: Sintel (1920×816), Kodak photos (768×512, stills), videotools' smoke assets.
  - Logo: the smoke `logo.png`, a white "S + GRAZ" wordmark.
- **Delivery:** every frame went through x264 CRF 23 before removal. CRF 28 gave the same
  picture.
- **"Removed"** means `mark_residual` < 0.2, checked against the images. `removal_score`
  under-reads faint marks; see eval/README "Metrics".
- **Outputs:** `runs/p0-h264-23`, `runs/p0-h264-28`, `runs/p0-wild` (hand-run tools) and
  `runs/p0-video`.

**Frames: cases removed out of 24**

| attacker | mask from | removed | what's left |
|---|---|---|---|
| `oracle-lama`, `sam-lama`, IOPaint LaMa / MAT | a box drawn or dragged by hand | 21–22 | nothing visible; the misses are blur-metric artefacts (kodim08, kodim15). Visually 24/24 |
| IOPaint SD 1.5 inpainting, empty prompt | same box | 17 | the mark is gone, but on dark backgrounds SD paints sign-like junk in the hole (a re-roll fixes it) |
| WatermarkRemover-AI, default (Florence-2 "watermark" → LaMa) | automatic | 18 | large logos (Florence-2 boxes only the "S" symbol, not "GRAZ") and faint blur |
| WatermarkRemover-AI, run twice | automatic | 21 | kodim15 blur-large over a face still partly visible |
| WatermarkRemover-AI `--mask-mode stroke --double-pass` | automatic | 17 | worse on blur: the stroke mask misses faint strokes |
| `florence-lama` (harness) | automatic | 17 | same failure modes as WatermarkRemover-AI |

**Clips: share of frames with the mark still visible**

ProPainter ran at `--resize_ratio 0.5`, pasted back inside the mask.

| clip (CRF 23) | WatermarkRemover-AI video (detect every frame) | SAM2 box on frame 0, propagated → ProPainter | hand mask per frame → ProPainter |
|---|---|---|---|
| Sintel desert, glass, static | 21% ¹ | 0% (SAM2 covers 98% of the mark) | 0% |
| Sintel desert, glass, rotating | 24% ¹ | **65%** (SAM2 covers 33%: it loses the mark at the first jump) | 0% ² |
| smoke, plain, static | 0% | 0% (covers 100%) | 0% |
| smoke, plain, rotating | 1% | **49%** (covers 50%) | 0% |

¹ The last 1.2 s only. When the trailer's "SINTEL" title fades in, Florence-2 removes the title
  instead of GRAZ. That's competing on-screen text, not anything about Mark.
² The metric flags 19% of frames, but those are ProPainter fill blotches as the scene darkens.
  No mark is left.

**Answer to the Phase 0 question**
- **Glass refraction resists nothing.** Glass came off as cleanly as plain in every pipeline
  (10/10 with a hand mask). Florence-2 still detects it, and inpainters fill from the ring, so
  the refraction never matters. Don't count glass as protection.
- **Blur resists automatic detection only, and only partly.** Faint blur marks over texture
  (kodim08 small, kodim15 large) slip past Florence-2 and stroke masks. With a hand box they come
  off like the rest.
- **Large size beats Florence-2 by accident.** On large wordmarks, Florence-2 boxes the symbol
  and leaves the text. A second pass of the same tool fixes it.
- **Rotation resists only "click once and propagate" video tools.** SAM2 tracks a static mark
  through the clip, but loses it at the first jump. The attacker has to re-prompt every stay
  (2 s), which is a real cost on long clips. Rotation does nothing against per-frame detection
  (WatermarkRemover-AI's video mode) or per-frame masks: ProPainter then removes it completely.
- **Compression doesn't help.** CRF 28 vs 23 changed nothing.

**What this means for fortify's targets**
- **Hand-mask attackers** remove every Mark variant today. Only an inpainter-disrupting term
  (`lama`, held-out MAT/SD for transfer) can raise their cost; detection suppression can't.
- **Florence-2 is the gate for the common automatic tool**, and it already wobbles on large and
  blur marks. Suppressing it is high leverage, so making the `florence2` surrogate affordable
  (see the bottleneck note below) was Phase 1's first job.
- **`sam` matters for static marks.** Rotating output already defeats single-prompt SAM2
  propagation. Phase 2 should prioritise `sam` for non-rotating jobs, and per stay (the first
  frames of each stay are where an attacker re-prompts).
- **Limits of this baseline:**
  - one logo;
  - CGI and low-res photo backgrounds, no live action;
  - SD with an empty prompt only;
  - ProPainter at half resolution;
  - two clips.

Findings and known issues:
- **"Disrupting" losses need a random start.** The `lama`/`toy` losses compare against
  the surrogate's own clean output, so their gradient at δ = 0 is exactly 0. A
  zero-start PGD or FGSM never moves (found and fixed: every preset has `random_start`,
  and `low` is R+FGSM). Keep this in mind for any new DWV-style surrogate.
- **Florence-2 was the bottleneck: fixed in Phase 1.** Per-sample and per-surrogate backward,
  a shared vision tower and 2-frame chunks took it from 41 s/step and 34 GiB to ~0.2 s per
  frame and step. See [Phase 1 results](#phase-1-results-2026-10-06).
  - bf16 was tried and rejected (§3.5).
  - Still untried, if time per group becomes the limit: one prompt instead of two;
    Florence-2-base; alternating which surrogates get a step.
- **Florence-2 must come from the native transformers 5 port.** The original
  `microsoft/Florence-2-*` remote code crashes on transformers 5.x
  (`Florence2LanguageConfig ... forced_bos_token_id`), so `models.py` loads
  `Florence2ForConditionalGeneration` from `florence-community/Florence-2-large`.
- **SAM2 memory:** 24.5 GiB with 2 frames × 2 EOT samples at 1024², measured before Phase 1.
  pgd now backpropagates per EOT sample and per surrogate, which should about halve that. `sam`
  still has no frame chunking; give it a `chunk` like `lama`/`florence2` in Phase 2.
- **Download stalls:** Hugging Face's Xet CDN failed once mid-download. Setting
  `HF_HUB_DISABLE_XET=1` fixed it.
- SAM2 load prints a harmless "`sam2_video` to instantiate `sam2`" notice.
- `eval/removers.py` `florence_lama` works on the native Florence-2 processor
  (`post_process_generation` parses boxes; verified in Phase 0).
- **videotools quirk (not fixed there):** a plain mark on yuv420p lands 1 px up/left of
  `watermarkLayout`'s corner when that corner is odd, because overlay floors to even.
  Glass and blur cells are even by construction.

- **Florence-2 label alignment: verified in Phase 1.** Teacher-forced on clean, the logits'
  argmax reproduces every `<loc_*>` token of the generated answer, except where beam search and
  greedy disagree on one bin. The hand-built forward (shared image features) matches the
  model's own `labels=` loss to 1e-3.

Still unverified:
- The DCT-JPEG proxy's quality range vs real x264.
- The ensemble weights. Gradients are balanced now, but 1:1 is a guess.

## Roadmap
- **Phase 0, baseline red-team: done 2026-10-06, results in [Status](#phase-0-results-2026-10-06).**
  - Export about 20 cases from videotools (plain, glass and blur × sizes × backgrounds)
    and run `eval/run.py` without `--shield`.
  - Also run WatermarkRemover-AI, IOPaint (LaMa/MAT/SD) and SAM2 + ProPainter by hand.
  - Question: how much do Mark's rotation and glass refraction already resist removal?
    The answer may change what fortify needs to target.
- **Phase 1, single-frame shield: done 2026-10-06, results in [Status](#phase-1-results-2026-10-06).**
  - `lama` + `florence2` end to end through the CLI, with ε = 8 (medium) compared before and
    after `h264:23`. `jpeg:75` was not run.
  - Added on the way: the targeted decoy loss, the View (API field `view`), balanced ensemble
    gradients, and per-sample/per-surrogate/per-chunk backward.
- **Phase 2, robustness: first pass done 2026-10-06, results in [Status](#phase-2-results-2026-10-06).**
  - Done:
    - `lama` over mask shapes, with momentum; default bank box 8 + 24 px. It defends exactly the
      masks it trains on, not a hand-drawn one.
    - The decoy no longer backfires on large logos (partial boxes kept, whole-logo hinge).
    - Ring SSIM as a cost metric.
    - Grid 1 for medium was tested and rejected.
    - The codec proxy is calibrated.
    - `sam` has frame chunking.
    - SLBR was checked and has no licence.
  - Left, in order:
    - **Match the real tools' masks.** Add `box:0` (WatermarkRemover-AI inpaints Florence-2's box
      undilated) to the bank, and make the harness's `florence-lama` and `sam-lama` dilate like
      the real tools (0 px and ~4 px). Then re-run WatermarkRemover-AI and IOPaint by hand on
      shielded frames.
    - **The decoy's visibility.** Ring SSIM is 0.90 on average and 0.78 at worst. Try a smaller
      or textured decoy, or put the SSIM term into the loss. Eyeball the worst cases first.
    - **`sam` in a full eval**, and the ensemble weights. Static marks are where SAM2 propagation
      works (Phase 0).
    - Whether `high` (ε = 12) is acceptable visually.
    - A per-stay universal δ over sampled frames, scored on whole clips. This would also check
      the calibration finding that delivery keeps more δ than eval's lone frames.
    - Hold out MAT/SD inpainting (IOPaint, by hand) to measure transfer. Eval needs to save the
      delivered shielded frames for that.
- **Phase 3, service.** Deploy (Modal or self-hosted 5090), latency measurements, a
  timing-aware strength choice. Phase 1 timing is ~35 s per 1-frame group and ~67 s per 3-frame
  group (largest crop, medium), so a multi-stay job needs fewer steps or async jobs.
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
