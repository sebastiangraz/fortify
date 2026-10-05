// Export red-team cases from videotools' Mark graph (Phase 0). Bun script.
//
//   bun eval/export_cases.ts [--videotools ../videotools] [--src eval/data/_sources] [--out eval/data]
//   bun eval/export_cases.ts --clips      # also the rotating clips for the video removers
//
// Per case: clean.png and marked.png from the same decoded frame, run through the same
// format chain so they differ only where Mark draws. logo.json is the logo box (the
// layout's LX/LY/LW/LH); meta.json adds the glass/blur cell and the case's settings.
// Sources (see eval/README.md, "Getting cases out of videotools"): videotools' smoke
// assets, the Sintel trailer (CC-BY, Blender Foundation) and Kodak PhotoCD images.

import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    videotools: { type: "string", default: "../videotools" },
    src: { type: "string", default: "eval/data/_sources" },
    out: { type: "string", default: "eval/data" },
    clips: { type: "boolean", default: false },
    only: { type: "string" },
  },
});
const VT = path.resolve(args.videotools);
const FF = process.env.FFMPEG || path.join(VT, "api/_bin/ffmpeg/win32-x64/ffmpeg.exe");
const { watermarkGraph, watermarkLayout, parseBounds } = await import(
  path.join(VT, "api/_lib/tools/mark-graph.ts")
);
const { parseSourceProfile } = await import(path.join(VT, "api/_lib/ffmpeg.ts"));

type Filter = "plain" | "glass" | "blur";
type Size = "small" | "medium" | "large";
type Case = { clip: string; file: string; t?: number; filter: Filter; size: Size; pos: string };

const SMOKE = path.join(VT, "scripts/smoke-assets");
const LOGO = path.join(SMOKE, "logo.png");
const src = (f: string) => (path.isAbsolute(f) ? f : path.join(args.src, f));

// clip, file, time (video only), filter, size, position. Plain ×7, glass ×10, blur ×7;
// small ×6, medium ×11, large ×7; flat, textured, faces, sky, text, graphics.
const CASES: Case[] = [
  ["sintel03", "sintel.mp4", 3, "plain", "medium", "bottom-right"],
  ["sintel03", "sintel.mp4", 3, "glass", "large", "center"],
  ["sintel12", "sintel.mp4", 12.5, "glass", "medium", "bottom-right"],
  ["sintel12", "sintel.mp4", 12.5, "blur", "medium", "top-left"],
  ["sintel14", "sintel.mp4", 14, "plain", "small", "top-right"],
  ["sintel14", "sintel.mp4", 14, "glass", "small", "center"],
  ["sintel19", "sintel.mp4", 19, "blur", "large", "center"],
  ["sintel22", "sintel.mp4", 22, "glass", "medium", "top-left"],
  ["sintel22", "sintel.mp4", 22, "plain", "large", "bottom"],
  ["sintel26", "sintel.mp4", 26, "blur", "small", "top-right"],
  ["sintel26", "sintel.mp4", 26, "glass", "medium", "right"],
  ["sintel37", "sintel.mp4", 37, "plain", "medium", "bottom-left"],
  ["sintel37", "sintel.mp4", 37, "glass", "large", "top"],
  ["sintel40", "sintel.mp4", 40, "blur", "medium", "center"],
  ["smoke1", path.join(SMOKE, "video.mp4"), 1, "glass", "medium", "bottom-right"],
  ["smoke4", path.join(SMOKE, "video.mp4"), 4, "plain", "small", "left"],
  ["gradient", path.join(SMOKE, "images/image1.png"), undefined, "blur", "medium", "bottom-right"],
  ["kodim04", "kodim04.png", undefined, "glass", "medium", "center"],
  ["kodim05", "kodim05.png", undefined, "plain", "medium", "bottom-right"],
  ["kodim08", "kodim08.png", undefined, "blur", "small", "top-left"],
  ["kodim13", "kodim13.png", undefined, "glass", "large", "bottom-right"],
  ["kodim15", "kodim15.png", undefined, "blur", "large", "center"],
  ["kodim19", "kodim19.png", undefined, "glass", "small", "top-right"],
  ["kodim20", "kodim20.png", undefined, "plain", "large", "bottom"],
].map(([clip, file, t, filter, size, pos]) => ({ clip, file, t, filter, size, pos }) as Case);

// Clips for the video removers (WatermarkRemover-AI video mode, SAM2 + ProPainter), each
// marked twice: static at bottom-right, and rotating (2 s stays from bottom-right on).
const CLIPS = [
  { clip: "sintel-desert", file: "sintel.mp4", t: 35.5, d: 6, filter: "glass" },
  { clip: "smoke", file: path.join(SMOKE, "video.mp4"), t: 0, d: 5, filter: "plain" },
] as const;

const ff = (a: string[]) => execFileSync(FF, ["-v", "error", "-y", ...a], { stdio: "inherit" });
const probe = (file: string) =>
  parseSourceProfile(spawnSync(FF, ["-hide_banner", "-i", file], { encoding: "utf8" }).stderr);

// The trailer is letterboxed (2.35:1 in 1920×1080): crop the bars once, near-lossless, so
// marks land on picture rather than flat black.
const sintel = src("sintel.mp4");
if (!fs.existsSync(sintel)) {
  ff(["-i", src("sintel_trailer-1080p.mp4"), "-an", "-vf", "crop=1920:816:0:132", "-c:v", "libx264",
    "-crf", "8", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
    "-color_trc", "bt709", sintel]);
}

const logo = probe(LOGO);
const bbox = spawnSync(
  FF,
  ["-hide_banner", "-i", LOGO, "-vf", "format=rgba,alphaextract,bbox=min_val=16", "-f", "null", "-"],
  { encoding: "utf8" },
).stderr;
const bounds = parseBounds(bbox, logo.width, logo.height);

for (const c of CASES) {
  const name = `${c.clip}-${c.filter}-${c.size}-${c.pos}`;
  if (args.only && !name.includes(args.only)) continue;
  const file = src(c.file);
  const still = c.t === undefined;
  const info = probe(file);
  // Mark keeps stills RGBA and videos yuv420p (mark.ts, addWatermark).
  const base = still ? "rgba" : "yuv420p";
  const L = watermarkLayout(info, bounds, c.size, c.pos);
  const graph = watermarkGraph(info, logo, {
    filter: c.filter,
    bounds,
    base,
    size: c.size,
    position: c.pos,
  });
  // Same opening as the graph's [base] chain, so clean and marked share every conversion.
  const open = still ? "format=rgba" : `format=yuv420p,crop=${L.VW}:${L.VH}:0:0`;
  // Both to RGB with one explicit matrix: the glass cell comes back tagged, and an untagged
  // clean frame would otherwise convert with another matrix and differ everywhere.
  const matrix = info.matrix ?? (L.VH >= 720 ? "bt709" : "bt601");
  const rgb = still ? "format=rgb24" : `scale=in_color_matrix=${matrix}:in_range=tv,format=rgb24`;
  const seek = still ? [] : ["-ss", String(c.t)];
  const dir = path.join(args.out, name);
  fs.mkdirSync(dir, { recursive: true });
  const toPng = ["-frames:v", "1", "-pix_fmt", "rgb24"];
  ff([...seek, "-i", file, "-vf", `${open},${rgb}`, ...toPng, path.join(dir, "clean.png")]);
  ff([
    ...seek,
    "-i",
    file,
    "-i",
    LOGO,
    "-filter_complex",
    `${graph};[out]${rgb}[rgb]`,
    "-map",
    "[rgb]",
    ...toPng,
    path.join(dir, "marked.png"),
  ]);
  const pad = c.filter === "plain" ? 0 : L.margin;
  // overlay onto yuv420p floors x/y to even, so an odd plain corner lands 1 px up/left.
  const snap = (n: number) => (c.filter === "plain" && !still ? n & ~1 : n);
  fs.writeFileSync(
    path.join(dir, "logo.json"),
    JSON.stringify({ x: snap(L.LX), y: snap(L.LY), w: L.LW, h: L.LH }) + "\n",
  );
  fs.writeFileSync(
    path.join(dir, "meta.json"),
    JSON.stringify(
      {
        source: path.basename(file),
        t: c.t ?? null,
        still,
        filter: c.filter,
        size: c.size,
        position: c.pos,
        frame: { w: L.VW, h: L.VH },
        cell: { x: snap(L.LX) - pad, y: snap(L.LY) - pad, w: L.LW + 2 * pad, h: L.LH + 2 * pad },
      },
      null,
      2,
    ) + "\n",
  );
  console.log(name);
}

if (args.clips) {
  const clipDir = path.join(args.out, "_clips");
  fs.mkdirSync(clipDir, { recursive: true });
  for (const c of CLIPS) {
    const file = src(c.file);
    // A clip of its own, so the graph's duration and stays match what Mark would see.
    const cut = path.join(clipDir, `${c.clip}-clean.mp4`);
    ff(["-ss", String(c.t), "-t", String(c.d), "-i", file, "-an", "-c:v", "libx264", "-crf", "12",
      "-pix_fmt", "yuv420p", cut]);
    const info = probe(cut);
    for (const rotatePosition of [false, true]) {
      const graph = watermarkGraph(info, logo, {
        filter: c.filter,
        bounds,
        size: "medium",
        position: "bottom-right",
        rotatePosition,
      });
      const out = path.join(clipDir, `${c.clip}-${c.filter}-${rotatePosition ? "rotating" : "static"}.mp4`);
      ff(["-i", cut, "-i", LOGO, "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264",
        "-crf", "23", "-pix_fmt", "yuv420p", out]);
      // Lossless twin: minus the clean cut it is exactly the mark (video_masks.py's truth).
      ff(["-i", cut, "-i", LOGO, "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264",
        "-qp", "0", "-pix_fmt", "yuv420p", out.replace(/\.mp4$/, ".truth.mp4")]);
      console.log(out);
    }
  }
}
