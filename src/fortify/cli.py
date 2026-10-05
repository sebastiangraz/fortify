"""fortify CLI: `fortify selftest | vaccinate | serve`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _rect(text: str):
    from .region import Rect

    x, y, w, h = (int(v) for v in text.split(","))
    return Rect(x, y, w, h)


def cmd_selftest(args: argparse.Namespace) -> int:
    """Check torch/CUDA and that PGD moves a weight-free toy remover. No downloads."""
    from .models import device
    from .region import Rect
    from .surrogates import ensemble
    from .vaccinate import Group, vaccinate

    dev = device()
    print(f"torch {torch.__version__}, device {dev}", end="")
    if dev.type == "cuda":
        print(
            f" ({torch.cuda.get_device_name()}, capability {torch.cuda.get_device_capability()})",
            end="",
        )
    print()
    frames = torch.rand((3, 3, 160, 256), generator=torch.Generator().manual_seed(0))
    group = Group(frames, Rect(64, 40, 128, 80), "selftest")
    for strength in ("low", "medium", "high"):
        r = vaccinate(group, ensemble("toy"), strength)
        levels = int((r.delta.abs().max() * 255).round())
        print(
            f"{strength:>6}: loss {r.trace[0]:.5f} → {r.trace[-1]:.5f}, max |δ| {levels}/255, {r.ms:.0f} ms"
        )
    zero = vaccinate(group, ensemble("null"), "high")
    print(f"  null: max |δ| {float(zero.delta.abs().max())} (must be 0.0)")
    return 0


def cmd_vaccinate(args: argparse.Namespace) -> int:
    from .attack import apply_delta
    from .imageio import encode_delta, load_png, save_png
    from .surrogates import ensemble
    from .vaccinate import Group, vaccinate

    frames = torch.cat([load_png(Path(f).read_bytes()) for f in args.frames])
    model = ensemble(args.surrogates)
    r = vaccinate(Group(frames, args.logo, "cli"), model, args.strength, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "delta.png").write_bytes(encode_delta(r.delta))
    # δ amplified ×8 around mid-grey, for eyeballing where the energy went.
    (out / "delta_x8.png").write_bytes(save_png((r.delta * 8 + 0.5).clamp(0, 1)))
    for i, f in enumerate(args.frames):
        (out / f"shielded_{i}_{Path(f).stem}.png").write_bytes(
            save_png(apply_delta(frames[i : i + 1], r.delta))
        )
    print(
        f"{args.strength}: {len(args.frames)} frame(s), loss {r.trace[0]:.5f} → {r.trace[-1]:.5f}, {r.ms:.0f} ms → {out}"
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("fortify.service:app", host=args.host, port=args.port, workers=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fortify")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "selftest", help="check torch/CUDA and the attack loop (no weights)"
    ).set_defaults(fn=cmd_selftest)

    v = sub.add_parser("vaccinate", help="compute one δ for a group of crops")
    v.add_argument(
        "--frames", nargs="+", required=True, help="PNG crops of the watermarked frames (same size)"
    )
    v.add_argument("--logo", type=_rect, required=True, help="x,y,w,h of the mark inside the crop")
    v.add_argument("--strength", choices=("low", "medium", "high"), default="medium")
    v.add_argument("--surrogates", default="lama,sam,florence2")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", default=f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    v.set_defaults(fn=cmd_vaccinate)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
