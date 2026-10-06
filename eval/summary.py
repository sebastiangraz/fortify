"""Summarise eval/run.py results the way the README's tables do.

Usage:
  uv run python eval/summary.py runs/p1-medium runs/p2-medium [...]

Per run, per codec and remover: cases removed (removal_score > 0.5) on marked → shielded
frames, and mask moves (logo_covered went from ≥ 0.5 to < 0.5: "off", or back: "onto").
Then the shield's cost and time. A broken fill reads as a large negative removal_score,
so "> 0.5" counts clean removals only; plain marks can still read high with a broken fill
(see eval/README "Metrics"), so look at the images too.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REMOVED = 0.5
COVERED = 0.5


def summarise(run: Path) -> None:
    with open(run / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    shielded = sorted({r["variant"] for r in rows} - {"marked"})
    by = {(r["case"], r["variant"], r["codec"], r["remover"]): r for r in rows}
    cases = sorted({r["case"] for r in rows})
    print(f"\n## {run}  ({len(cases)} cases)")
    for variant in shielded:
        print(f"\n{variant}")
        print(f"| {'remover':<14} | {'codec':<8} | removed marked → shielded | mask off / onto |")
        print(f"|{'-' * 16}|{'-' * 10}|{'-' * 27}|{'-' * 17}|")
        combos = sorted({(r["codec"], r["remover"]) for r in rows}, key=lambda c: (c[1], c[0]))
        for codec, remover in combos:
            n_marked = n_shield = off = onto = 0
            covered = False
            for case in cases:
                m, s = (
                    by.get((case, "marked", codec, remover)),
                    by.get((case, variant, codec, remover)),
                )
                if not m or not s:
                    continue
                n_marked += float(m["removal_score"]) > REMOVED
                n_shield += float(s["removal_score"]) > REMOVED
                if m.get("logo_covered"):
                    covered = True
                    before, after = float(m["logo_covered"]), float(s["logo_covered"])
                    off += before >= COVERED > after
                    onto += after >= COVERED > before
            moves = f"{off} / {onto}" if covered else "—"
            print(f"| {remover:<14} | {codec:<8} | {n_marked:>2} → {n_shield:<21} | {moves:<15} |")

        cost = defaultdict(list)
        for case in cases:
            r = next((r for k, r in by.items() if k[0] == case and k[1] == variant), None)
            if r:
                cost["psnr"].append(float(r["shield_cost_psnr"]))
                cost["ms"].append(float(r["shield_ms"]))
                if r.get("shield_cost_ssim_ring"):
                    cost["ssim"].append((float(r["shield_cost_ssim_ring"]), case))
        psnr, ms = cost["psnr"], cost["ms"]
        print(
            f"cost: PSNR {sum(psnr) / len(psnr):.1f} dB mean, {min(psnr):.1f} worst; "
            f"time {sum(ms) / len(ms) / 1000:.0f} s mean, {max(ms) / 1000:.0f} s max"
        )
        if cost["ssim"]:
            ssim = sorted(cost["ssim"])
            mean = sum(v for v, _ in ssim) / len(ssim)
            worst = ", ".join(f"{c} {v:.3f}" for v, c in ssim[:3])
            print(f"ring SSIM {mean:.3f} mean; worst: {worst}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run in sys.argv[1:]:
        summarise(Path(run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
