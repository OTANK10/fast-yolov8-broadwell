#!/usr/bin/env python3
"""FLOPs/cycle vs k for the isolated stride-1 microkernels (Assignment 2 style).

    python scripts/plot_ukr.py results/ukr_sweep.csv [-o results/ukr_sweep.png]

One line per tile: the best of the runs solid, the other runs faint. Dashed
line = peak (32 FLOPs/cycle). Dotted lines = where the first tile's working
set (its weights + the input rows it touches) outgrows L1 (32 KB) and L2.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--peak", type=float, default=32.0)
ap.add_argument("--machine", default="ece028")
ap.add_argument("--tiles", default=None, help="comma list of tiles to plot (default: all in the CSV)")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv)))
if a.tiles:
    keep = set(a.tiles.split(","))
    rows = [r for r in rows if r["tile"] in keep]
tiles = list(dict.fromkeys(r["tile"] for r in rows))
by = defaultdict(lambda: defaultdict(dict))  # tile -> run -> k -> fpc
ws = defaultdict(dict)                         # tile -> k -> working set KB
for r in rows:
    by[r["tile"]][int(r["run"])][int(r["k"])] = float(r["flops_per_cycle"])
    ws[r["tile"]][int(r["k"])] = float(r["working_set_kb"])
src = rows[0]["cycle_src"]
silu = rows[0]["silu"] == "1"

fig, ax = plt.subplots(figsize=(9, 4.8))
colors = ["#0C6A87", "#B7731A", "#2C7A4F", "#8A3FA0", "#B4442E", "#566573"]
for i, t in enumerate(tiles):
    runs = by[t]
    ks = sorted(next(iter(runs.values())).keys())
    best_run = max(runs, key=lambda rn: sum(runs[rn].values()))
    for rn, pts in runs.items():
        if rn != best_run:
            ax.plot(ks, [pts[k] for k in ks], color=colors[i % len(colors)], alpha=0.25, lw=1)
    note = next(r["note"] for r in rows if r["tile"] == t)
    ax.plot(ks, [runs[best_run][k] for k in ks], color=colors[i % len(colors)], marker="o", ms=3, lw=1.8,
            label=f"{t}: {note}, best of {len(runs)}")

ax.axhline(a.peak, color="#444", ls="--", lw=1)
ax.text(ax.get_xlim()[0], a.peak + 0.4, f"  peak ({a.peak:.0f})", fontsize=9, color="#444")
t0 = tiles[0]
for cap, name in ((32, "L1 32 KB"), (256, "L2 256 KB")):
    over = [k for k in sorted(ws[t0]) if ws[t0][k] > cap]
    if over:
        ax.axvline(over[0], color="#B4442E", ls=":", lw=1.2)
        ax.text(over[0], 2, f" {t0} working set > {name}", color="#B4442E", fontsize=8.5, rotation=90, va="bottom")
ax.set_xlabel("k = C x 9 (reduction length: input channels x 3x3 taps)")
ax.set_ylabel("FLOPs per cycle")
ymax = max(float(r["flops_per_cycle"]) for r in rows)
ax.set_ylim(0, max(a.peak * 1.12, ymax * 1.05))
ax.grid(alpha=0.3)
ax.set_title(f"Stride-1 microkernel alone on {a.machine} (1 core, {src} cycles, SiLU {'on' if silu else 'off'})", fontsize=11)
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
out = a.out or a.csv.rsplit(".", 1)[0] + ".png"
fig.savefig(out, dpi=160)
print("wrote", out)

# summary like the assignment discussion: best point and plateau per tile
for t in tiles:
    runs = by[t]
    allpts = [(fpc, k, rn) for rn, pts in runs.items() for k, fpc in pts.items()]
    fpc, k, rn = max(allpts)
    print(f"{t:8s} best {fpc:5.2f} FLOPs/cycle = {100*fpc/a.peak:4.1f}% of peak at k = {k} (run {rn})")
