#!/usr/bin/env python3
"""Design space of every stride-1 kernel run on ece028: % of peak vs register tile.

    python scripts/plot_design_space.py [-o results/ukr_design_space.png]

x = register tile, ordered by accumulators. Filled markers = the microkernel
alone (bench_ukr, C = 64, best of 3); hollow = all 32 real layers (bench_layer
totals, or the 4-layer tile-sweep mean for the small tiles). Grey = intrinsics
(GCC allocates registers), blue = inline asm (registers fixed by hand). A line
joins the same kernel alone and in the layers. Every number is read from
results/*.csv except 2x5asm alone, a single ece028 run (see ISOLATED_EXTRA).
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default="results/ukr_design_space.png")
a = ap.parse_args()
R = "results"


def layer_pct(name):
    p = f"{R}/{name}_b1_t1.csv"
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["stride"] == "1" and r["R"] == "3"]
    return 100 * sum(float(r["flops"]) for r in rows) / sum(float(r["best_cycles"]) for r in rows) / 32


def iso_pct(path, tile, C=64):
    if not os.path.exists(path):
        return None
    v = [float(r["pct_peak"]) for r in csv.DictReader(open(path)) if r["tile"] == tile and int(r["C"]) == C]
    return max(v) if v else None


ISOLATED_EXTRA = {"2x5asm": 86.7}  # bench_ukr 64 --tiles 2x5,2x5asm,2x6,2x6asm on ece028, 2026-10-25, one run

sweep = defaultdict(list)
for r in csv.DictReader(open(f"{R}/s1_tile_sweep.csv")):
    sweep[r["tile"][1:]].append(float(r["pct_peak"]))
sweep = {k: sum(v) / len(v) for k, v in sweep.items()}

tiles = ["1x1", "1x2", "2x1", "1x4", "2x2", "1x5", "2x4", "1x10", "2x5", "2x6", "4x3"]
acc = {"1x1": 1, "1x2": 2, "2x1": 2, "1x4": 4, "2x2": 4, "1x5": 5, "2x4": 8, "1x10": 10, "2x5": 10, "2x6": 12, "4x3": 12}
X = {t: i for i, t in enumerate(tiles)}

# (tile, label, impl, in-layer %, isolated %)
pts = [
    ("2x5", "v2", "intr", layer_pct("s1_v2"), iso_pct(f"{R}/ukr_sweep_2x6.csv", "2x5")),
    ("2x6", "v4", "intr", layer_pct("s1_v4"), iso_pct(f"{R}/ukr_sweep_2x6.csv", "2x6")),
    ("4x3", "v8", "intr", layer_pct("s1_v8"), None),
    ("2x5", "v2a", "asm", layer_pct("s1_v2a"), ISOLATED_EXTRA["2x5asm"]),
    ("2x6", "v10", "asm", layer_pct("s1_v10"), iso_pct(f"{R}/ukr_sweep_2x6.csv", "2x6asm")),
    ("4x3", "v8a", "asm", layer_pct("s1_v8a"), iso_pct(f"{R}/ukr_sweep.csv", "4x3asm")),
]
lecture = [("v6 unroll 9 taps", layer_pct("s1_v6")), ("v6b unroll 3", layer_pct("s1_v6b")),
           ("v7 pipelined", layer_pct("s1_v7")), ("v7b", layer_pct("s1_v7b"))]

GREY, BLUE, LIGHT, AMBER = "#6B7782", "#0C6A87", "#B8C1C8", "#B7731A"
fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1.45]})

# ---- left: step 2, accumulators (in-layer tile sweep, intrinsics)
axl.plot([0, 100 / 10.1, 12.5], [0, 100, 100], color=AMBER, ls="--", lw=1.2)
axl.text(6.6, 14, "latency bound:\n~10% of peak\nper accumulator", color=AMBER, fontsize=8.5)
lab_off = {"1x2": (-14, 6), "2x1": (10, -10), "1x4": (-14, 6), "2x2": (10, -10), "2x4": (-14, 6), "1x10": (10, -10), "2x5": (-16, 8)}
for t in ["1x1", "1x2", "2x1", "1x4", "2x2", "1x5", "2x4", "1x10", "2x5"]:
    c = BLUE if t == "2x5" else GREY
    axl.scatter(acc[t], sweep[t], s=50, color=c, zorder=3)
    dx, dy = lab_off.get(t, (10, -4))
    axl.annotate(f"{t}: {sweep[t]:.0f}%", (acc[t], sweep[t]), xytext=(dx, dy), textcoords="offset points",
                 ha="right" if dx < 0 else "left", fontsize=8.5, color=c)
axl.axhline(100, color="#333", ls=":", lw=1)
axl.set_xlim(0, 12.5); axl.set_ylim(0, 108)
axl.set_xlabel("independent accumulators in the tile")
axl.set_ylabel("% of one core's peak (32 FLOPs/cycle)")
axl.set_title("1. How many accumulators: 10 (tile sweep, 4 layers)", fontsize=10.5, loc="left")
axl.grid(alpha=0.3)

# ---- right: the design space at >= 10 accumulators, alone vs in the layers
rows = [(f"{lab}: {t} {'asm' if impl == 'asm' else 'intrinsics'}", impl, lay, iso) for t, lab, impl, lay, iso in pts]
rows += [(f"{lab}: 2x5 intrinsics", "lect", v, None) for lab, v in lecture]
rows = sorted(rows, key=lambda r: -(r[3] if r[3] is not None else (r[2] or 0) - 100))
for y, (lab, impl, lay, iso) in enumerate(rows):
    c = BLUE if impl == "asm" else (LIGHT if impl == "lect" else GREY)
    if lay is not None and iso is not None:
        axr.plot([lay, iso], [y, y], color=c, lw=2, alpha=0.5, zorder=2)
    if iso is not None:
        axr.scatter(iso, y, s=70, color=c, zorder=4)
        axr.annotate(f"{iso:.1f}", (iso, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=c)
    if lay is not None:
        axr.scatter(lay, y, s=70, facecolors="white", edgecolors=c if impl != "lect" else GREY, linewidths=1.8, zorder=4)
        axr.annotate(f"{lay:.1f}", (lay, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.5,
                     color=c if impl != "lect" else GREY)
    elif iso is not None:
        axr.annotate("in layers: run pending", (iso - 1, y), xytext=(-8, 0), textcoords="offset points", ha="right",
                     va="center", fontsize=8, color=c, style="italic")
axr.set_yticks(range(len(rows)))
axr.set_yticklabels([r[0] for r in rows], fontsize=9)
axr.set_ylim(len(rows) - 0.3, -0.8)
axr.set_xlim(40, 100); axr.axvline(100, color="#333", ls=":", lw=1)
axr.set_xlabel("% of one core's peak")
axr.grid(axis="x", alpha=0.3)
axr.set_title("2. Same math, 50% to 89%: tile and who allocates registers", fontsize=10.5, loc="left")
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="o", ls="", color="#444", label="microkernel alone (bench_ukr, C = 64)"),
       Line2D([], [], marker="o", ls="", mfc="white", mec="#444", mew=1.6, label="all 32 YOLOv8n layers"),
       Line2D([], [], color=BLUE, lw=6, alpha=0.5, label="inline asm (registers fixed by hand)"),
       Line2D([], [], color=GREY, lw=6, alpha=0.5, label="intrinsics (GCC allocates)")]
axr.legend(handles=leg, fontsize=8, loc="upper left", frameon=True)
fig.suptitle("Stride-1 kernel design space on ece028: the hand-allocated 2×6 tile reaches 89% of peak alone",
             fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.005, "1 core, perf_event core cycles, GCC 8.5 -O3 -march=broadwell, FP32. Sources: results/s1_*_b1_t1.csv, s1_tile_sweep.csv, ukr_sweep*.csv; v2a alone = one ece028 run.",
         fontsize=7.5, color="#666")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(a.out, dpi=170)
print("wrote", a.out)
