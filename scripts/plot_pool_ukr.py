#!/usr/bin/env python3
"""Plot the SPPF pool microkernel sweep (bench_pool_ukr --sweep ... --csv).

    python scripts/plot_pool_ukr.py results/pool_ukr_sweep.csv [-o results/pool_ukr_sweep.png] [--ukrs v1,v2,v5]

Left: vmaxps per cycle vs plane size, against the port-1 peak of 1.0 (the pool's
"FLOPs/cycle"). Right: cycles per output vector (8 channels of one pixel), lower is
better, with each version's floor = its vmaxps per output at 1 per cycle. Best of the
runs per point. Dotted line: the plane size where the version's working set passes the
32 KB L1. The model.9 plane (20 x 20) is marked.
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
ap.add_argument("--ukrs", default=None, help="comma list, default all")
ap.add_argument("--cycles-only", action="store_true", help="only the cycles-per-output panel (for a slide)")
a = ap.parse_args()
out = a.out or a.csv.replace(".csv", ".png")

best = defaultdict(dict)   # ukr -> H -> row with the fewest cycles
for r in csv.DictReader(open(a.csv)):
    H = int(r["H"])
    if H not in best[r["ukr"]] or float(r["best_cycles_per_call"]) < float(best[r["ukr"]][H]["best_cycles_per_call"]):
        best[r["ukr"]][H] = r
ukrs = [u for u in sorted(best) if not a.ukrs or u in a.ukrs.split(",")]
COL = {"v1": "#6B7782", "v2": "#0C6A87", "v3": "#B7731A", "v4": "#B8C1C8", "v5": "#BB0027"}

if a.cycles_only:
    fig, ar = plt.subplots(figsize=(8, 5.2)); al = None
else:
    fig, (al, ar) = plt.subplots(1, 2, figsize=(13, 5.2))
for u in ukrs:
    Hs = sorted(best[u])
    vpc = [float(best[u][h]["vmaxps_per_cycle"]) for h in Hs]
    cpv = [float(best[u][h]["cycles_per_output_vec"]) for h in Hs]
    c = COL.get(u, None)
    if al: al.plot(Hs, vpc, "-o", ms=3, lw=1.6, color=c, label=f"{u}: {best[u][Hs[0]]['note']}")
    ar.plot(Hs, cpv, "-o", ms=3, lw=1.6, color=c, label=f"{u}: {best[u][Hs[0]]['note']}" if a.cycles_only else u)
    floor = [float(best[u][h]["vmaxps_per_call"]) / (h * h) for h in Hs]   # vmaxps per output at 1 per cycle
    ar.plot(Hs, floor, "--", color=c, lw=0.9, alpha=0.6)
    l1 = [h for h in Hs if float(best[u][h]["working_set_kb"]) > 32]
    if l1:
        for ax in (al, ar):
            if ax: ax.axvline(l1[0], color=c, ls=":", lw=1, alpha=0.7)
if al:
    al.axhline(1.0, color="#333", ls=":", lw=1.2)
    al.text(al.get_xlim()[0] + 0.5, 1.02, "peak: 1 vmaxps/cycle (port 1 only on Broadwell)", fontsize=8.5, color="#333")
for ax in [x for x in (al, ar) if x]:
    ax.axvline(20, color="#999", lw=6, alpha=0.15)
    ax.set_xlabel("plane H = W (one 8-channel block); model.9 is 20 × 20 (shaded)")
    ax.grid(alpha=0.3)
if al:
    al.set_ylabel("vmaxps per cycle")
    al.set_ylim(0, 1.15)
    al.set_title("Pool microkernel alone: vmaxps per cycle vs plane size", fontsize=10.5, loc="left")
    al.legend(fontsize=7.5, loc="lower right")
ar.set_ylabel("cycles per output vector (8 channels, lower is better)")
ar.set_ylim(0, None)
ar.set_title("Pool microkernel alone: cycles per output vector; dashed = vmaxps floor (1 per cycle)" if a.cycles_only
             else "Cycles per output vector; dashed = each version's vmaxps floor", fontsize=10.5, loc="left")
ar.legend(fontsize=7.5 if a.cycles_only else 8, loc="upper left" if a.cycles_only else "upper right")
fig.text(0.01, 0.005, "ece028, 1 core (taskset), cycles = TSC × 3.4/2.4 (Assignment 2), best of runs; dotted: working set passes the 32 KB L1. "
         "Source: " + a.csv, fontsize=7.5, color="#666")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, dpi=170)
print("wrote", out)
