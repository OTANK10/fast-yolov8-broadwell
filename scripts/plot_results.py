#!/usr/bin/env python3
"""Per-layer FLOPs/cycle: your kernel vs the two baselines, with the machine peak.

    python scripts/plot_results.py results/s1_v1_b1_t1.csv results/baseline_b1_t1.csv
    -> results/s1_v1_b1_t1.png   (one bar group per layer, dashed line = T-core peak)
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kern_csv, base_csv = sys.argv[1], sys.argv[2]
kern = list(csv.DictReader(open(kern_csv)))
base = {r["layer"]: r for r in csv.DictReader(open(base_csv))}
threads = int(kern[0]["threads"])
peak = 32.0 * threads

names = [r["layer"] for r in kern]
ours = [float(r["flops_per_cycle"]) for r in kern]
ocv = [float(base[n]["ocv_flops_per_cycle"]) if n in base else float("nan") for n in names]
ort = [float(base[n]["ort_flops_per_cycle"]) if n in base else float("nan") for n in names]

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order: ours, OpenCV, ORT

fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(names)), 4.6))
w = 0.27
xs = range(len(names))
for i, (vals, label) in enumerate([(ours, kern[0]["kernel"]), (ocv, "OpenCV DNN"), (ort, "ONNX Runtime")]):
    ax.bar([x + (i - 1) * w for x in xs], vals, w * 0.92, label=label, color=COLORS[i], zorder=2)
ax.axhline(peak, color=INK, ls="--", lw=1.2, zorder=3)
ax.text(len(names) - 0.5, peak, f"peak {peak:.0f} FLOPs/cycle ({threads} core{'s' if threads > 1 else ''})",
        ha="right", va="bottom", fontsize=8, color=INK)
ax.set_xticks(list(xs))
ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7, color=MUTED)
ax.set_ylabel("FLOPs / cycle", color=INK)
top = max([peak] + [v for v in ours + ocv + ort if v == v])
ax.set_ylim(0, top * 1.12)  # never clip: >peak means the cycle count is wrong (turbo / TSC)
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
ax.set_title(f"yolov8n per-layer performance, batch {kern[0]['N']}, {threads} thread(s)", fontsize=10, color=INK, loc="left", pad=22)
fig.tight_layout()
out = kern_csv.rsplit(".", 1)[0] + ".png"
fig.savefig(out, dpi=160)
print("wrote", out)
