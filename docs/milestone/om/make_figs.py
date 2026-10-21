#!/usr/bin/env python3
"""Figures for Om's milestone report and slides, from the ece028 CSVs in results/.
   python docs/milestone/om/make_figs.py      -> docs/milestone/om/figs/*.png
Baseline FLOPs/cycle are converted with the measured single-core clock
(our core cycles / wall time = 3.39 GHz on ece028), so both sides use real cycles."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; OUT = os.path.join(os.path.dirname(__file__), "figs"); os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#1f2329", "#6b7280", "#e5e7eb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9aa1ab"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
PEAK = 32.0

def load(n): return {r["layer"]: r for r in csv.DictReader(open(f"{R}/{n}.csv"))}
def overall(d):
    fl = sum(float(r["flops"]) for r in d.values()); cy = sum(float(r["best_cycles"]) for r in d.values())
    return fl / cy
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, zorder=0)

v = {k: load(f"s1_{k}_b1_t1") for k in ["v0", "v1", "v2", "v4", "v4b", "v5"]}
v["ref"] = load("ref_b1_t1")
v2 = v["v2"]
ghz = sum(float(r["best_cycles"]) / float(r["time_us"]) / 1e3 for r in v2.values()) / len(v2)
base = load("baseline_b1_t1")

# 1. staircase ------------------------------------------------------------
names = ["ref", "v0", "v1", "v2", "v4", "v5"]
labels = ["ref\ntextbook", "v0\nscalar,\nblocked", "v1\nSIMD,\n1 acc", "v2\n2x5 tile,\n10 acc", "v4\n2x6 tile,\n12 acc", "v5\nv4 +\npacking"]
vals = [overall(v[n]) for n in names]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cols = [GRAY, GRAY, GRAY, BLUE, GRAY, GRAY]
ax.bar(range(len(vals)), vals, 0.62, color=cols, zorder=2)
for i, x in enumerate(vals):
    ax.text(i, x + 0.6, f"{x:.2f}\n({100 * x / PEAK:.0f}%)" if x > 1 else f"{x:.2f}\n(1%)", ha="center", va="bottom", fontsize=8, color=INK)
ax.axhline(PEAK, color=INK, ls="--", lw=1); ax.text(5.4, PEAK + 0.4, "peak 32", ha="right", fontsize=8, color=INK)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("FLOPs / cycle (1 core)"); ax.set_ylim(0, 40); style(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/staircase.png", dpi=220); plt.close(fig)

# 2. accumulator sweep ------------------------------------------------------
sw = list(csv.DictReader(open(f"{R}/s1_tile_sweep.csv")))
tiles = list(dict.fromkeys(r["tile"] for r in sw))
pts = []
for t in tiles:
    rr = [r for r in sw if r["tile"] == t]
    pts.append((int(rr[0]["accumulators"]), sum(float(r["flops_per_cycle"]) for r in rr) / len(rr), t))
fig, ax = plt.subplots(figsize=(5.6, 3.1))
xs = list(range(0, 13)); ax.plot(xs, [min(3.2 * x, PEAK) for x in xs], color=MUTED, ls="--", lw=1.2, zorder=2,
                                  label="model: min(acc x 32/10, 32)")
for a, f, t in pts:
    c = BLUE if t == "t2x5" else (ORANGE if t == "t1x10" else INK)
    ax.scatter([a], [f], s=36, color=c, zorder=3, edgecolor="white", linewidth=1.5)
    dx, ha = (0.25, "left")
    if t in ("t2x1", "t2x2"): dx, ha = (-0.25, "right")
    dy = 0
    if t == "t1x10": dx, ha, dy = (0.0, "center", -2.2)
    if t == "t2x2": dy = -0.8
    if t == "t1x4": dy = 0.8
    ax.text(a + dx, f + dy, t.replace("t", "") + (" (load-bound)" if t == "t1x10" else ""), fontsize=8, va="center", ha=ha, color=c)
sl = [r["layer"] for r in sw if r["tile"] == "t2x5"]
f12 = sum(float(v["v4"][L]["flops_per_cycle"]) for L in sl) / len(sl)
ax.scatter([12], [f12], s=36, color=INK, zorder=3, edgecolor="white", linewidth=1.5)
ax.text(12, f12 - 2.2, "2x6 (v4)", fontsize=8, ha="right", color=INK)
print("2x6 on sweep layers", round(f12, 1))
ax.set_xlabel("independent accumulators (KB x OW)"); ax.set_ylabel("FLOPs / cycle (1 core)")
ax.set_xlim(0, 12); ax.set_ylim(0, 36); style(ax); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/accumulators.png", dpi=220); plt.close(fig)

# 3. per layer shape vs baselines ------------------------------------------------
def bfpc(k, col): return float(base[k]["flops"]) / (float(base[k][col]) * 1e-6 * ghz * 1e9)
ocv = [bfpc(k, "ocv_us") for k in v2]; ort = [bfpc(k, "ort_us") for k in v2]; ours = [float(v2[k]["flops_per_cycle"]) for k in v2]
groups = {}
for k in v2:
    key = (int(v2[k]["C"]), int(v2[k]["H"]))
    g = groups.setdefault(key, {"fl": 0.0, "ours": 0.0, "ocv": 0.0, "ort": 0.0, "n": 0})
    fl = float(v2[k]["flops"]); g["fl"] += fl; g["n"] += 1
    g["ours"] += fl / float(v2[k]["flops_per_cycle"])          # cycles
    g["ocv"] += float(base[k]["ocv_us"]) * 1e-6 * ghz * 1e9
    g["ort"] += float(base[k]["ort_us"]) * 1e-6 * ghz * 1e9
keys = sorted(groups, key=lambda c: (c[0], -c[1]))
fig, ax = plt.subplots(figsize=(7.0, 3.0)); w = 0.27; x = range(len(keys))
for off, col, lab, cname in [(-w, ORANGE, "OpenCV DNN", "ocv"), (0, BLUE, "ours (s1_v2)", "ours"), (w, AQUA, "ONNX Runtime", "ort")]:
    ax.bar([i + off for i in x], [groups[c]["fl"] / groups[c][cname] for c in keys], w * 0.92, color=col, label=lab, zorder=2)
ax.axhline(PEAK, color=INK, ls="--", lw=1); ax.text(len(keys) - 0.5, PEAK + 0.5, "peak 32", ha="right", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([f"C={c}\n{h}x{h}\n({groups[(c, h)]['n']})" for c, h in keys], fontsize=7)
ax.set_ylabel("FLOPs / cycle (1 core)"); ax.set_ylim(0, 37); style(ax)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.tight_layout(); fig.savefig(f"{OUT}/layers_vs_baselines.png", dpi=220); plt.close(fig)

# summary numbers for the text
tot = lambda col: sum(float(base[k][col]) for k in v2)
fl = sum(float(v2[k]["flops"]) for k in v2)
print(f"clock {ghz:.2f} GHz")
for n in names: print(n, f"{overall(v[n]):.2f}", f"{100 * overall(v[n]) / PEAK:.1f}%")
print("ocv overall fpc", f"{fl / (tot('ocv_us') * 1e-6 * ghz * 1e9):.2f}", "ort", f"{fl / (tot('ort_us') * 1e-6 * ghz * 1e9):.2f}")
print("ours total us", sum(float(v2[k]['time_us']) for k in v2), "ocv", tot("ocv_us"), "ort", tot("ort_us"))
print("ocv range", f"{min(ocv):.1f}-{max(ocv):.1f}", "ort range", f"{min(ort):.1f}-{max(ort):.1f}", "ours", f"{min(ours):.1f}-{max(ours):.1f}")
for a, f, t in pts: print(t, a, f"{f:.1f}", f"{100*f/min(3.2*a,32):.0f}% of model")
