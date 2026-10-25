#!/usr/bin/env python3
"""The 2x5 output tile drawn as tensors (input, filters, output), not registers.

    python docs/milestone/om/make_tile_fig.py   ->  docs/milestone/om/figs/tile_tensors.png

Isometric boxes: width W to the right, height H downward, channels as depth.
Left: input map, with the 3 x 7 x C window one tile reads. Middle: the 16 filters
(2 blocks of 8), each 3 x 3 x C. Right: output map, with the tile: 1 row x 5 pixels
x 16 channels = 10 vectors of 8 = 10 accumulator registers, SiLU, one store.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

TEAL, TEAL_SOFT, AMBER, AMBER_SOFT, GREY, INK, LINE = "#0C6A87", "#D7EAF0", "#B7731A", "#F5E6CF", "#E6EBEE", "#1B2630", "#8A97A2"
DX, DY = 0.55, 0.32  # projection of one unit of depth


def P(x, y, z):
    return (x + z * DX, -y + z * DY)


def box(ax, x, y, z, w, h, d, fc, ec=LINE, lw=1.0, alpha=1.0, z_order=1):
    """Front face, top face, right face of an axis-aligned box (y grows downward)."""
    front = [P(x, y, z), P(x + w, y, z), P(x + w, y + h, z), P(x, y + h, z)]
    top = [P(x, y, z), P(x + w, y, z), P(x + w, y, z + d), P(x, y, z + d)]
    right = [P(x + w, y, z), P(x + w, y, z + d), P(x + w, y + h, z + d), P(x + w, y + h, z)]
    for face, shade in ((top, 0.92), (right, 0.82), (front, 1.0)):
        ax.add_patch(Polygon(face, closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z_order))


fig, ax = plt.subplots(figsize=(15, 5.2))
ax.set_aspect("equal")
ax.axis("off")

# ---- input map: W 9, H 6, C depth 4 (schematic)
ix, iy = 0.0, 0.0
box(ax, ix, iy, 0, 9, 6, 4, GREY, z_order=1)
# window read by one tile: rows 1..3, cols 1..7, all C
box(ax, ix + 1, iy + 1, 0, 7, 3, 4, AMBER_SOFT, ec=AMBER, lw=1.6, z_order=2)
for c in range(1, 7):  # pixel grid on the window's front face
    ax.plot(*zip(P(ix + 1 + c, iy + 1, 0), P(ix + 1 + c, iy + 4, 0)), color=AMBER, lw=0.6, zorder=3)
for r in range(1, 3):
    ax.plot(*zip(P(ix + 1, iy + 1 + r, 0), P(ix + 8, iy + 1 + r, 0)), color=AMBER, lw=0.6, zorder=3)
ax.text(*P(ix + 4.5, iy + 6.7, 0), "input map  H × W × C", ha="center", va="top", fontsize=17, color=INK)
ax.text(*P(ix + 4.5, iy + 7.5, 0), "reads a 3 × 7 window, all C", ha="center", va="top", fontsize=14.5, color=AMBER)
ax.text(*P(ix + 9.2, iy + 6, 2), "C", fontsize=17, color=INK)

# ---- filters: 16 of 3 x 3 x C, shown as 2 stacks (block kb, kb+1)
fx = 15.0
for b in range(2):
    for i in range(3):
        box(ax, fx + b * 3.3 + i * 0.25, 1.0 + i * 0.25, 0, 1.5, 1.5, 4, AMBER_SOFT if b == 0 else "#FBF1E2", ec=AMBER, lw=1.0, z_order=3 + i)
    ax.text(*P(fx + b * 3.3 + 1.0, 3.6, 0), f"8 filters\nblock {'kb' if b == 0 else 'kb+1'}", ha="center", va="top", fontsize=14, color=INK)
ax.text(*P(fx + 2.6, 6.7, 0), "16 filters, 3 × 3 × C", ha="center", va="top", fontsize=17, color=INK)
ax.text(*P(fx + 2.6, 7.5, 0), "one output: k = C × 9 MACs", ha="center", va="top", fontsize=14.5, color=AMBER)

# ---- output map with the tile
ox = 26.0
box(ax, ox, 0, 0, 7, 6, 4, GREY, z_order=1)
# tile: one row (row 2), 5 pixels, 16 channels = 2 blocks of 8 (depth 2 + 2)
for blk in range(2):
    for j in range(5):
        box(ax, ox + 1 + j, 2, blk * 2.0, 1, 1, 2.0, TEAL_SOFT if blk == 0 else "#C3DFE8", ec=TEAL, lw=1.4, z_order=5 - blk)
ax.text(*P(ox + 0.4, 4.4, 0), "front row: block kb (8 ch)\nback row: block kb+1 (8 ch)", fontsize=13.5, color=TEAL, va="top")
ax.text(*P(ox + 3.5, 6.7, 0), "output map  H × W × K", ha="center", va="top", fontsize=17, color=INK)
ax.text(*P(ox + 3.5, 7.5, 0), "tile: 5 pixels × 16 ch = 80", ha="center", va="top", fontsize=14.5, color=TEAL)
lab = P(ox + 6.5, 1.4, 4.6)
ax.annotate("each cell = 8 channels of one pixel\n= one YMM register (accumulator)\n2 blocks × 5 pixels = 10 accumulators",
            xy=P(ox + 5.9, 2.2, 3.0), xytext=(lab[0] + 1.2, lab[1] + 0.6), fontsize=14, color=TEAL,
            arrowprops=dict(arrowstyle="-", color=TEAL, lw=1))
ax.text(lab[0] + 1.2, lab[1] - 3.2, "after the C × 9 steps:\n+ bias, SiLU in registers,\nthen one store", fontsize=14, color=INK)

for a, b in ((P(10.6, 3, 2), P(fx - 0.6, 3, 2)), (P(fx + 6.0, 3, 2), P(ox - 0.4, 3, 2))):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.4))

ax.set_xlim(-0.5, 44)
ax.set_ylim(-10, 3.6)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "tile_tensors.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print("wrote", out)
