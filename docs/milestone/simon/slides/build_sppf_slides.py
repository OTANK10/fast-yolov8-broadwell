#!/usr/bin/env python3
"""SPPF max-pool microkernel slides (Simon), CMU Digital Tartan template.

    python docs/milestone/simon/slides/build_sppf_slides.py   ->  sppf_slides_cmu.pptx

Slide 1 uses the template's "Title Slide" layout (red tartan); every other slide uses
"Title Only", so the template's title style, tartan band and wordmark come from its
layouts. Content stays above the band (y <= 4.6 in). Same content and numbers as
build_mid_deck.js; every % of peak assumes 3.4 GHz (Assignment 2 convention).
"""
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
AMBER_SOFT = RGBColor(0xF5, 0xE6, 0xCF)
TEMPLATE = os.path.join(HERE, "cmu_template.pptx")
OUT = os.path.join(HERE, "sppf_slides_cmu.pptx")
fig = lambda p: os.path.join(ROOT, p)

RED, INK, MUTED, TEAL, TEALSOFT = RGBColor(0xBB, 0x00, 0x27), RGBColor(0x1B, 0x26, 0x30), RGBColor(0x5A, 0x68, 0x75), RGBColor(0x0C, 0x6A, 0x87), RGBColor(0xD7, 0xEA, 0xF0)
SOFT, GREY, AMBER, WHITE, PINK = RGBColor(0xF1, 0xF3, 0xF5), RGBColor(0x6B, 0x77, 0x82), RGBColor(0xB7, 0x73, 0x1A), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF8, 0xE5, 0xE9)
AMBER_SOFT = RGBColor(0xF5, 0xE6, 0xCF)
FONT = "Open Sans"  # the template's body font

prs = Presentation(TEMPLATE)
# drop the template's two sample slides
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
L_TITLE, L_ONLY = prs.slide_layouts[0], prs.slide_layouts[2]


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT, name=None):
    """runs: a string, or a list of paragraphs, each a string or a list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
        if i: p.space_before = Pt(o.get("space", 6) if not isinstance(para, str) else 6)
    return tb


def card(slide, x, y, w, h, fill=SOFT, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if name: s.name = name
    return s


def content_slide(title, notes):
    s = prs.slides.add_slide(L_ONLY)
    s.shapes.title.text = title
    s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, big, small, color=TEAL):
    text(slide, x, y, w, 0.6, big, size=30, bold=True, color=color)
    text(slide, x, y + 0.62, w, 0.6, small, size=12, color=MUTED)


def style_chart(chart, fmt, color):
    chart.has_legend = False
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.font.size, dl.font.name, dl.font.color.rgb = Pt(11), FONT, INK
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.series[0].format.fill.solid(); plot.series[0].format.fill.fore_color.rgb = color
    for ax in (chart.category_axis, chart.value_axis):
        ax.tick_labels.font.size, ax.tick_labels.font.name, ax.tick_labels.font.color.rgb = Pt(10), FONT, MUTED
        ax.format.line.fill.background()
    chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE3, 0xE7)
    chart.has_title = True
    chart.chart_title.text_frame.paragraphs[0].runs or chart.chart_title.text_frame.paragraphs[0].add_run()


def chart_title(chart, t):
    tf = chart.chart_title.text_frame
    tf.text = t
    f = tf.paragraphs[0].runs[0].font
    f.size, f.name, f.bold, f.color.rgb = Pt(12), FONT, False, INK



def cell(slide, x, y, w, h, lab, fc, ec, size=8, bold=False):
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fc; b.line.color.rgb = ec; b.line.width = Pt(0.75); b.shadow.inherit = False
    tf = b.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = lab; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r.font.size, r.font.name, r.font.color.rgb, r.font.bold = Pt(size), FONT, ec, bold
    return b


def table(slide, x, y, w, h, rows, widths, size=10, hl=()):
    tb = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for c, wd in enumerate(widths): tb.columns[c].width = Inches(wd)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cl = tb.cell(r, c); cl.text = v
            cl.fill.solid(); cl.fill.fore_color.rgb = SOFT if r == 0 else (PINK if r in hl else WHITE)
            cl.margin_left = cl.margin_right = Inches(0.06); cl.margin_top = cl.margin_bottom = Inches(0.03)
            f = cl.text_frame.paragraphs[0].runs[0].font
            f.size, f.name, f.color.rgb, f.bold = Pt(size), FONT, INK, r == 0 or c == 0
    return tb



import csv as _csv
from collections import defaultdict
best = defaultdict(dict)
for r in _csv.DictReader(open(fig("results/pool_ukr_sweep.csv"))):
    H = int(r["H"])
    if H not in best[r["ukr"]] or float(r["best_cycles_per_call"]) < float(best[r["ukr"]][H]["best_cycles_per_call"]):
        best[r["ukr"]][H] = r
at20 = {u: (float(best[u][20]["cycles_per_output_vec"]), float(best[u][20]["vmaxps_per_cycle"]), float(best[u][20]["vmaxps_per_call"]) / 400) for u in best}
mean = lambda u, hs: sum(float(best[u][h]["cycles_per_output_vec"]) for h in hs) / len(hs)
fit_v2, beyond_v2_even, beyond_v2_odd = mean("v2", range(5, 18)), mean("v2", range(22, 41, 2)), mean("v2", range(21, 41, 2))
beyond_v5 = mean("v5", range(21, 41))

# ------------------------------------------------------------------ 1. design
s = content_slide("SPPF max-pool microkernel: 6 vmaxps per output",
    "SPPF in model.9 runs three chained 5 by 5, stride 1 max-pools on a 128-channel 20 by 20 tensor and concatenates them. "
    "The microkernel is one 5 by 5 pool of one 8-channel plane. There are no FLOPs, so the unit of work is vmaxps: one instruction takes the max of 8 channels. "
    "On Broadwell vmaxps issues only on port 1, so the peak is 1 per cycle, our equivalent of 32 FLOPs per cycle. "
    "A direct 5 by 5 max is 24 vmaxps per output. Splitting it into a max over 5 columns, then over 5 rows, makes it 8: that is v1. "
    "v2 shares pair maxes: P of j is the max of s j and s j plus 1, and each output is max of P i minus 2, P i, s i plus 2, so 3 per pass and 6 per output. "
    "Padding has to act like minus infinity because the input is after SiLU and can be negative; we clamp the window at the 2 edge pixels instead of storing a minus-infinity border. "
    "Every version matches a scalar 5 by 5 max exactly and matches OpenCV's real model.9 outputs with zero error.")
steps = [("Unit of work", "vmaxps = max of 8 channels (one YMM register). Port 1 only on Broadwell: peak 1 per cycle."),
         ("Separable (v1)", "5×5 = max over 5 columns, then 5 rows: 24 → 8 vmaxps per output."),
         ("Shared pairs (v2)", "P[j] = max(s[j], s[j+1]); out[i] = max(P[i−2], P[i], s[i+2]): 3 per pass, 6 per output."),
         ("Edges = −∞", "SiLU output can be negative, so a zero border is wrong: clamp the window at the 2 edge pixels.")]
for i, (h, b) in enumerate(steps):
    y = 1.08 + i * 0.86
    card(s, 0.5, y, 4.6, 0.78, PINK if i == 2 else SOFT, f"step-{i}")
    text(s, 0.62, y + 0.06, 4.4, 0.26, h, size=11.5, bold=True, color=RED if i == 2 else INK)
    text(s, 0.62, y + 0.33, 4.4, 0.45, b, size=9.5, color=MUTED)
rows = [("Version", "vmaxps / output", "Cycles / output", "vmaxps / cycle")] + [
        (lab, f"{at20[u][2]:.1f}", f"{at20[u][0]:.2f}", f"{at20[u][1]:.2f}")
        for u, lab in (("v1", "v1 separable"), ("v2", "v2 shared pairs"), ("v5", "v5 5-row ring"))]
text(s, 5.35, 1.08, 4.2, 0.3, "Alone, one 20×20 plane (model.9), ece028", size=11.5, bold=True)
table(s, 5.35, 1.4, 4.15, 1.4, rows, (1.45, 0.9, 0.9, 0.9), size=9.5, hl=(2,))
text(s, 5.35, 2.85, 4.15, 0.3, "vmaxps / output includes the cheaper edge pixels; at 1 vmaxps per cycle it is the floor in cycles.", size=8.5, color=MUTED)
stat(s, 5.35, 3.25, 2.0, f"{at20['v2'][1] * 100:.0f}%", "of the port-1 peak (v2: 0.86 vmaxps/cycle)", RED)
stat(s, 7.45, 3.25, 2.05, "24 → 6", "vmaxps per output: direct → separable + shared pairs", TEAL)

# ------------------------------------------------------------------ 2. sweep
s = content_slide("Alone: v2 wins in L1, v5 removes the L1 cliff",
    "Same benchmark as the conv microkernel: one plane, pooled over and over, warm-up, TSC times 3.4 over 2.4, swept over the plane size. "
    f"While the plane fits in the 32 KB L1, up to 17 by 17, v2 is fastest at {fit_v2:.1f} cycles per output vector against its floor of 6. "
    "v2 keeps a whole temporary plane between the rows pass and the columns pass, so source, temporary and destination pass 32 KB at 18 by 18. "
    f"Beyond that it falls to {beyond_v2_even:.0f} cycles at even sizes and {beyond_v2_odd:.0f} at odd sizes, likely because the planes then alias in the cache. "
    f"v5 keeps only the 5 rows the columns pass needs, a 3.2 KB ring, so it stays at about {beyond_v5:.1f} cycles at every size. "
    "At model.9's 20 by 20, v2 is still just ahead. Like Prof. Low said about tiling, v5 does not raise the peak, it flattens the drop-off.")
s.shapes.add_picture(fig("results/pool_ukr_sweep_cycles.png"), Inches(0.5), Inches(1.0), height=Inches(3.55))
stat(s, 6.95, 1.0, 2.6, f"{fit_v2:.1f}", "cycles/output, v2 while in L1 (floor 6.0)", RED)
stat(s, 6.95, 2.15, 2.6, f"{beyond_v2_even:.0f}–{beyond_v2_odd:.0f}", "cycles/output, v2 past L1 (even–odd sizes)", GREY)
stat(s, 6.95, 3.3, 2.6, f"{beyond_v5:.1f}", "cycles/output, v5 at every size past L1", TEAL)

prs.save(OUT)
print("wrote", OUT)
