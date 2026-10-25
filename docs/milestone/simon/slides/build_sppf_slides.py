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
best = {}
for r in _csv.DictReader(open(fig("results/pool_ukr_sweep.csv"))):
    if r["ukr"] != "v2":
        continue
    H = int(r["H"])
    if H not in best or float(r["best_cycles_per_call"]) < float(best[H]["best_cycles_per_call"]):
        best[H] = r
c20, v20 = float(best[20]["cycles_per_output_vec"]), float(best[20]["vmaxps_per_cycle"])
cyc20 = float(best[20]["best_cycles_per_call"])
fit = sum(float(best[h]["cycles_per_output_vec"]) for h in range(5, 18)) / 13
VIOLET = RGBColor(0x5B, 0x4B, 0x8A)

# ------------------------------------------------------------------ 1. what and the peak
s = content_slide("SPPF max-pool microkernel: the work and its peak",
    "SPPF in model.9 runs three chained 5 by 5, stride 1, pad 2 max-pools on a 128-channel 20 by 20 tensor and concatenates input and the three results. "
    "Our microkernel is one 5 by 5 pool of one 8-channel plane: 20 by 20 pixels, 8 channels each. "
    "There are no FLOPs, so the unit of work is vmaxps, one instruction that takes the max of 8 channels. "
    "On Broadwell vmaxps has a latency of 3 cycles and issues only on port 1, one per cycle, so the peak is 1 vmaxps per cycle. "
    "A direct 5 by 5 window is 24 maxes per output. We split it: a max over 5 columns, then over 5 rows, 4 plus 4. "
    "Then each row pass shares pair maxes, so it needs 3 per output instead of 4: 6 vmaxps per output, so at best 6 cycles per output vector. "
    "Unlike a convolution there is no running sum, so outputs do not depend on each other and we do not need many accumulators to hide latency.")
cards = [("Task (model.9)", "3 chained 5×5 max-pools, stride 1, pad 2, on 128 ch × 20 × 20; concat X, P1, P2, P3 → 512 ch."),
         ("Microkernel", "one 5×5 pool of one 8-channel plane (20 × 20 pixels × 8 channels)."),
         ("Unit of work", "vmaxps: max of 8 channels in one YMM register. No FLOPs."),
         ("Peak", "vmaxps: latency 3, port 1 only → 1 vmaxps per cycle.")]
for i, (h, b) in enumerate(cards):
    y = 1.08 + i * 0.86
    card(s, 0.5, y, 4.4, 0.78, PINK if i == 3 else SOFT, f"c-{i}")
    text(s, 0.62, y + 0.06, 4.2, 0.26, h, size=11.5, bold=True, color=RED if i == 3 else INK)
    text(s, 0.62, y + 0.33, 4.2, 0.45, b, size=9.5, color=MUTED)
text(s, 5.2, 1.08, 4.3, 0.3, "vmaxps per output vector", size=12, bold=True)
rows = [("Step", "vmaxps / output"), ("Direct 5 × 5 window", "24"), ("Separable: 5 columns, then 5 rows", "4 + 4 = 8"),
        ("Shared pair maxes in the row pass", "3 + 3 = 6")]
table(s, 5.2, 1.4, 4.3, 1.4, rows, (3.0, 1.3), size=10.5, hl=(3,))
stat(s, 5.2, 3.05, 2.1, "6", "cycles per output vector at best (6 vmaxps at 1 per cycle)", RED)
stat(s, 7.4, 3.05, 2.1, "0", "running sums: outputs are independent, no accumulator chain", TEAL)

# ------------------------------------------------------------------ 2. (a)-(c)
s = content_slide("(a)–(c): size, registers, allocation",
    "(a) One call computes one plane: 20 by 20 pixels times 8 channels, 3200 outputs, 400 output vectors. Each step of the inner loop finishes one output vector, 8 outputs. "
    "(b) The inner loop uses 4 of the 16 YMM registers. It needs so few because nothing accumulates: each output is done after 3 vmaxps, "
    "and consecutive outputs are independent, so out-of-order execution overlaps them and hides the 3-cycle latency. The limit is port 1, not registers. "
    "(c) Inputs: s of i is loaded into a register; s of i plus 1 and s of i plus 2 are read straight from memory as the vmaxps operand. "
    "Temporaries: the two previous pair maxes and the new one. Output: the result register, stored right away. "
    "These counts are from GCC's generated code for the loop; we write intrinsics and GCC assigns the physical registers.")
regs = [("x = s[i]", RGBColor(0xE4, 0xE0, 0xF0), VIOLET), ("p2 = P[i]", AMBER_SOFT, AMBER), ("p1 = P[i−1]", AMBER_SOFT, AMBER),
        ("p0 = P[i−2] → out", TEALSOFT, TEAL)] + [("free", WHITE, GREY)] * 12
text(s, 0.5, 1.05, 4.7, 0.3, "16 YMM registers in the inner loop (8 floats each)", size=11, bold=True)
for i, (lab, fc, ec) in enumerate(regs):
    col, row = i % 4, i // 4
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5 + col * 1.15), Inches(1.4 + row * 0.5), Inches(1.08), Inches(0.42))
    b.fill.solid(); b.fill.fore_color.rgb = fc; b.line.color.rgb = ec; b.line.width = Pt(1.25); b.shadow.inherit = False
    if lab == "free": b.line.dash_style = 4
    tf = b.text_frame; tf.margin_left = tf.margin_right = 0
    r = tf.paragraphs[0].add_run(); r.text = lab; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r.font.size, r.font.name, r.font.color.rgb, r.font.bold = Pt(9), "Courier New", ec, lab != "free"
text(s, 0.5, 3.45, 4.6, 1.1, [[("input ", {"bold": True, "color": VIOLET}), ("s[i]; s[i+1], s[i+2] come from memory as vmaxps operands", {})],
                            [("temporaries ", {"bold": True, "color": AMBER}), ("P[j] = max(s[j], s[j+1]): the new one and the last two", {})],
                            [("output ", {"bold": True, "color": TEAL}), ("max(P[i−2], P[i], s[i+2]), stored at once", {})]], size=10, color=MUTED)
qa = [("(a) Size", "One call: 20 × 20 pixels × 8 channels = 3200 outputs (400 vectors). One inner step: 1 vector = 8 outputs."),
      ("(b) Registers", "Need 4 of the 16 YMM registers. No accumulation, so latency is hidden by overlapping independent outputs; port 1 is the limit."),
      ("(c) Allocation", "Input: 1 register + 2 memory operands. Temporaries: 3 pair maxes. Output: 1 register, stored immediately. Pointers and counters in integer registers.")]
for i, (h, b) in enumerate(qa):
    y = 1.05 + i * 1.2
    card(s, 5.25, y, 4.25, 1.1, PINK if i == 1 else SOFT, f"qa-{i}")
    text(s, 5.38, y + 0.08, 4.0, 0.28, h, size=12.5, bold=True, color=RED if i == 1 else INK)
    text(s, 5.38, y + 0.38, 4.0, 0.7, b, size=10, color=MUTED)

# ------------------------------------------------------------------ 3. (d)-(e)
s = content_slide("(d)–(e): register reuse and the code",
    "(d) Reuse. Each pair max P of j is computed once and used by two outputs, j and j plus 2, so it stays in a register for two steps: "
    "the three pair registers rotate by one each step. Every vmaxps works on 8 channels at once, so one instruction serves 8 outputs. "
    "The rows-pass result is written to a temporary plane, and each value there is read by 5 outputs of the columns pass. "
    "(e) The code. The plane is in our NCHW8c layout, 8 channels of a pixel in 32 contiguous bytes. "
    "Rows pass: for each row, the first two and last two outputs use the clamped window, which is the same as minus-infinity padding; "
    "the interior runs the 3-vmaxps step. Columns pass: the same step down each column of the temporary plane, into the output plane.")
reuse = [("Pair max P[j]", "computed once, used by outputs j and j+2: kept in registers, rotated each step"),
         ("8 channels", "one vmaxps = 8 maxes: one instruction serves 8 outputs"),
         ("Rows-pass plane", "each value read by 5 outputs of the columns pass"),
         ("Per output", "3 vmaxps + 3 loads + 1 store per pass; port 1 is the limit")]
text(s, 0.5, 1.05, 3.6, 0.3, "(d) Reuse", size=13, bold=True, color=RED)
for i, (h, b) in enumerate(reuse):
    y = 1.4 + i * 0.8
    card(s, 0.5, y, 3.6, 0.72, SOFT, f"reuse-{i}")
    text(s, 0.62, y + 0.06, 3.4, 0.25, h, size=11.5, bold=True)
    text(s, 0.62, y + 0.31, 3.4, 0.42, b, size=9.5, color=MUTED)
text(s, 4.35, 1.05, 5.1, 0.3, "(e) Implementation, AVX2 intrinsics", size=13, bold=True, color=RED)
code = [
    "// one pass: d[i] = max(s[i-2..i+2]) along a line of n vectors",
    "d[0] = max(s0,s1,s2);   d[1] = max(s0..s3);     // clamped edges",
    "p0 = _mm256_max_ps(s[0], s[1]);                 // P[0]",
    "p1 = _mm256_max_ps(s[1], s[2]);                 // P[1]",
    "for (i = 2; i < n-2; i++) {",
    "  x  = _mm256_load_ps(s + 8*i);",
    "  p2 = _mm256_max_ps(x, s[i+1]);                // P[i]",
    "  out = _mm256_max_ps(_mm256_max_ps(p0, p2), s[i+2]);",
    "  _mm256_store_ps(d + 8*i, out);",
    "  p0 = p1;  p1 = p2;                            // rotate",
    "}",
    "d[n-2], d[n-1]: clamped edges;  plane: rows (in→tmp), columns (tmp→out)",
]
card(s, 4.35, 1.4, 5.15, 2.75, RGBColor(0xF7, 0xF8, 0xFA), "code")
text(s, 4.45, 1.46, 5.0, 2.65, code, size=7.5, font="Courier New", color=INK)
text(s, 4.35, 4.22, 5.15, 0.4, "Layout: NCHW8c, 8 channels of a pixel = 32 contiguous bytes = one aligned load. "
     "Padding acts as −∞ (input is after SiLU and can be negative).", size=9, color=MUTED)

# ------------------------------------------------------------------ 4. measured alone
s = content_slide(f"Alone on ece028: {v20 * 100:.0f}% of the port-1 peak",
    "Benchmarked like Assignment 2: one plane pooled over and over so it stays in cache, a 2-second warm-up, one pinned core, cycles from the TSC times 3.4 over 2.4. "
    f"At model.9's 20 by 20 plane: {cyc20:.0f} cycles per call, {c20:.2f} cycles per output vector against the floor of 6, so {v20:.2f} vmaxps per cycle, {v20 * 100:.0f} percent of port 1's peak. "
    f"Swept over the plane size, it holds about {fit:.1f} cycles per output while the source, temporary and output planes fit in the 32 KB L1, up to 17 by 17. "
    "Past that the temporary plane pushes the working set out of L1 and the time per output rises. Keeping only the 5 rows the columns pass needs is the next step. "
    "Every result matches a scalar 5 by 5 max exactly, and the full pool matches OpenCV's real model.9 outputs with zero error.")
s.shapes.add_picture(fig("results/pool_ukr_sweep_v2_cycles.png"), Inches(0.5), Inches(1.0), height=Inches(3.55))
stat(s, 6.85, 1.0, 2.7, f"{v20 * 100:.0f}%", f"of the port-1 peak at 20×20 ({v20:.2f} vmaxps/cycle)", RED)
stat(s, 6.85, 2.15, 2.7, f"{c20:.2f}", "cycles per output vector (floor 6.0)", TEAL)
stat(s, 6.85, 3.3, 2.7, "0 error", "vs scalar 5×5 max and OpenCV's model.9 outputs", GREY)

prs.save(OUT)
print("wrote", OUT)
