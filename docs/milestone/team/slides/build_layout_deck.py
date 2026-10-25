#!/usr/bin/env python3
"""Team slides on the shared memory layout (all kernels), CMU Digital Tartan template.

    python docs/milestone/team/slides/build_layout_deck.py   ->  layout_slides_cmu.pptx

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
OUT = os.path.join(HERE, "layout_slides_cmu.pptx")
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


# ------------------------------------------------------------------ 1. what the layout is
s = content_slide("One layout for every kernel: NCHW8c + halo",
    "All three of our kernels, stride 1, stride 2 and 1 by 1, read and write the same layout, defined once in include/common.h. "
    "ONNX stores NCHW: all of channel 0, then all of channel 1, so the 8 channels of one pixel are H times W floats apart and filling one register takes 8 loads. "
    "We store NCHW8c: channels in blocks of 8, and the 8 channels of a pixel are 32 contiguous bytes, one aligned load into one YMM register. "
    "Around every plane we store a 1-pixel border of zeros, the convolution's padding written out, so a 3 by 3 window at the edge just reads zeros. "
    "Weights are packed once offline, with BatchNorm already folded in, so the 8 output-channel weights for one input channel and tap are one aligned load. "
    "Because every layer uses the same layout, layers chain with no conversion; we convert only at the network input and output, outside the timing.")
CH = [RGBColor(0xD7, 0xEA, 0xF0), RGBColor(0xF5, 0xE6, 0xCF), RGBColor(0xE4, 0xE0, 0xF0), RGBColor(0xDF, 0xEE, 0xDC)]
CE = [TEAL, AMBER, RGBColor(0x5B, 0x4B, 0x8A), RGBColor(0x3D, 0x7A, 0x3A)]
cw = 0.33
text(s, 0.5, 1.05, 5.6, 0.3, [[("ONNX NCHW: ", {"bold": True, "color": INK}), ("channel planes one after another", {})]], size=11.5, color=MUTED)
for i in range(16):
    c, px = i // 4, i % 4
    cell(s, 0.5 + i * cw, 1.4, cw - 0.02, 0.36, f"c{c}p{px}", CH[c] if px == 0 else WHITE, CE[c], size=7, bold=px == 0)
text(s, 0.5, 1.8, 5.6, 0.3, "pixel p0's channels are H×W floats apart: 8 loads to fill one register", size=10, color=RED)
text(s, 0.5, 2.2, 5.6, 0.3, [[("NCHW8c: ", {"bold": True, "color": INK}), ("[N][C/8][H+2][W+2][8], 8 channels of a pixel together", {})]], size=11.5, color=MUTED)
for i in range(16):
    px, c = i // 8, i % 8
    cell(s, 0.5 + i * cw, 2.55, cw - 0.02, 0.36, f"p{px}c{c}", CH[c % 4] if px == 0 else WHITE, CE[c % 4], size=7, bold=px == 0)
br = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.48), Inches(2.52), Inches(8 * cw), Inches(0.42))
br.fill.background(); br.line.color.rgb = RED; br.line.width = Pt(2); br.shadow.inherit = False
text(s, 0.5, 2.95, 5.6, 0.3, "8 contiguous floats = 32 bytes = one _mm256_load_ps = one YMM register", size=10, color=RED)
card(s, 0.5, 3.4, 5.3, 1.15, SOFT, "w-card")
text(s, 0.62, 3.47, 5.1, 0.28, "Weights, packed once offline: [K/8][C/8][R][S][8 in][8 out]", size=11.5, bold=True)
text(s, 0.62, 3.77, 5.1, 0.75, "Same format for 3×3 and 1×1 (R = S = 1). The 8 output-channel weights of one input channel and tap "
     "are one aligned load, read front to back. BatchNorm is folded into weights and bias.", size=9.5, color=MUTED)
gx, gy, g = 6.25, 1.4, 0.36
text(s, 6.15, 1.05, 3.4, 0.3, [[("Zero halo: ", {"bold": True, "color": INK}), ("padding stored", {})]], size=11.5, color=MUTED)
for r in range(6):
    for c in range(8):
        border = r in (0, 5) or c in (0, 7)
        win = r <= 2 and c <= 2
        b = cell(s, gx + c * g, gy + r * g, g - 0.03, g - 0.03, "0" if border else "",
                 RGBColor(0xE6, 0xEB, 0xEE) if border else TEALSOFT, AMBER if win else (GREY if border else TEAL))
        if win: b.line.width = Pt(2)
text(s, 6.15, 3.65, 3.4, 0.9, [[("grey: ", {"bold": True, "color": GREY}), ("stored zeros, 1 pixel wide", {})],
                              [("amber: ", {"bold": True, "color": AMBER}), ("a 3×3 window at the corner reads zeros, no if-statement", {})],
                              [("teal: ", {"bold": True, "color": TEAL}), ("interior, the only part a kernel writes", {})]], size=9.5, color=MUTED)

# ------------------------------------------------------------------ 2. how it fits each kernel
s = content_slide("Does it suit every kernel? Mostly, with two costs",
    "We chose one layout for the whole network. Here is how it fits each kernel, and to be clear, this is reasoning about each kernel's loads; "
    "we have not benchmarked an alternative layout for any of them. "
    "Stride 1: close to required. The 2 by 5 tile needs 8 output channels per register and one broadcast input per pair of FMAs; NCHW8c makes each a single load and the halo makes every call identical. "
    "1 by 1: fits just as well, it is a matrix multiply over channels with the same broadcast-input, vector-output pattern. But it never reads a neighbour, so the halo is pure overhead for it: memory, and skipping the border while walking a row. "
    "Stride 2: open question. Within one tap it reads every other pixel, so half of each 64-byte cache line, but across the 3 horizontal taps it reads every column, so the lines are probably fully used over the window. "
    "SPPF: its max-pool needs a 2-pixel border of minus infinity, not zeros, so it needs its own buffers.")
rows = [("Kernel", "Fit", "Why", "Cost or open question"),
        ("3×3 stride 1 (Om)", "Strong", "8 output channels = 1 load; 1 broadcast feeds 2 FMAs; halo: no edge branches", "None found"),
        ("1×1 (Simon)", "Strong", "Matrix multiply over channels: same broadcast-input, vector-output pattern", "Halo is unused: memory, and skipping the border"),
        ("3×3 stride 2 (O'Malley)", "Open", "Same loads and halo as stride 1", "Every other pixel per tap = half a cache line; all columns across 3 taps. Even/odd split untested"),
        ("SPPF max-pool (Simon)", "Own buffers", "8 channels per register: one vector max per 8 channels", "Needs a 2-pixel −∞ border, not 1-pixel zeros")]
table(s, 0.5, 1.1, 9.0, 2.6, rows, (1.75, 0.9, 3.25, 3.1), size=10, hl=(3,))
text(s, 0.5, 3.95, 9.0, 0.6, [[("Shared win: ", {"bold": True, "color": INK}),
                              ("no conversion between layers. C2f's Split is a pointer offset (each 8-channel block is its own plane), and Concat is the producer writing into its slice.", {})]],
     size=11, color=MUTED)

# ------------------------------------------------------------------ 3. where it costs, what is unmeasured
s = content_slide("Where it costs us, and what we have not measured",
    "Two real costs. First, the first layer, model.0, has only 3 input channels, the RGB image. NCHW8c rounds that up to one block of 8, "
    "so 5 of every 8 multiply-adds in that layer are on zeros: the kernel does 2.7 times the useful work. The usual fix is a special first-layer kernel or layout. "
    "Second, the halo's memory: 2.5 percent at 160 by 160, but 21 percent at 20 by 20, where the border is a big share of a small plane. "
    "And to be honest about the evidence: we picked this layout for the microkernels' loads and to avoid conversions, but we have not timed any kernel on a second layout. "
    "The cheapest experiment is the 2 by 5 stride-1 kernel on plain NCHW against NCHW8c on a few layers; then an even/odd column split for stride 2.")
card(s, 0.5, 1.1, 4.3, 1.75, PINK, "c0")
text(s, 0.62, 1.18, 4.1, 0.3, "model.0: 3 input channels padded to 8", size=12.5, bold=True, color=RED)
stat(s, 0.62, 1.5, 1.6, "2.7×", "the useful work", RED)
text(s, 2.2, 1.55, 2.5, 1.25, "RGB input, C = 3 → one block of 8: 5 of 8 FMAs multiply zeros. "
     "One of O'Malley's 7 stride-2 layers (88.5 MFLOPs). Fix: a first-layer kernel or layout.", size=9.5, color=MUTED)
text(s, 0.5, 3.0, 4.3, 0.3, "Halo memory: (H+2)(W+2) / HW", size=12, bold=True)
table(s, 0.5, 3.32, 4.3, 1.2, [("Plane", "160×160", "80×80", "40×40", "20×20"), ("Extra memory", "+2.5%", "+5.1%", "+10.3%", "+21%")],
      (1.3, 0.75, 0.75, 0.75, 0.75), size=10)
card(s, 5.1, 1.1, 4.4, 3.42, SOFT, "c1")
text(s, 5.22, 1.18, 4.2, 0.3, "Not measured yet", size=12.5, bold=True)
text(s, 5.22, 1.55, 4.15, 2.9, [[("What we claim: ", {"bold": True, "color": INK}), ("the layout suits every kernel's loads and removes conversions between layers.", {})],
                               [("What we have not done: ", {"bold": True, "color": INK}), ("timed any kernel on a second layout.", {})],
                               [("Next experiments:", {"bold": True, "color": INK})],
                               "1. 2×5 stride-1 kernel: plain NCHW vs NCHW8c on a few layers",
                               "2. Stride 2: even/odd column split vs NCHW8c",
                               "3. A model.0 kernel that does not pad C = 3 to 8",
                               "4. 1×1: output without a halo when the next layer is 1×1"],
     size=10.5, color=MUTED)

prs.save(OUT)
print("wrote", OUT)
