#!/usr/bin/env python3
"""Team intro slide: YOLOv8n, the ece028 machine, and the kernels, CMU Digital Tartan template.

    python docs/milestone/team/slides/build_intro_slide.py   ->  intro_slide_cmu.pptx

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
OUT = os.path.join(HERE, "intro_slide_cmu.pptx")
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



s = content_slide("YOLOv8n on one Broadwell machine: what we build",
    "Quick recap, since the project is approved. We take the pretrained YOLOv8n detector and write our own FP32 inference kernels for its convolutions; no training. "
    "One image is 640 by 640, batch 1, 8.74 GFLOP, almost all of it in 63 convolution layers. "
    "Our machine is ece028, a Xeon E5-2640 v4, Broadwell: AVX2 and FMA, 16 vector registers, so one core peaks at 32 FLOPs per cycle, 2 FMA units times 8 floats times 2. "
    "The convolutions split into three kernels: 3 by 3 stride 1, 66 percent of the work; 3 by 3 stride 2, the downsampling, 15 percent; and 1 by 1, 19 percent. "
    "57 of the 63 convolutions are followed by SiLU, which we fuse into the kernels, and SPPF adds max-pooling. "
    "We compare against OpenCV DNN on the same model and images. Each of us presents our own kernel.")
s.shapes.add_picture(fig("docs/milestone/team/slides/yolov8_arch.png"), Inches(5.95), Inches(0.98), height=Inches(3.5))
text(s, 5.95, 4.5, 3.4, 0.2, "YOLOv8 architecture. Diagram: RangeKing (Ultralytics GitHub).", size=8, color=MUTED)
text(s, 0.5, 1.0, 5.25, 0.5, [[("Model: ", {"bold": True, "color": INK}),
                              ("pretrained YOLOv8n, FP32 inference, batch 1, 640×640: 8.74 GFLOP per image in 63 conv layers. Baseline: OpenCV DNN, same ONNX model.", {})]],
     size=10.5, color=MUTED)
card(s, 0.5, 1.62, 5.25, 1.12, SOFT, "machine")
text(s, 0.62, 1.68, 3.4, 0.28, "Machine: ece028", size=12, bold=True)
text(s, 0.62, 1.95, 3.5, 0.8, ["Xeon E5-2640 v4 (Broadwell), 2.4 GHz, 3.4 turbo",
                              "2 sockets × 10 cores, AVX2 + FMA, 16 YMM registers",
                              "L1D 32 KB, L2 256 KB per core; L3 25 MB per socket"], size=9.5, color=MUTED)
text(s, 4.15, 1.7, 1.5, 0.5, "32", size=30, bold=True, color=RED, align=PP_ALIGN.CENTER)
text(s, 4.05, 2.22, 1.7, 0.5, "FLOPs/cycle/core\n2 FMA × 8 lanes × 2", size=8.5, color=MUTED, align=PP_ALIGN.CENTER)
rows = [("Kernel", "Owner", "Layers", "Work"),
        ("3×3 conv, stride 1", "Om", "32", "66%"),
        ("3×3 conv, stride 2", "O'Malley", "7", "15%"),
        ("1×1 conv", "Simon", "24", "19%"),
        ("SiLU (fused into the conv)", "Simon", "57 of 63", "in the kernels"),
        ("SPPF max-pool", "Simon", "1", "< 1%")]
table(s, 0.5, 2.88, 5.25, 1.65, rows, (2.35, 1.0, 0.8, 1.1), size=10)

prs.save(OUT)
print("wrote", OUT)
