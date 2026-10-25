#!/usr/bin/env python3
"""Om's mid-project talk on the CMU Digital Tartan template (cmu_template.pptx).

    python docs/milestone/om/slides/build_cmu_deck.py   ->  om_mid_presentation_cmu.pptx

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
TEMPLATE = os.path.join(HERE, "cmu_template.pptx")
OUT = os.path.join(HERE, "om_mid_presentation_cmu.pptx")
fig = lambda p: os.path.join(ROOT, p)

RED, INK, MUTED, TEAL, TEALSOFT = RGBColor(0xBB, 0x00, 0x27), RGBColor(0x1B, 0x26, 0x30), RGBColor(0x5A, 0x68, 0x75), RGBColor(0x0C, 0x6A, 0x87), RGBColor(0xD7, 0xEA, 0xF0)
SOFT, GREY, AMBER, WHITE, PINK = RGBColor(0xF1, 0xF3, 0xF5), RGBColor(0x6B, 0x77, 0x82), RGBColor(0xB7, 0x73, 0x1A), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF8, 0xE5, 0xE9)
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


# ------------------------------------------------------------------ 1. title
s = prs.slides.add_slide(L_TITLE)
text(s, 2.33, 1.5, 6.9, 1.3, [[("The 3×3 stride-1 convolution kernel", {})]], size=34, bold=True, color=WHITE, anchor=MSO_ANCHOR.BOTTOM, name="title")
text(s, 2.33, 3.0, 6.9, 0.5, [[("Om Tank", {"bold": True}), ("  ·  YOLOv8n on one Broadwell core, built like the Assignment 2 kernel", {})]], size=15, color=WHITE)
for i, (big, small) in enumerate([("32", "layers"), ("66%", "of YOLOv8n's work"), ("94%", "of peak, alone"), ("84%", "of peak, 32 layers")]):
    text(s, 2.33 + i * 1.75, 3.85, 1.65, 0.55, big, size=26, bold=True, color=WHITE)
    text(s, 2.33 + i * 1.75, 4.4, 1.65, 0.4, small, size=11, color=WHITE)
s.notes_slide.notes_text_frame.text = (
    "My kernel is the 3x3 stride-1 convolution: 32 layers, the C2f bottlenecks and the Detect head, two thirds of YOLOv8n's work. "
    "The story: it is the Assignment 2 kernel, I followed the same four steps, and the measurements chose the tile. "
    "Best microkernel alone: 94% of the 32 FLOPs/cycle peak; inside all 32 real layers: 84%. All percentages assume the core runs at 3.4 GHz, as in Assignment 2.")

# ------------------------------------------------------------------ 2. same as A2
s = content_slide("It is the Assignment 2 kernel, with floats",
    "Same machine as Assignment 2 and the same kernel. In A2, m was 5 rows of A broadcast, n was 8 doubles = 2 vectors of B. "
    "Here m is 5 output pixels, n is 16 output channels = 2 vectors of 8 floats, and k is input channels times the 9 taps of the 3x3 filter. "
    "10 accumulators, 13 registers, exactly the A2 budget. Peak is 32 FLOPs per cycle because floats give 8 lanes instead of 4. "
    "10 accumulators is the minimum (5 x 2); the 2x6 tile uses 12, two chains of slack.")
rows = [("", "Assignment 2", "Stride-1 microkernel"), ("m (broadcast)", "5 rows of A", "5 output pixels"),
        ("n (vector loads)", "8 = 2 × 4 doubles", "16 ch = 2 × 8 floats"), ("k (reduction)", "32 to 512", "C × 9 (144 to 2304)"),
        ("Accumulators", "10", "10 (2×5) or 12 (2×6)"), ("Registers", "13 of 16", "13 or 15 of 16"), ("Peak per core", "16 FLOPs/cycle", "32 FLOPs/cycle")]
tbl = s.shapes.add_table(len(rows), 3, Inches(0.5), Inches(1.2), Inches(5.0), Inches(2.6)).table
for c, wd in enumerate((1.45, 1.6, 1.95)): tbl.columns[c].width = Inches(wd)
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = v
        cell.fill.solid(); cell.fill.fore_color.rgb = SOFT if r == 0 else WHITE
        cell.margin_left = cell.margin_right = Inches(0.06)
        f = cell.text_frame.paragraphs[0].runs[0].font if v else None
        if f: f.size, f.name, f.bold, f.color.rgb = Pt(11), FONT, r == 0 or c == 0 and False, INK
text(s, 0.5, 3.95, 5.0, 0.6, "Same Broadwell core: 2 FMA units × 8 lanes × 2 FLOPs = 32 FLOPs/cycle. Latency 5 × throughput 2 = 10 chains in flight.", size=12, color=MUTED)
for i, (h, b) in enumerate([("C", "input channels of the layer (16 to 256)"),
                            ("k = C × 9", "multiply-adds per output: C channels × the 9 taps of the 3×3 filter (A2's k)"),
                            ("accumulator", "one YMM register = 8 outputs (8 channels of one pixel); one dependency chain"),
                            ("tile 2×5 (10)", "outputs per call: 2 blocks of 8 ch × 5 pixels = 80 outputs in 10 accumulators")]):
    y = 1.2 + i * 0.84
    card(s, 5.75, y, 3.75, 0.76, PINK if i == 3 else SOFT, f"term-{i}")
    text(s, 5.88, y + 0.07, 3.5, 0.25, h, size=12, bold=True, color=RED if i == 3 else INK, font="Courier New")
    text(s, 5.88, y + 0.33, 3.5, 0.42, b, size=10, color=MUTED)

# ------------------------------------------------------------------ 3. tile as tensors
s = content_slide("One kernel call = 80 outputs",
    "One call computes the teal strip: one output row, 5 pixels wide, 16 output channels deep, 80 numbers. "
    "Each cell is 8 channels of one pixel, exactly one YMM register, so the strip is 10 accumulators. "
    "It reads a 3 by 7 window of the input across all C channels and 16 filters of 3 by 3 by C. "
    "Each step multiplies 2 weight vectors by 5 broadcast inputs, 10 FMAs, repeated k = C x 9 times, then bias, SiLU, store.")
s.shapes.add_picture(fig("docs/milestone/om/figs/tile_tensors.png"), Inches(0.5), Inches(1.15), width=Inches(9.0))
text(s, 0.5, 3.95, 7.8, 0.6, "Per step (one input channel at one of the 9 taps): load 2 weight vectors, broadcast 5 inputs, 10 FMAs. "
     "Repeat k = C × 9 times, then bias, SiLU and one store per register.", size=13)

# ------------------------------------------------------------------ 4. four steps
s = content_slide("Four steps: ops → accumulators → tile → benchmark",
    "Step 1: every output is an independent dot product; the only chain is the reduction. "
    "Step 2: 5-cycle latency times 2 FMAs per cycle means 10 independent accumulators, 80 outputs in flight. The chart: 1 accumulator is 10%, each adds about 10 points. "
    "Step 3: 2x5 gives 10 accumulators in 13 registers, 7 loads per 10 FMAs, and 5 divides 160, 80, 40 and 20. "
    "Step 4: benchmark the kernel alone exactly like Assignment 2.")
steps = [("Independent ops", "Each output is a dot product of length C × 9: 102,400 in a 40×40, 64-ch layer."),
         ("Outputs in flight", "5 × 2 FMA/cycle = 10 accumulators = 80 outputs; 1 accumulator = 10% of peak."),
         ("Kernel size", "2×5 tile: 10 acc + 2 weights + 1 broadcast = 13 regs; 7 loads per 10 FMAs."),
         ("Benchmark like A2", "Kernel alone, k = 72 to 4608, 3 runs, 2 s warm-up, cycles at 3.4 GHz, checked.")]
for i, (h, b) in enumerate(steps):
    y = 1.15 + i * 0.85
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.5), Inches(y + 0.02), Inches(0.36), Inches(0.36))
    o.fill.solid(); o.fill.fore_color.rgb = RED; o.line.fill.background(); o.shadow.inherit = False
    text(s, 0.5, y + 0.02, 0.36, 0.36, str(i + 1), size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.0, y - 0.02, 3.6, 0.28, h, size=13, bold=True)
    text(s, 1.0, y + 0.27, 3.6, 0.52, b, size=10.5, color=MUTED)
cd = CategoryChartData()
cd.categories = ["1×1 (1)", "2×2 (4)", "1×5 (5)", "2×4 (8)", "1×10 (10)", "2×5 (10)"]
cd.add_series("% of peak", (9.9, 39.3, 49.6, 72.4, 85.8, 85.3))  # results/ukr_C64_nosilu_34.csv
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.8), Inches(1.05), Inches(4.7), Inches(3.55), cd).chart
style_chart(ch, '0"%"', TEAL)
ch.value_axis.maximum_scale, ch.value_axis.minimum_scale, ch.value_axis.major_unit = 100, 0, 25
ch.value_axis.tick_labels.number_format, ch.value_axis.tick_labels.number_format_is_linked = '0"%"', False
chart_title(ch, "Microkernel alone vs accumulators (C = 64, SiLU off)")

# ------------------------------------------------------------------ 5. design space
s = content_slide("Same math, 50% to 95%: tile and register allocation",
    "Every kernel I built, same arithmetic, same layout. Filled dots: the microkernel alone; hollow: all 32 layers. "
    "The 2x6 tile loses with intrinsics because GCC adds 11 register copies per tap at 15 live registers, "
    "but with the registers fixed in inline asm it is the best kernel, 95% alone. 4x3 needs all 16 registers; GCC spills it to 49%. "
    "Unrolling and software pipelining made it worse: full unrolling overflowed the uop cache and partial unrolling made GCC spill.")
s.shapes.add_picture(fig("results/ukr_design_space.png"), Inches(0.5), Inches(1.1), height=Inches(3.5))
for i, (h, b) in enumerate([("Intrinsics", "GCC: clean at 13 regs, copies at 15, spills at 16"),
                            ("Inline asm", "2×6 by hand: 95% alone, 84% in layers"),
                            ("Unroll, pipeline", "lost: uop cache, spills")]):
    y = 1.1 + i * 1.18
    card(s, 7.05, y, 2.45, 1.05, PINK if i == 1 else SOFT, f"ds-card-{i}")
    text(s, 7.17, y + 0.08, 2.2, 0.28, h, size=12, bold=True, color=RED if i == 1 else INK)
    text(s, 7.17, y + 0.4, 2.2, 0.6, b, size=11, color=MUTED)

# ------------------------------------------------------------------ 6. sweep
s = content_slide("Benchmarked like Assignment 2: FLOPs/cycle vs k",
    "This is the Assignment 2 figure for my kernel: FLOPs per cycle against the reduction length k. "
    "Cycles are time times 3.4 GHz, the max turbo, exactly as in Assignment 2, so this can only understate. "
    "2x6 asm holds 30 to 31 FLOPs per cycle, 94 to 97 percent of peak; 2x5 intrinsics 83 to 89. "
    "The dip near k = 500 is the 2x5 weights leaving L1, like k = 315 in A2. "
    "Cross-checked: the hardware FP-instruction counter matches our FLOP count exactly.")
s.shapes.add_picture(fig("results/ukr_sweep_nosilu_34.png"), Inches(0.5), Inches(1.05), height=Inches(3.2))
text(s, 0.5, 4.3, 6.0, 0.3, "SiLU off, cycles = time × 3.4 GHz, as in A2. 3 runs, every point checked (576 pass).", size=10, color=MUTED)
stat(s, 6.75, 1.05, 2.75, "94–97%", "2×6 asm: 30–31 FLOPs/cycle, ~102 GFLOP/s on one core", RED)
stat(s, 6.75, 2.2, 2.75, "83–89%", "2×5 intrinsics (v2's tile)", GREY)
stat(s, 6.75, 3.35, 2.75, "k ≈ 500", "L1 dip: 2×5 weights pass 32 KB, like k ≈ 315 in A2", AMBER)

# ------------------------------------------------------------------ 7. alone vs layer + baselines
s = content_slide("Alone vs in a layer: keep the kernel a function",
    "Kernel A, 2x5 in asm, beat kernel B, 2x5 intrinsics, by 7 points alone but lost by 4 in the layers. "
    "The assembly shows why: inlined into the layer loop, GCC keeps 3 accumulators on the stack and reloads them around every tap. "
    "Calling the microkernel as a function, exactly as the benchmark does, fixes it: 83.1% for 2x5 and 83.7% for v10, all at 3.4 GHz. "
    "v10 is 1.36 times faster than OpenCV and within 2% of ONNX Runtime on all 32 layers. Next: cache blocking and threading.")
text(s, 0.5, 1.1, 4.6, 0.55, "v2a (2×5 asm): 92.2% alone, 72.5% when GCC inlines it into the layer loop", size=13, bold=True)
card(s, 0.5, 1.75, 4.6, 0.95, SOFT, "spill-code")
text(s, 0.62, 1.82, 4.4, 0.82, ["vmovaps 168(%rsp), %ymm1   # reload acc", "  ...asm block: 80 FMAs...", "vmovaps %ymm1, 168(%rsp)   # store acc"],
     size=10.5, font="Courier New", anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.5, 2.82, 4.6, 0.85, "Inlined, GCC keeps 3 of 10 accumulators on the stack around every tap. Called as a function, as in the benchmark: 83.1% (v2b); with the 2×6 tile, 83.7% (v10).",
     size=11.5, color=MUTED)
text(s, 0.5, 3.8, 4.6, 0.75, "Next: cache blocking for C ≥ 128 (the 4×3 drop-off), NUMA-aware threads (v10 on 20 cores: 4.74 ms).", size=11.5, italic=True, color=RED)
cd = CategoryChartData()
cd.categories = ["ONNX Runtime", "v10 (2×6 asm)", "v2 (intrinsics)", "OpenCV DNN"]
cd.add_series("ms", (62.6, 63.8, 69.5, 86.4))
ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(5.3), Inches(1.05), Inches(4.2), Inches(3.55), cd).chart
style_chart(ch, '0.0" ms"', GREY)
ch.value_axis.maximum_scale, ch.value_axis.minimum_scale, ch.value_axis.major_unit = 110, 0, 25
chart_title(ch, "32 layers, 1 thread, ms (lower is better)")

# ------------------------------------------------------------------ backup
s = content_slide("Backup: the same sweeps with SiLU on",
    "With the fused SiLU on, everything sits about 4 to 5 points lower: SiLU costs a fixed 160 to 290 cycles per call. "
    "Right: the 4x3 asm tile is the fastest while its weights are small, then collapses past C about 128 as its weights leave L2: the case cache blocking fixes.")
s.shapes.add_picture(fig("results/ukr_sweep_2x6_34.png"), Inches(0.5), Inches(1.15), width=Inches(4.4))
s.shapes.add_picture(fig("results/ukr_sweep_34.png"), Inches(5.1), Inches(1.15), width=Inches(4.4))
text(s, 0.5, 3.6, 4.4, 0.7, "2×6 asm 90.2% mean (C = 32–256); 2×5 81.4%. SiLU costs a fixed 160–290 cycles per call.", size=10.5, color=MUTED)
text(s, 5.1, 3.6, 4.4, 0.7, "4×3 asm: best 87.5% at C = 120, 57% at C = 256: its weights leave L2 (cache blocking fixes this).", size=10.5, color=MUTED)

prs.save(OUT)
print("wrote", OUT)
