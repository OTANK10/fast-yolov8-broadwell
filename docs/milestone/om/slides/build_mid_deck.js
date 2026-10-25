// Om's mid-project presentation (5 min, 6 slides): the 3x3 stride-1 microkernel.
//   node docs/milestone/om/slides/build_mid_deck.js   (run from the repo root)
// Every number is an ece028 measurement in results/ (see the talk brief doc).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const SKILL = "/root/.claude/skills/synced/bb302a05-132c-4cd9-b920-69a3b164e959_5ab3dc19-579c-4c3a-b291-19db009d8a1a/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const ROOT = path.resolve(__dirname, "../../../..");
const OUT = path.join(__dirname, "om_mid_presentation.pptx");
const fig = (p) => path.join(ROOT, p);

const THEME = {
  name: "FastYoloStride1",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2630", lt1: "FFFFFF", dk2: "5A6875", lt2: "EEF2F4",
    accent1: "0C6A87", accent2: "B7731A", accent3: "6B7782", accent4: "D7EAF0",
    accent5: "0E2F3B", accent6: "F5E6CF", hlink: "0C6A87", folHlink: "0E2F3B",
  },
};
const HEX = { ink: "1B2630", muted: "5A6875", grid: "DDE3E7", teal: "0C6A87", amber: "B7731A", grey: "6B7782",
              tealSoft: "D7EAF0", amberSoft: "F5E6CF", soft: "EEF2F4", navy: "0E2F3B" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "3x3 stride-1 convolution microkernel";
pres.author = "Om Tank";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.9, w: 8.8, h: 1.3, fontSize: 40, bold: true, align: "left",
                                color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.3, w: 8.8, h: 0.8, fontSize: 18, align: "left",
                                color: C.accent4, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.7, fontSize: 26, bold: true, align: "left",
                                color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Om Tank · 18-645 · 3×3 stride-1 kernel · ece028 (Xeon E5-2640 v4)",
              options: { x: 0.5, y: 5.25, w: 7, h: 0.25, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: HEX.muted, align: "right" },
});

const T = (slide, text, o) => slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top" }, o));
const card = (slide, x, y, w, h, fill, name) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
                    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
                    dataLabelFontSize: 11, dataLabelColor: HEX.ink, valGridLine: { color: HEX.grid, size: 0.75 },
                    catGridLine: { style: "none" }, valAxisLineShow: false };
function stat(slide, x, y, w, big, small, color) {
  T(slide, big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: color || C.accent1, fontFace: "Cambria" });
  T(slide, small, { x, y: y + 0.78, w, h: 0.6, fontSize: 13, color: C.text2 });
}

// ---------------------------------------------------------------- 1. intro
pres.addSection({ title: "Intro" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Intro" });
  s.addText("The 3×3 stride-1 convolution kernel", { placeholder: "title" });
  s.addText("Om Tank · YOLOv8n on one Broadwell core, built like the Assignment 2 kernel", { placeholder: "body" });
  const stats = [["32", "layers: C2f bottlenecks + Detect head"], ["66%", "of YOLOv8n's 8.74 GFLOP"], ["94%", "of peak: best microkernel alone"], ["84%", "of peak across all 32 real layers"]];
  stats.forEach(([big, small], i) => {
    const x = 0.6 + i * 2.25;
    T(s, big, { x, y: 3.45, w: 2.0, h: 0.75, fontSize: 38, bold: true, color: C.background1, fontFace: "Cambria" });
    T(s, small, { x, y: 4.2, w: 1.95, h: 0.6, fontSize: 13, color: C.accent4 });
  });
  s.addNotes("My kernel is the 3x3 stride-1 convolution: 32 layers, the C2f bottlenecks and the Detect head, two thirds of YOLOv8n's work. " +
    "The story: it is the Assignment 2 kernel, I followed the same four steps, and the measurements chose the tile. " +
    "Best microkernel alone: 94% of the 32 FLOPs/cycle peak; inside all 32 real layers: 84%. All percentages assume the core runs at 3.4 GHz, as in Assignment 2.");
}

// ---------------------------------------------------------------- 2. same as A2
pres.addSection({ title: "Method" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("It is the Assignment 2 kernel, with floats", { placeholder: "title" });
  const hdr = { bold: true, color: HEX.ink, fill: { color: HEX.soft } };
  const rows = [
    [{ text: "", options: hdr }, { text: "Assignment 2", options: hdr }, { text: "Stride-1 microkernel", options: hdr }],
    ["m (broadcast)", "5 rows of A", "5 output pixels"],
    ["n (vector loads)", "8 = 2 × 4 doubles", "16 channels = 2 × 8 floats"],
    ["k (reduction)", "32 to 512", "C × 9 (144 to 2304)"],
    ["Accumulators", "10", "10 (2×5) or 12 (2×6)"],
    ["Registers", "13 of 16", "13 or 15 of 16"],
    ["Peak per core", "16 FLOPs/cycle", "32 FLOPs/cycle"],
  ];
  s.addTable(rows, { x: 0.5, y: 1.2, w: 4.9, colW: [1.45, 1.55, 1.9], fontSize: 12, fontFace: "Calibri", color: HEX.ink,
                     border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.42, valign: "middle", objectName: "a2-table" });
  T(s, "Same Broadwell core: 2 FMA units × 8 lanes × 2 FLOPs = 32 FLOPs/cycle. Latency 5 × throughput 2 = 10 chains in flight.",
    { x: 0.5, y: 4.3, w: 4.9, h: 0.7, fontSize: 13, color: C.text2 });
  // the four terms the rest of the talk uses
  const defs = [["C", "input channels of the layer (16 to 256)"],
                ["k = C × 9", "multiply-adds per output: C channels × the 9 taps of the 3×3 filter (A2's k)"],
                ["accumulator", "one YMM register = 8 outputs (8 channels of one pixel); one dependency chain"],
                ["tile 2×5 (10)", "outputs per call: 2 blocks of 8 channels × 5 pixels = 80 outputs in 10 accumulators"]];
  defs.forEach(([h, b], i) => {
    const y = 1.2 + i * 0.95;
    card(s, 5.65, y, 3.9, 0.85, i === 3 ? HEX.tealSoft : HEX.soft, `term-${i}`);
    T(s, h, { x: 5.8, y: y + 0.08, w: 3.65, h: 0.28, fontSize: 13, bold: true, color: i === 3 ? C.accent1 : C.text1, fontFace: "Courier New" });
    T(s, b, { x: 5.8, y: y + 0.38, w: 3.65, h: 0.45, fontSize: 11, color: C.text2 });
  });
  s.addNotes("Same machine as Assignment 2 and the same kernel. In A2, m was 5 rows of A broadcast, n was 8 doubles = 2 vectors of B. " +
    "Here m is 5 output pixels, n is 16 output channels = 2 vectors of 8 floats, and k is input channels times the 9 taps of the 3x3 filter. " +
    "10 accumulators, 13 registers, exactly the A2 budget. Peak is 32 FLOPs per cycle because floats give 8 lanes instead of 4. " +
    "Terms: C is input channels; k = C x 9 is the length of each output's sum; an accumulator is one register holding 8 outputs; the tile is the block of outputs one call computes.");
}

// ---------------------------------------------------------------- 2b. the tile as tensors
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("One kernel call = 80 outputs", { placeholder: "title" });
  s.addImage({ path: fig("docs/milestone/om/figs/tile_tensors.png"), x: 0.5, y: 1.2, w: 9.0, h: 2.62,
               altText: "Input map with a 3 by 7 window across all C channels, 16 filters of 3 by 3 by C, and the output tile of 1 row, 5 pixels and 16 channels" });
  T(s, "Per step (one input channel at one of the 9 taps): load 2 weight vectors, broadcast 5 inputs, 10 FMAs. Repeat k = C × 9 times, then bias, SiLU and one store per register.",
    { x: 0.5, y: 4.15, w: 9.0, h: 0.8, fontSize: 15 });
  s.addNotes("One call of the microkernel computes this teal strip: one output row, 5 pixels wide, 16 output channels deep, 80 numbers. " +
    "Each cell is 8 channels of one pixel, which is exactly one YMM register, so the strip is 10 accumulators. " +
    "To compute it, the kernel reads a 3 by 7 window of the input across all C channels and 16 filters of 3 by 3 by C. " +
    "Each step multiplies 2 weight vectors by 5 broadcast inputs, 10 FMAs, and it repeats that k = C times 9 times before applying bias and SiLU and storing.");
}

// ---------------------------------------------------------------- 3. four steps
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Four steps: ops → accumulators → tile → benchmark", { placeholder: "title" });
  const steps = [
    ["Independent ops", "Every output is its own dot product of length C × 9: 102,400 in a 40×40, 64-channel layer."],
    ["Outputs in flight", "Latency 5 × 2 FMA/cycle = 10 accumulators; 1 accumulator runs at exactly 10% of peak."],
    ["Kernel size", "2×5 tile: 10 acc + 2 weights + 1 broadcast = 13 regs; 7 loads per 10 FMAs; 5 divides every width."],
    ["Benchmark like A2", "Microkernel alone, k = 72 to 4608, 3 runs, 2 s warm-up, cycles at 3.4 GHz as in A2, checked vs a reference."],
  ];
  steps.forEach(([h, b], i) => {
    const y = 1.2 + i * 0.98;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.02, w: 0.42, h: 0.42, fill: { color: HEX.teal }, line: { color: HEX.teal }, objectName: `step-${i + 1}` });
    T(s, String(i + 1), { x: 0.5, y: y + 0.02, w: 0.42, h: 0.42, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, h, { x: 1.1, y, w: 3.6, h: 0.3, fontSize: 15, bold: true });
    T(s, b, { x: 1.1, y: y + 0.32, w: 3.6, h: 0.62, fontSize: 12, color: C.text2 });
  });
  const tiles = ["1×1 (1)", "2×2 (4)", "1×5 (5)", "2×4 (8)", "1×10 (10)", "2×5 (10)"];
  const vals = [9.9, 39.3, 49.6, 72.4, 85.8, 85.3]; // bench_ukr 64 --no-silu (3.4 GHz), results/ukr_C64_nosilu_34.csv
  s.addChart(pres.charts.BAR, [{ name: "% of peak", labels: tiles, values: vals }], Object.assign({}, chartText, {
    x: 5.0, y: 1.05, w: 4.55, h: 3.95, barDir: "col", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0\"%\"", valAxisMaxVal: 100, valAxisMinVal: 0, valAxisMajorUnit: 25, valAxisLabelFormatCode: "0\"%\"",
    showLegend: false, showTitle: true, title: "Microkernel alone vs accumulators (C = 64, SiLU off)", titleFontSize: 12, titleColor: HEX.ink,
    catAxisTitle: "tile (accumulators)", showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisTitleColor: HEX.muted, objectName: "accumulator-chart" }));
  s.addNotes("Step 1: every output is an independent dot product, so there is plenty of parallelism; the only chain is the reduction. " +
    "Step 2: 5-cycle latency times 2 FMAs per cycle means 10 independent accumulators. The chart shows it: 1 accumulator is 10%, each one adds about 10 points. " +
    "Step 3: 2x5 gives 10 accumulators in 13 registers, 7 loads per 10 FMAs, and 5 divides 160, 80, 40 and 20. " +
    "Step 4: benchmark the kernel alone exactly like Assignment 2.");
}

// ---------------------------------------------------------------- 4. design space
pres.addSection({ title: "Results" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Same math, 50% to 95%: tile and register allocation", { placeholder: "title" });
  s.addImage({ path: fig("results/ukr_design_space.png"), x: 0.5, y: 1.0, w: 7.5, h: 4.08, altText: "Design space: percent of peak for every stride-1 kernel, alone and in all 32 layers" });
  const pts = [["Intrinsics", "GCC: clean at 13 regs, copies at 15, spills at 16"],
               ["Inline asm", "2×6 by hand: 95% alone, 84% in layers"],
               ["Unroll, pipeline", "lost: uop cache, spills"]];
  pts.forEach(([h, b], i) => {
    const y = 1.05 + i * 1.37;
    card(s, 8.15, y, 1.4, 1.25, i === 1 ? HEX.tealSoft : HEX.soft, `ds-card-${i}`);
    T(s, h, { x: 8.25, y: y + 0.08, w: 1.22, h: 0.3, fontSize: 12, bold: true, color: i === 1 ? C.accent1 : C.text1 });
    T(s, b, { x: 8.25, y: y + 0.4, w: 1.22, h: 0.82, fontSize: 11, color: C.text2 });
  });
  s.addNotes("Every kernel I built, same arithmetic, same layout. Filled dots are the microkernel alone, hollow are all 32 layers. " +
    "Left: the accumulator count. Right: the 2x6 tile my teammates suggested loses with intrinsics because GCC adds 11 register copies per tap at 15 live registers, " +
    "but with the registers fixed in inline asm it is the best kernel, 95% alone. 4x3 needs all 16 registers; GCC spills it to 49%. " +
    "The lecture techniques, unrolling and software pipelining, made it worse: full unrolling overflowed the uop cache and partial unrolling made GCC spill.");
}

// ---------------------------------------------------------------- 5. sweep
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Benchmarked like Assignment 2: FLOPs/cycle vs k", { placeholder: "title" });
  s.addImage({ path: fig("results/ukr_sweep_nosilu_34.png"), x: 0.5, y: 1.0, w: 6.5, h: 3.47, altText: "FLOPs per cycle vs k for the 2x5, 2x6 and 2x6 asm microkernels with SiLU off" });
  T(s, "SiLU off, cycles = time × 3.4 GHz, as in A2. 3 runs, every point checked (576 pass).", { x: 0.5, y: 4.55, w: 6.5, h: 0.5, fontSize: 12, color: C.text2 });
  stat(s, 7.3, 1.05, 2.25, "94–97%", "2×6 asm: 30–31 FLOPs/cycle, ~102 GFLOP/s on one core");
  stat(s, 7.3, 2.45, 2.25, "83–89%", "2×5 intrinsics (v2's tile)", C.accent3);
  stat(s, 7.3, 3.85, 2.25, "k ≈ 500", "L1 dip: 2×5 weights pass 32 KB, like k ≈ 315 in A2", C.accent2);
  s.addNotes("This is the Assignment 2 figure for my kernel: FLOPs per cycle against the reduction length k. " +
    "Cycles are time times 3.4 GHz, the max turbo, exactly as in Assignment 2, so this can only understate. 2x6 asm holds 30 to 31 FLOPs per cycle, 94 to 97 percent of peak. 2x5 intrinsics is 83 to 89. " +
    "The dip near k = 500 is the 2x5 weights leaving L1, the same effect as k = 315 in A2. " +
    "Cross-checked: the hardware FP-instruction counter matches our FLOP count exactly, and real perf cycles give the same or slightly higher numbers.");
}

// ---------------------------------------------------------------- 6. alone vs layers + baselines
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Alone vs in a layer: keep the kernel a function", { placeholder: "title" });
  T(s, "v2a (2×5 asm): 92.2% alone, 72.5% when GCC inlines it into the layer loop", { x: 0.5, y: 1.1, w: 4.6, h: 0.55, fontSize: 14, bold: true });
  card(s, 0.5, 1.75, 4.6, 1.15, HEX.soft, "spill-code");
  T(s, "vmovaps 168(%rsp), %ymm1   # reload acc\n  ...asm block: 80 FMAs...\nvmovaps %ymm1, 168(%rsp)   # store acc",
    { x: 0.65, y: 1.85, w: 4.35, h: 0.95, fontSize: 11, fontFace: "Courier New", color: C.text1, valign: "middle" });
  T(s, "Inlined, GCC keeps 3 of 10 accumulators on the stack around every tap. Called as a function, as in the benchmark: 83.1% (v2b); with the 2×6 tile, 83.7% (v10).",
    { x: 0.5, y: 3.0, w: 4.6, h: 0.95, fontSize: 13, color: C.text2 });
  T(s, "Next: cache blocking for C ≥ 128 (the 4×3 drop-off), NUMA-aware threads (v10 on 20 cores: 4.74 ms).",
    { x: 0.5, y: 4.15, w: 4.6, h: 0.8, fontSize: 13, italic: true, color: C.accent1 });
  const labels = ["OpenCV DNN", "v2 (intrinsics)", "v10 (2×6 asm)", "ONNX Runtime"];
  const ms = [86.4, 69.5, 63.8, 62.6];
  s.addChart(pres.charts.BAR, [{ name: "ms, 32 layers", labels, values: ms }], Object.assign({}, chartText, {
    x: 5.35, y: 1.05, w: 4.2, h: 3.95, barDir: "bar", chartColors: [HEX.grey], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0\" ms\"", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, showLegend: false, showTitle: true,
    title: "All 32 layers, 1 thread, SiLU on (lower is better)", titleFontSize: 12, titleColor: HEX.ink, catAxisOrientation: "maxMin",
    objectName: "baseline-chart" }));
  s.addNotes("Kernel A, 2x5 in asm, beat kernel B, 2x5 intrinsics, by 7 points alone but lost by 4 in the layers. " +
    "The assembly shows why: inlined into the layer loop, GCC keeps 3 accumulators on the stack and reloads them around every tap. " +
    "Calling the microkernel as a function, exactly as the benchmark does, fixes it: 83.1% for 2x5 and 83.7% for v10, all at 3.4 GHz. " +
    "v10 is 1.36 times faster than OpenCV and within 2% of ONNX Runtime on all 32 layers. Next: cache blocking and threading.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
