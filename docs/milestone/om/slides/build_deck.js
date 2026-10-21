// Om's milestone slides (3x3 stride-1 kernel). Builds om_s1_kernel.pptx.
//   node docs/milestone/om/slides/build_deck.js
// Numbers are from the ece028 CSVs in results/ (see ../make_figs.py and ../report.tex).
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");

const THEME = {
  name: "FastYolo",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2329", lt1: "FFFFFF", dk2: "4B5563", lt2: "F3F4F6",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "9AA1AB",
    accent5: "1E2761", accent6: "CADCFC", hlink: "2A78D6", folHlink: "1E2761",
  },
};
const HEX = { ink: "1F2329", muted: "6B7280", grid: "E5E7EB", blue: "2A78D6", orange: "EB6834",
              aqua: "1BAF7A", gray: "9AA1AB", navy: "1E2761", ice: "CADCFC", soft: "F3F4F6" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "3x3 stride-1 convolution kernel";
pres.author = "Om Tank";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.2, fontSize: 40, bold: true,
                                color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.9, w: 8.8, h: 1.4, fontSize: 18,
                                color: C.accent6, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 30, bold: true,
                                color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Om Tank · 18-645 · 3x3 stride-1 kernel", options: { x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
});

const T = (slide, text, o) => slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1 }, o));
function stat(slide, x, y, w, big, small, color) {
  T(slide, big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: color || C.accent1 });
  T(slide, small, { x, y: y + 0.78, w, h: 0.6, fontSize: 13, color: C.text2, valign: "top" });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || HEX.soft }, line: { color: fill || HEX.soft } });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
                    legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted,
                    catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, valGridLine: { color: HEX.grid, size: 0.75 },
                    catGridLine: { style: "none" }, catAxisLineShow: false, valAxisLineShow: false };

// 1. Title ---------------------------------------------------------------------------------------
pres.addSection({ title: "Kernel" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Kernel" });
s.addText("3×3 stride-1 convolution", { placeholder: "title" });
s.addText("Om Tank  ·  fast YOLOv8n on a Broadwell Xeon (ece028)  ·  32 layers, 66% of the model", { placeholder: "body" });
s.addNotes("I own the 3x3 stride-1 convolution. It is 32 of YOLOv8n's 64 conv layers and two thirds of its FLOPs, so it is most of the network.");

// 2. What the kernel is ----------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Kernel" });
s.addText("One output = one long chain of multiply-adds", { placeholder: "title" });
card(s, 0.5, 1.25, 4.4, 3.7);
T(s, "y[k][h][w] = SiLU( b[k] + Σc Σr Σs  x[c][h+r−1][w+s−1] · W[k][c][r][s] )", { x: 0.75, y: 1.45, w: 4.0, h: 0.7, fontSize: 14, fontFace: "Cambria", italic: true });
T(s, [
  { text: "Independent: ", options: { bold: true } }, { text: "every output element, 25,600 to 512,000 per layer", options: { breakLine: true } },
  { text: "Dependent: ", options: { bold: true } }, { text: "9C fused multiply-adds into one sum (144 to 2,304 long)", options: { breakLine: true } },
  { text: "Units: ", options: { bold: true } }, { text: "2 FMA ports, 2 load ports, 1 store port", options: { breakLine: true } },
  { text: "SIMD: ", options: { bold: true } }, { text: "AVX2 + FMA3, 16 registers × 8 floats", options: { breakLine: true } },
  { text: "Special units: ", options: { bold: true } }, { text: "none for FP32 (no AVX-512); BatchNorm is already folded into W" },
], { x: 0.75, y: 2.25, w: 4.0, h: 2.6, fontSize: 14, valign: "top", paraSpaceAfter: 6 });
stat(s, 5.4, 1.35, 4.1, "32 layers", "the 2 convs in every C2f bottleneck + the first 2 of each Detect branch");
stat(s, 5.4, 2.6, 4.1, "5.80 of 8.74", "GFLOP per 640×640 image: 66% of the whole model", C.accent2);
stat(s, 5.4, 3.85, 4.1, "≤ 2.6e-5", "max relative error vs the real model's outputs, all layers (limit 1e-4)", C.accent3);
s.addNotes("Every output is an independent dot product of length 9C, so there is plenty of parallelism. Inside one output, the reduction is a dependent chain of FMAs. That chain is what we have to hide. Everything is checked against the baseline's own outputs on real images.");

// 3. Peak --------------------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Kernel" });
s.addText("Peak: 32 FLOPs/cycle, if 10 sums are in flight", { placeholder: "title" });
s.addTable([
  [{ text: "Instruction", options: { bold: true, color: HEX.muted } }, { text: "Latency", options: { bold: true, color: HEX.muted } },
   { text: "Throughput", options: { bold: true, color: HEX.muted } }, { text: "Ports", options: { bold: true, color: HEX.muted } }],
  ["vfmadd231ps (8-lane FMA)", "5 cycles", "2 / cycle", "0, 1"],
  ["vbroadcastss (1 input → 8 lanes)", "≈5–6", "2 / cycle", "2, 3"],
  ["vmovaps load (8 weights)", "≈5–6", "2 / cycle", "2, 3"],
  ["vmovaps store", "–", "1 / cycle", "4"],
], { x: 0.5, y: 1.3, w: 5.4, colW: [2.6, 0.9, 1.1, 0.8], fontSize: 13, fontFace: "Calibri", color: HEX.ink,
     border: { type: "solid", color: HEX.grid, pt: 0.75 }, rowH: 0.42 });
card(s, 6.3, 1.3, 3.2, 3.6, HEX.navy);
T(s, "2 FMA/cycle × 8 lanes × 2 FLOPs", { x: 6.55, y: 1.5, w: 2.8, h: 0.5, fontSize: 14, color: C.accent6 });
T(s, "= 32", { x: 6.55, y: 1.95, w: 2.8, h: 0.75, fontSize: 40, bold: true, color: C.background1 });
T(s, "FLOPs/cycle/core", { x: 6.55, y: 2.65, w: 2.8, h: 0.35, fontSize: 13, color: C.accent6 });
T(s, "5-cycle latency × 2 per cycle", { x: 6.55, y: 3.2, w: 2.8, h: 0.4, fontSize: 14, color: C.accent6 });
T(s, "= 10 sums", { x: 6.55, y: 3.55, w: 2.8, h: 0.6, fontSize: 30, bold: true, color: C.background1 });
T(s, "must be independent to never wait", { x: 6.55, y: 4.15, w: 2.8, h: 0.4, fontSize: 13, color: C.accent6 });
T(s, "ece028: Xeon E5-2640 v4 (Broadwell), 2 × 10 cores, 32 KB L1 / 256 KB L2 per core, 25 MB L3 per socket. 1 core runs at 3.39 GHz; 20 cores = 640 FLOPs/cycle.",
  { x: 0.5, y: 3.6, w: 5.4, h: 0.9, fontSize: 13, color: C.text2, valign: "top" });
s.addNotes("Peak per core is 2 FMAs per cycle times 8 lanes times 2 FLOPs, 32 FLOPs per cycle. The catch is latency: an FMA takes 5 cycles, so to start 2 per cycle we need 10 independent running sums. SiLU and bias are negligible: SiLU off gains under 1 percent.");

// 4. Register tile -------------------------------------------------------------------------------
pres.addSection({ title: "Design" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design" });
s.addText("The 2×5 register tile: 80 outputs, 13 of 16 registers", { placeholder: "title" });
const gx = 1.55, gy = 2.05, cw = 0.62, ch = 0.55, gap = 0.08;
T(s, "5 output pixels  w … w+4", { x: gx, y: 1.3, w: 5 * cw + 4 * gap, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
for (let j = 0; j < 5; j++) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx + j * (cw + gap), y: 1.62, w: cw, h: 0.32, rectRadius: 0.05,
    fill: { color: "FDEBE2" }, line: { color: HEX.orange, width: 1 } });
  T(s, `x[${j}]`, { x: gx + j * (cw + gap), y: 1.62, w: cw, h: 0.32, fontSize: 11, align: "center", valign: "middle", fontFace: "Courier New" });
  for (let k = 0; k < 2; k++) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx + j * (cw + gap), y: gy + k * (ch + gap), w: cw, h: ch, rectRadius: 0.06,
      fill: { color: "E3EEFB" }, line: { color: HEX.blue, width: 1.25 } });
    T(s, `acc\n${k},${j}`, { x: gx + j * (cw + gap), y: gy + k * (ch + gap), w: cw, h: ch, fontSize: 11, align: "center", valign: "middle", fontFace: "Courier New" });
  }
}
for (let k = 0; k < 2; k++) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: gy + k * (ch + gap), w: 0.85, h: ch, rectRadius: 0.06,
    fill: { color: "DDF4EA" }, line: { color: HEX.aqua, width: 1.25 } });
  T(s, `wv[${k}]`, { x: 0.5, y: gy + k * (ch + gap), w: 0.85, h: ch, fontSize: 11, align: "center", valign: "middle", fontFace: "Courier New" });
}
T(s, "8 output channels\nper row", { x: 0.5, y: 3.35, w: 1.0, h: 0.5, fontSize: 11, color: C.text2 });
T(s, [
  { text: "Per step (1 input channel, 1 tap): ", options: { bold: true } },
  { text: "2 weight loads + 5 broadcasts feed 10 FMAs. Every weight meets every input.", options: { breakLine: true } },
], { x: 0.5, y: 3.95, w: 5.0, h: 0.9, fontSize: 13, valign: "top" });
card(s, 5.9, 1.3, 3.6, 3.6);
T(s, [
  { text: "Size", options: { bold: true, breakLine: true } },
  { text: "2 blocks × 8 ch × 5 px = 80 outputs", options: { breakLine: true } },
  { text: "Registers", options: { bold: true, breakLine: true } },
  { text: "10 sums + 2 weights + 1 broadcast = 13", options: { breakLine: true } },
  { text: "Reuse", options: { bold: true, breakLine: true } },
  { text: "weight ×5, input ×2, sums stay in registers for all 9C steps, stored once; bias = initial value", options: { breakLine: true } },
  { text: "Why 2×5", options: { bold: true, breakLine: true } },
  { text: "divides every width (160, 80, 40, 20) and every K/8; 7 loads per 10 FMAs fits the 2 load ports" },
], { x: 6.1, y: 1.45, w: 3.25, h: 3.35, fontSize: 13, valign: "top", paraSpaceAfter: 3 });
s.addNotes("The microkernel computes 2 blocks of 8 output channels times 5 pixels. Each step loads 2 weight vectors and broadcasts 5 inputs into 10 FMAs. 13 of the 16 YMM registers, no spills in the assembly. 1x10 has the same 10 sums but needs 11 loads per 10 FMAs, more than the 2 load ports can do; 3x4 and 4x3 don't fit the channel counts or the registers. The tile size is a template parameter so the compiler fully unrolls and keeps every sum in its own register.");

// 5. Staircase -----------------------------------------------------------------------------------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Each version removes one bottleneck", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "FLOPs/cycle (1 core, all 32 layers)",
  labels: ["ref", "v0 scalar", "v1 SIMD, 1 sum", "v2 2×5, 10 sums", "v4 2×6, 12 sums", "v5 v4 + packing"],
  values: [0.29, 2.84, 3.23, 24.68, 24.17, 24.58] }],
  Object.assign({ x: 0.4, y: 1.2, w: 5.9, h: 3.85, barDir: "col", chartColors: [HEX.blue], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0", dataLabelColor: HEX.ink, dataLabelFontSize: 11, valAxisMaxVal: 32, valAxisMinVal: 0, valAxisMajorUnit: 8,
    showLegend: false, showTitle: false, barGapWidthPct: 60 }, chartText));
T(s, [
  { text: "v1: 3.23 on every layer", options: { bold: true, breakLine: true } },
  { text: "= 32 ÷ 10 exactly: one FMA per 5-cycle latency", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "v2: 24.7 = 77% of peak", options: { bold: true, breakLine: true } },
  { text: "10 independent sums hide the latency (21.3–25.8 per layer)", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "v0 is not really scalar", options: { bold: true, breakLine: true } },
  { text: "the blocked layout let the compiler vectorize it (mul + add)" },
], { x: 6.55, y: 1.35, w: 3.0, h: 3.6, fontSize: 13, valign: "top" });
s.addNotes("The textbook loop runs at 0.29. v1 measures 3.23 FLOPs per cycle on every single layer, exactly 32 over 10: one FMA every 5 cycles, the latency bound. v2 with 10 independent sums jumps to 24.7, 77 percent of peak. v4 and v5 come back on the next slides.");

// 6. Accumulator experiment ------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("More sums help until 8, then it flattens", { placeholder: "title" });
s.addChart(pres.charts.LINE, [
  { name: "model: min(sums × 3.2, 32)", labels: ["1", "2", "4", "5", "8", "10", "12"], values: [3.2, 6.4, 12.8, 16, 25.6, 32, 32] },
  { name: "measured (best tile)", labels: ["1", "2", "4", "5", "8", "10", "12"], values: [3.2, 6.3, 12.0, 14.7, 20.5, 24.6, 23.2] },
], Object.assign({ x: 0.4, y: 1.2, w: 5.9, h: 3.85, chartColors: [HEX.gray, HEX.blue], lineSize: 2.5, lineDataSymbol: "circle",
  lineDataSymbolSize: 7, valAxisMaxVal: 32, valAxisMinVal: 0, valAxisMajorUnit: 8, showLegend: true, legendPos: "t", legendFontSize: 11,
  showCatAxisTitle: true, catAxisTitle: "independent sums (tile 1×1 … 2×6)", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.muted,
  showValAxisTitle: true, valAxisTitle: "FLOPs / cycle", valAxisTitleFontSize: 11, valAxisTitleColor: HEX.muted }, chartText));
stat(s, 6.6, 1.35, 2.9, "90–101%", "of the latency model with up to 5 sums");
stat(s, 6.6, 2.6, 2.9, "77–80%", "of the model with 8 to 12 sums", C.accent2);
T(s, "1×10 has 10 sums too but needs 11 loads per 10 FMAs: load-bound at 21.4", { x: 6.6, y: 3.9, w: 2.9, h: 0.9, fontSize: 13, color: C.text2, valign: "top" });
s.addNotes("Same microkernel, different tile shapes, averaged over 4 layers. Up to 5 sums we land on the latency model. From 8 sums on we flatten around 77 to 80 percent. 12 sums (v4) does not help. So past 10, latency is no longer what limits us.");

// 7. Where the 23% goes ------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Where the last 23% goes", { placeholder: "title" });
const cols = [
  ["2.95 / 4", "uops issued per cycle: the front end has room", C.accent3],
  ["80%", "of cycles each FMA port is busy, almost only with FMAs (7.75 G uops for 7.39 G FMAs)", C.accent1],
  ["24%", "of cycles allocation stalls; half on a full scheduler, none on ROB or store buffer", C.accent2],
];
cols.forEach(([big, small, col], i) => { card(s, 0.5 + i * 3.08, 1.3, 2.84, 2.05); stat(s, 0.7 + i * 3.08, 1.45, 2.5, big, small, col); });
card(s, 0.5, 3.55, 9.0, 1.4, "E3EEFB");
T(s, [
  { text: "Reading: ", options: { bold: true } },
  { text: "the core issues faster than 2 FMA units retire, the scheduler fills with FMAs waiting on their running sums, and the next steps' broadcast loads can't enter early, so their latency shows. That is why 12 sums (v4) and one-stream weight packing (v5) did not help. Fix: issue loads one step ahead (software pipelining), as BLIS does in assembly.", options: {} },
], { x: 0.75, y: 3.65, w: 8.5, h: 1.2, fontSize: 13, valign: "middle" });
s.addNotes("perf counters on the C=64 layer, 1000 runs. Front end: 2.95 of 4 uops per cycle, not the limit. FMA ports: busy 80 percent, and almost everything on them is an FMA, so nothing steals slots. Allocation stalls 24 percent of cycles, half on the scheduler. SiLU off gains under 1 percent. Next step is software pipelining the loads.");

// 8. Memory layout ----------------------------------------------------------------------------------
pres.addSection({ title: "Layout" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Layout" });
s.addText("Intermediate layout: NCHW8c with a zero halo", { placeholder: "title" });
const sx = 1.6, sw2 = 0.45;
T(s, "NCHW", { x: 0.5, y: 1.45, w: 1.0, h: 0.4, fontSize: 14, bold: true });
T(s, "NCHW8c", { x: 0.5, y: 2.35, w: 1.0, h: 0.4, fontSize: 14, bold: true });
for (let i = 0; i < 16; i++) {
  const blueA = i < 8;
  s.addShape(pres.shapes.RECTANGLE, { x: sx + i * (sw2 + 0.03), y: 1.4, w: sw2, h: 0.45, fill: { color: blueA ? "E3EEFB" : HEX.soft }, line: { color: blueA ? HEX.blue : HEX.grid, width: 0.75 } });
  T(s, blueA ? "c0" : "c1", { x: sx + i * (sw2 + 0.03), y: 1.4, w: sw2, h: 0.45, fontSize: 10, align: "center", valign: "middle", fontFace: "Courier New" });
  s.addShape(pres.shapes.RECTANGLE, { x: sx + i * (sw2 + 0.03), y: 2.3, w: sw2, h: 0.45, fill: { color: blueA ? "E3EEFB" : HEX.soft }, line: { color: blueA ? HEX.blue : HEX.grid, width: 0.75 } });
  T(s, `c${i % 8}`, { x: sx + i * (sw2 + 0.03), y: 2.3, w: sw2, h: 0.45, fontSize: 10, align: "center", valign: "middle", fontFace: "Courier New" });
}
T(s, "channel 0 of every pixel, then channel 1 …", { x: sx, y: 1.9, w: 7.9, h: 0.3, fontSize: 11, color: C.text2 });
T(s, "pixel 1: channels 0–7 = one register, one aligned load", { x: sx, y: 2.8, w: 7.9, h: 0.3, fontSize: 11, color: C.text2 });
const items = [
  ["Halo", "1-pixel ring of zeros stored around every plane: no edge tests, every call identical"],
  ["Weights", "packed once, offline: [K/8][C/8][3][3][8 in][8 out]; v5 interleaves 2 blocks"],
  ["Shared", "all three kernels use it, so layers chain with no conversion; C2f Split/Concat become free"],
];
items.forEach(([h, b], i) => {
  card(s, 0.5 + i * 3.08, 3.35, 2.84, 1.6);
  T(s, h, { x: 0.7 + i * 3.08, y: 3.45, w: 2.5, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  T(s, b, { x: 0.7 + i * 3.08, y: 3.8, w: 2.5, h: 1.1, fontSize: 12, valign: "top" });
});
s.addNotes("ONNX's NCHW scatters the 8 channels of a pixel. NCHW8c puts them together, one aligned load per register. The halo stores the padding so the 3x3 window never branches. The weights are packed offline. The whole team uses this one layout, which makes Split and Concat free. Weight slabs are 576 times C bytes per tile pair; only the two C=256 layers overflow L2 and drop to about 70 percent.");

// 9. Baselines ------------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Layout" });
s.addText("Against the baselines (1 core, real cycles)", { placeholder: "title" });
const G = require("./baseline_groups.json"); // from results/ (see commit); exact per-shape FLOPs/cycle
s.addChart(pres.charts.BAR, [
  { name: "OpenCV DNN", labels: G.labels, values: G.ocv },
  { name: "ours (v2)", labels: G.labels, values: G.ours },
  { name: "ONNX Runtime", labels: G.labels, values: G.ort },
], Object.assign({ x: 0.4, y: 1.15, w: 6.4, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [HEX.orange, HEX.blue, HEX.aqua],
  valAxisMaxVal: 32, valAxisMinVal: 0, valAxisMajorUnit: 8, showLegend: true, legendPos: "t", legendFontSize: 11, catAxisLabelFontSize: 9,
  barGapWidthPct: 40, showValAxisTitle: true, valAxisTitle: "FLOPs / cycle (peak 32)", valAxisTitleFontSize: 11, valAxisTitleColor: HEX.muted }, chartText));
stat(s, 7.0, 1.3, 2.5, "77%", "ours: 24.7 FLOPs/cycle, 69.5 ms for all 32 layers", C.accent1);
stat(s, 7.0, 2.5, 2.5, "62%", "OpenCV DNN: 19.8, 86.4 ms", C.accent2);
stat(s, 7.0, 3.7, 2.5, "86%", "ONNX Runtime: 27.4, 62.6 ms", C.accent3);
s.addNotes("Same ONNX model, same images, single thread; baseline times converted to real cycles at the measured 3.39 GHz. OpenCV DNN reaches 62 percent of peak, ONNX Runtime 86, ours 77 and the most uniform across shapes: on the 16-channel 160 by 160 layers we beat both.");

// 10. Multi-core + next ---------------------------------------------------------------------------------
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Layout" });
s.addText("20 cores: 13.7× faster, 1,140 GFLOP/s", { placeholder: "title" });
s.addText("v3 = v2 + OpenMP, 74% of the 20-core peak at base clock.  Next: software-pipelined microkernel for the last ~20%, cache blocking for the C=256 layers, NUMA-aware threading, batch 16.", { placeholder: "body" });
s.addNotes("v3 runs all 32 layers in 5.1 ms on 20 cores versus 69.5 ms on one, 13.7 times, 1140 GFLOP/s. Next steps: software pipelining, cache blocking for the two 256-channel layers, NUMA first-touch and batch 16.");

const out = path.join(__dirname, "om_s1_kernel.pptx");
pres.writeFile({ fileName: out }).then(async () => { await applyTheme(out, THEME); console.log("wrote", out); });
