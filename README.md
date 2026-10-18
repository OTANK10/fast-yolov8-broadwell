# Fast YOLOv8n on Intel Broadwell

**CMU 18-645: How to Write Fast Code — Fall 2026**  
**Status:** Starter repository; performance optimization is *not* implemented yet.

## Start here (5 minutes)

We are **not training YOLO**. We are rebuilding selected YOLOv8n inference operations as fast C++ kernels and measuring how efficiently they use the Intel Xeon E5-2640 v4 (Broadwell) CPU. Our source model is the **YOLOv8n** variant, not whatever newest YOLO model Ultralytics advertises.

1. **Build the starting C++ code** (macOS/Linux, CMake and C++17 compiler needed):

   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ./build/bench_conv
   ```

2. **Export the YOLOv8n model** (optional first day; needs Python + internet):

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python scripts/export_model.py
   ```

   Expected result: `data/models/yolov8n.onnx`. This is just a model export; **it does not yet extract per-layer reference tensors**. Keep the large model and data files untracked.

3. **Create branches:** `om/conv3x3-s1`, `sherlock/conv3x3-s2`, `simon/sppf`. Open pull requests for shared changes.

4. **Find the target machine**. Develop correctness anywhere, but run reportable AVX2/FMA performance experiments on the department's Broadwell Xeon.

### What Om is doing

Om owns **3×3, stride-1 FP32 convolution inside YOLOv8n C2f bottleneck blocks**:

- Find one actual stride-1 layer's input shape and weight shape in the ONNX model.
- First verify the scalar (slow but correct) reference, including channels and padding.
- Implement a new optimized version in `src/conv3x3_s1.cpp` using AVX2/FMA and (eventually) blocked channel layout, without silently changing the answer.
- Measure cycles/time and derive FP32 FLOPs/cycle; explain register allocation, throughput, memory layout, and why actual speed differs from theoretical peak.

**Current state:** `conv3x3_s1` is a *correctness-first placeholder* that calls the scalar reference. Benchmark output is **not** evidence of an optimized kernel.

## Who owns what

| Owner | Kernel | Milestone evidence |
| --- | --- | --- |
| **Om Tank** | 3×3 stride-1 convolution (C2f bottleneck) | Correctness, SIMD/FMA register map, performance vs. reference and peak |
| **O'Malley Sherlock** | 3×3 stride-2 convolution (downsampling) | Correctness, addressing and tiling explanation, per-layer timings |
| **Bohan (Simon) Yu** | SPPF pooling/fused operations; investigate 1×1 | Working pooling benchmark and memory-access strategy; confirm 1×1 scope with instructor |

Per Professor Low's feedback, folding batch norm into weights **does not itself add inference-time compute**, so that must not be presented as a standalone performance kernel. The included SPPF reference covers the pooling cascade only, not its surrounding convolution/activation layers.

## What are we measuring?

Model: pretrained **Ultralytics YOLOv8n**, exported to ONNX at image size 640.  
Primary baseline: **OpenCV DNN** (as specified in the proposal).  
Secondary reference: **ONNX Runtime CPU**.  
Target platform: **Intel Xeon E5-2640 v4**, 2 sockets × 10 physical cores, AVX2/FMA, 16 YMM registers, 32 KB L1D/core, 256 KB L2/core, 25 MB L3/socket.

For 32-bit FMA, Broadwell's advertised **per-core compute peak is 32 FP32 FLOPs/cycle** (two 256-bit FMA instructions per cycle × eight lanes × two FLOPs per fused multiply-add). Actual sustained rate may be lower due to dependencies, loads, addresses, issue limits, and cache effects. The 5-cycle FMA latency suggests ~10 independent accumulator chains are required to sustain peak FMA throughput, subject to other resource constraints.

For each real layer shape, report: tensor shapes, validation error (absolute **and** relative tolerance), median time/cycles, operation count, FLOPs/cycle, % of relevant theoretical peak, baseline timing, compiler options, thread count, and conversion/copy costs when relevant. The toy benchmark provided here is **only a smoke test**, not a scientifically controlled result.

We propose **NCHW8c** channel blocking and a register-tiled AVX2/FMA microkernel. The starter reference operates in plain **NCHW** and **OIHW**. Converting to NCHW8c, designing packed weights, and measuring conversion cost are *future implementation tasks*.

## Schedule: milestone execution

| Date | Internal target |
| --- | --- |
| Sun Oct 18 | Create GitHub repo, assign owners, compile and test starter code; confirm Xeon access |
| Mon Oct 19 | Select real YOLOv8n layer shapes; extract reference tensors; measure OpenCV DNN baseline; initial kernel designs |
| Tue–Wed Oct 20–21 | Correct optimized kernels for assigned operations; tests and error measurements |
| Thu Oct 22 | Measure and explain per-kernel performance; decide/register-map and memory-layout rationale |
| Fri Oct 23 | Finish report draft, performance plots, presentation slides and first run-through |
| **Sat Oct 24** | **Internal freeze: all content ready** |
| **Sun Oct 25 AoE** | Design report deadline per milestone instructions |
| **Wed Oct 28** | Anticipated 20-minute presentation — **verify date/order on Canvas**, since older overview slides mention Oct 19–21 |

**Milestone outputs:** 20-minute team presentation with each member explaining their kernel and a design report (check current Canvas instructions for the exact report page count and each member's submission responsibility). Focus on kernel implementation, hardware peak and utilization, correctness, performance, and why/when intermediate layouts are required. A later phase adds deeper parallelization, memory movement, and cross-kernel optimizations.

### Suggested ~20 minute presentation

- 0:00–0:30: one-slide project intro and responsibilities
- 0:30–2:00: machine model, per-core peak, SIMD and FMA constraints
- 2:00–7:00: Om's stride-1 convolution, register tile and results
- 7:00–12:00: Sherlock's stride-2 convolution, design and results
- 12:00–16:00: Simon's SPPF/fused work and measured results
- 16:00–19:00: cross-kernel comparisons and memory layout decisions
- 19:00–20:00: current limitations and next steps

## Directory map

```text
include/       Public reference and placeholder kernel interfaces
src/           Reference operations and per-member kernel files
benchmarks/    Small repeatable smoke-test benchmark (not final methodology)
tests/         Automated numerical correctness/unit tests
scripts/       Export pretrained YOLOv8n to ONNX
data/          Local models, tensors and images (ignored by Git)
results/       Tables/plots and documented configs (not raw giant dumps)
docs/milestone/ Presentation and report work
.github/       CI to compile and run starter tests
```

## Developer commands

On Broadwell **only** (or another compiler/CPU supporting the target flags) you may additionally pass `-DYOLO_ENABLE_BROADWELL=ON`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Do **not** enable Broadwell targeting on Apple Silicon. Do not use `-ffast-math` without accounting for changed floating-point behavior. Do not compare different compiler flags as though only the algorithm changed.

## Next pieces that are NOT implemented

- A script to obtain **intermediate per-layer activations, weights and reference outputs** from a selected ONNX model (team must write this; ONNX export alone is insufficient).
- A true **AVX2 direct convolution** and NCHW8c conversion/packed-weight code.
- OpenCV DNN / ONNX Runtime per-layer baseline measurement infrastructure.
- Robust benchmark harness (pin cores, warm caches consistently, synchronize threads, repeated trials, optional hardware performance counters).
- 1×1 kernel if approved as meaningful project scope.

## Workflow rules

- Each owner adds a correctness test before promoting a new optimization.
- File I/O and ONNX export are **not** inside timed kernel calls.
- Include reproducibility notes with every plot: machine, layer shape, compiler, options, thread count, median and runs.
- Use pull requests and review shared interface changes. Don't commit pretrained model binaries or COCO data.

## References

- Project proposal: *Fast Direct Convolution for YOLOv8 Inference on a Broadwell Xeon* (team proposal, October 2026)
- CMU 18-645 project overview and milestone instructions (Canvas)
- Ultralytics: https://github.com/ultralytics/ultralytics
- YOLOv8 documentation: https://docs.ultralytics.com/models/yolov8/
- OpenCV DNN: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
