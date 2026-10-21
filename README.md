# Fast YOLOv8n on Intel Broadwell (ece028)

**CMU 18-645: How to Write Fast Code — Fall 2026**  
**Status:** Shared harness, layouts, per-layer data dumps and baselines are in. Stride-1 is measured on ece028: `s1_v2` reaches 77% of peak on one core (24.7 FLOPs/cycle over all 32 layers) and `s1_v3` runs 13.5× faster on 20 cores; see `docs/milestone/` (design_report.pdf) and `docs/milestone/om/`. Stride-2, 1×1 and SPPF are correct scalar fallbacks waiting for their owners.

## Start here (5 minutes)

**Teammates: read [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md) first**, then your brief in [docs/briefs/](docs/briefs/). Claude Code picks up the project context from [CLAUDE.md](CLAUDE.md) automatically.

We are **not training YOLO**. We are rebuilding selected YOLOv8n inference operations as fast C++ kernels and measuring how efficiently they use the Intel Xeon E5-2640 v4 (Broadwell) CPU of ece028. (We moved from ece007 to ece028 on Oct 20; both have the same CPU.) Our source model is the **YOLOv8n** variant, not whatever newest YOLO model Ultralytics advertises.

1. **One-time setup on ece028** (venv, model export, real-layer dumps, Broadwell build, tests):

   ```sh
   cd /scratch/$USER/645 && git clone git@github.com:OTANK10/fast-yolov8-broadwell.git
   cd fast-yolov8-broadwell && bash scripts/setup.sh
   source /scratch/$USER/645/venv/bin/activate      # every new shell
   ```

   This exports `data/models/yolov8n_b1.onnx` and dumps every conv layer's real input, weights, bias and baseline outputs to `data/tensors/b1/<layer>/` (both are symlinks to scratch, never committed). The layer list with owners and FLOPs is in [docs/LAYERS.md](docs/LAYERS.md).

2. **The daily loop**:

   ```sh
   cmake --build build -j && ctest --test-dir build --output-on-failure    # random-data checks, any machine
   ./build/bench_layer data/tensors/b1/model.4.m.0.cv1 s1_v1               # one real layer: check vs baseline, then time
   scripts/run_kernels.sh s1_v1 3x3s1                                      # all your layers -> results/*.csv
   python scripts/baseline_bench.py --batch 1 --threads 1                  # OpenCV DNN + ONNX Runtime per layer
   python scripts/plot_results.py results/s1_v1_b1_t1.csv results/baseline_b1_t1.csv
   ```

   Kernel names: `s1_v0`..`s1_v3` and `s1_t<KB>x<OW>` (Om), `s2_v1` (O'Malley), `1x1_v1`, plus `generic` and `ref`. Kinds for `run_kernels.sh`: `3x3s1`, `3x3s2`, `1x1`.

3. **Branches:** `om/conv3x3-s1`, `sherlock/conv3x3-s2`, `simon/sppf`. Open pull requests for shared changes. Git setup and rules: [docs/WORKFLOW.md](docs/WORKFLOW.md).

4. **Correctness anywhere, performance on ece028.** Without `-DYOLO_ENABLE_BROADWELL=ON` (e.g. on a Mac) the AVX2 kernels fall back to scalar code so everything still builds and tests pass, but only ece028 numbers go in the report.

### What Om is doing

Om owns **3×3, stride-1 FP32 convolution**: the 20 C2f bottleneck layers plus the 12 stride-1 layers in the Detect head, which the same kernel handles (32 layers, 66% of the model's FLOPs). Design notes: [docs/STRIDE1_GUIDE.md](docs/STRIDE1_GUIDE.md).

**Current state:** a staircase of versions, each removing one bottleneck: `s1_v0` scalar, `s1_v1` AVX2 with 1 accumulator, `s1_v2` 2×5 register tile (10 accumulators, 1 core), `s1_v3` v2 + OpenMP, plus a tile-shape sweep (`scripts/tile_sweep.sh`). All pass on all 32 real layers. Measured on ece028 (GCC 8.5.0): v1 3.2, v2 24.7 FLOPs/cycle (77% of peak), v4/v5 (2×6 tile, packing) no faster; report and slides in `docs/milestone/` (design_report.pdf) and `docs/milestone/om/`.

## Who owns what

| Owner | Kernel | Milestone evidence |
| --- | --- | --- |
| **Om Tank** | 3×3 stride-1 convolution (C2f bottleneck) | Correctness, SIMD/FMA register map, performance vs. reference and peak |
| **O'Malley Sherlock** | 3×3 stride-2 convolution (downsampling) | Correctness, addressing and tiling explanation, per-layer timings |
| **Bohan (Simon) Yu** | 1×1 convolutions, SPPF max-pool, SiLU epilogue | Correctness, GEMM tile and L1 analysis for 1×1, pooling benchmark and memory-access strategy |

Per Professor Low's feedback, folding batch norm into weights **does not itself add inference-time compute**, so that must not be presented as a standalone performance kernel. The included SPPF reference covers the pooling cascade only, not its surrounding convolution/activation layers.

## What are we measuring?

Model: pretrained **Ultralytics YOLOv8n**, exported to ONNX at image size 640.  
Primary baseline: **OpenCV DNN** (as specified in the proposal).  
Secondary reference: **ONNX Runtime CPU**.  
Target platform: **ece028, Intel Xeon E5-2640 v4** (Broadwell-EP, 2.40 GHz base, 3.4 GHz turbo), 2 sockets × 10 physical cores (40 threads with Hyper-Threading; use at most 20), 2 NUMA nodes, AVX2/FMA, no AVX-512, 16 YMM registers, 32 KB L1D/core, 256 KB L2/core, 25 MB L3/socket.

For 32-bit FMA with AVX2, the **per-core compute peak is 32 FP32 FLOPs/cycle** (two 256-bit FMA instructions per cycle × eight lanes × two FLOPs per fused multiply-add). Actual sustained rate may be lower due to dependencies, loads, addresses, issue limits, and cache effects. The 5-cycle FMA latency suggests ~10 independent accumulator chains are required to sustain peak FMA throughput, subject to other resource constraints.

For each real layer shape, report: tensor shapes, validation error (absolute **and** relative tolerance), median time/cycles, operation count, FLOPs/cycle, % of relevant theoretical peak, baseline timing, compiler options, thread count, and conversion/copy costs when relevant. The toy benchmark provided here is **only a smoke test**, not a scientifically controlled result.

**Shared layout contract** (`include/common.h`): activations are **NCHW8c** with a 1-pixel zero halo on every plane (no edge branches in 3×3 kernels, and any layer's output feeds the next directly); weights are packed offline to `[K/8][C/8][R][S][8 ic][8 oc]`; every kernel has the signature `f(const conv_desc_t*, const float *in, const float *w, const float *bias, float *out)`. The ONNX export already folds BatchNorm into each Conv's weights and bias (the graph has no BatchNormalization nodes), and SiLU is applied in registers by `include/epilogue.h`. Because everyone codes against this one header and the harness feeds every kernel the same real tensors, the three kernels are developed in parallel with no hand-offs.

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
include/       common.h (layouts, conv_desc_t, signature, timing), epilogue.h (AVX2 SiLU), kernels.h
src/           layout.cpp, reference_conv.cpp (never optimize), one file per owner's kernel
benchmarks/    bench_layer.cpp: real-layer correctness (max rel err <= 1e-4) + cycles, FLOPs/cycle, % peak
tests/         Random-data correctness of every kernel on yolov8n channel counts (runs in CI)
scripts/       export_model, dump_layers, baseline_bench, run_kernels, plot_results, setup_ece028, fetch_coco_val
data/          models/ and tensors/ (symlinks to scratch, ignored by Git)
results/       Tables/plots and documented configs (not raw giant dumps)
docs/milestone/ Presentation and report work
docs/          WORKFLOW.md, STRIDE1_GUIDE.md, LAYERS.md
.github/       CI: build + ctest with and without AVX2
```

## Developer commands

On ece028 (or any x86 CPU with AVX2/FMA) pass `-DYOLO_ENABLE_BROADWELL=ON`; this is what turns on the fast kernels. The name is historical: `-march=broadwell` emits AVX2 + FMA code, which runs on every newer Intel Xeon:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --build build --target asm_s1      # generated assembly of the stride-1 kernel -> build/conv3x3_s1.s
```

Do **not** enable Broadwell targeting on Apple Silicon. Do not use `-ffast-math` without accounting for changed floating-point behavior. Do not compare different compiler flags as though only the algorithm changed.

## Next pieces that are NOT implemented

- Optimized stride-2 (`src/conv3x3_s2.cpp`), 1×1 (`src/conv1x1.cpp`) and SPPF max-pool kernels; each currently forwards to a correct scalar version.
- SPPF entry in `bench_layer` (the MaxPool tensors are already dumped).
- Cache blocking, NUMA-aware first-touch and thread-scaling study across all of ece028's cores.
- Cross-layer fusion: Bottleneck residual add in the epilogue, free C2f Split/Concat in NCHW8c.
- Batch 16 dumps from COCO val (`scripts/fetch_coco_val.sh`, then `dump_layers.py --batch 16 --only 3x3s1`).

## Workflow rules

- Each owner adds a correctness test before promoting a new optimization.
- File I/O and ONNX export are **not** inside timed kernel calls.
- Include reproducibility notes with every plot: machine, layer shape, compiler, options, thread count, median and runs.
- Use pull requests and review shared interface changes. Don't commit pretrained model binaries or COCO data.

## References

- Project proposal: *Fast Direct Convolution for YOLOv8 Inference on a Broadwell Xeon* (team proposal, October 2026)
- CMU 18-645 project overview and milestone instructions (Canvas)
- Ultralytics: https://github.com/ultralytics/ultralytics (YOLOv8 architecture in `ultralytics/cfg/models/v8/yolov8.yaml`, "n" scale = depth 0.33, width 0.25; layers in `ultralytics/nn/modules/conv.py`, `block.py` (C2f, Bottleneck, SPPF), `head.py` (Detect)). Weights `yolov8n.pt` download from https://github.com/ultralytics/assets/releases
- Netron (view the exported ONNX graph): https://netron.app
- YOLOv8 documentation: https://docs.ultralytics.com/models/yolov8/
- OpenCV DNN: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
