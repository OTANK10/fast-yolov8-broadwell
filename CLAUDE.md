# Project context for Claude

This file is read automatically by Claude Code when it runs in this repo. If you use Claude in a chat instead, paste this file plus your brief from `docs/briefs/` at the start of the conversation.

## What this project is

CMU 18-645 (How to Write Fast Code), Fall 2026, team of three: Om Tank, O'Malley Sherlock, Bohan (Simon) Yu. We write fast FP32 C++ kernels for the convolution layers of the pretrained YOLOv8n object detector and measure how close they get to the peak of one machine: **ece028, Intel Xeon E5-2640 v4 (Broadwell-EP) @ 2.40 GHz (turbo 3.4), 2 sockets x 10 cores (40 hardware threads with Hyper-Threading), 2 NUMA nodes (even CPU ids on node 0, odd on node 1), AVX2 + FMA, no AVX-512, 16 YMM registers, 32 KB L1D, 256 KB L2 per core, 25 MB L3 per socket**. We moved from ece007 to ece028 on Oct 20; it is the same CPU model. Peak is 32 FP32 FLOPs/cycle/core (2 FMA ports x 8 lanes x 2). Use at most 20 threads (one per physical core); `run_kernels.sh` pins with `OMP_PLACES=cores`. We are not training anything. Baselines are OpenCV DNN (primary) and ONNX Runtime, run on the same ONNX model and images.

Grading cares about: correctness against the baseline, FLOPs/cycle and % of peak per real layer, and a clear explanation of the design (independent ops, dependency chains, functional units, peak, register tile, register reuse, memory layout). Each kernel owner presents their own kernel.

## Who owns what (do not edit someone else's kernel file)

| Owner | Files | Layers |
|---|---|---|
| Om | `src/conv3x3_s1.cpp` | 32 layers of 3x3 stride 1 (C2f bottlenecks + Detect head), 66% of FLOPs |
| O'Malley | `src/conv3x3_s2.cpp` | 7 layers of 3x3 stride 2 (downsampling), 15% of FLOPs |
| Simon | `src/conv1x1.cpp`, `src/sppf.cpp`, `include/epilogue.h` | 24 layers of 1x1 (19% of FLOPs), the SPPF max-pool, and the SiLU epilogue every kernel calls |

Shared files, change only in a small separate PR that the whole team reviews: `include/common.h`, `include/kernels.h`, `src/layout.cpp`, `src/reference_conv.cpp`, `benchmarks/`, `tests/`, `scripts/`, `CMakeLists.txt`. Never optimize `src/reference_conv.cpp`; it is the ground truth.

## The contract every kernel follows (`include/common.h`)

- Activations: NCHW8c, stored `[N][C/8][H+2][W+2][8]`. The 8 innermost floats are 8 consecutive channels (one YMM register). A 1-pixel zero border (halo) is stored around every plane, so 3x3 kernels never branch on image edges. A kernel writes only the interior of its output; the halo must stay zero.
- Weights: packed offline to `[K/8][C/8][R][S][8 ic][8 oc]`, so the 8 output-channel weights for one (input channel, r, s) are one aligned load.
- Bias: `[K]` floats, padded to a multiple of 8. BatchNorm is already folded into the weights and bias by the ONNX export. There is no BatchNorm work at inference time.
- Signature: `void f(const conv_desc_t *d, const float *in, const float *w, const float *bias, float *out)`. `d` gives N, C, H, W, K, R, S, stride, pad, OH, OW, halos and `fuse_silu`.
- Activation: if `d->fuse_silu`, apply `epilogue_ps(acc, 1)` from `include/epilogue.h` to each accumulator register before storing it. Bias is added by initializing the accumulators with it.
- Kernels with no AVX2 code yet forward to `conv_blk_generic`, which is correct for any shape. That is why the three of us never block on each other.

## Commands

```sh
# ece028, once:   bash scripts/setup.sh       then every shell:  source /scratch/$USER/645/venv/bin/activate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON   # ON = AVX2 kernels; OFF on a Mac
cmake --build build -j
ctest --test-dir build --output-on-failure                    # random-data check of every kernel vs the reference
./build/bench_layer data/tensors/b1/<layer> <kernel>          # one real layer: correctness vs baseline, then timing
scripts/run_kernels.sh <kernel> <kind> [batch] [threads]      # every layer of a kind -> results/<kernel>_b<N>_t<T>.csv
python scripts/baseline_bench.py --batch 1 --threads 1        # OpenCV DNN + ONNX Runtime per layer -> results/baseline_*.csv
python scripts/plot_results.py results/<kernel>_b1_t1.csv results/baseline_b1_t1.csv
cmake --build build --target asm_s1                           # assembly of the stride-1 kernel
```

Kernel names: `ref`, `generic`, `s1_v0`..`s1_v3`, `s1_t<KB>x<OW>`, `s2_v1`, `1x1_v1`. Machine facts: `bash scripts/machine_info.sh`. Kinds: `3x3s1`, `3x3s2`, `1x1`. Layer list with shapes, owners and FLOPs: `docs/LAYERS.md`.

## Rules for any change Claude makes here

1. Work on the owner's branch (`om/conv3x3-s1`, `sherlock/conv3x3-s2`, `simon/sppf`), never directly on `main`.
2. A new optimization is a new version (`conv3x3_s2_v2`), not an edit of the old one. Register it in `include/kernels.h`, the table in `benchmarks/bench_layer.cpp`, and `tests/test_kernels.cpp`. Old versions stay for the presentation.
3. Before saying a kernel works: `ctest` passes and `scripts/run_kernels.sh <kernel> <kind>` prints "all layers PASS". The tolerance is max relative error 1e-4 against the baseline's real outputs.
4. Only numbers from ece028 built with `-DYOLO_ENABLE_BROADWELL=ON` go in the report. Over 100% of peak means the cycle count is wrong (turbo vs TSC), not that the kernel is that fast; see `docs/WORKFLOW.md`.
5. Timed regions contain only the kernel call. No file I/O, layout conversion or weight packing inside timing.
6. Explain every optimization in terms the owner can present: which bottleneck it removes (FMA latency, load ports, L1/L2 capacity, memory bandwidth, threads), and what the numbers before and after were.
7. Do not commit `data/models`, `data/tensors`, `results/raw`, or any `.onnx`, `.pt`, `.bin` file.

## Where to read more

`docs/WORKFLOW.md` (git setup, daily loop, measurement rules), `docs/STRIDE1_GUIDE.md` (worked design of the stride-1 kernel, the template for the other two), `docs/briefs/` (one page per person: your layers, your first three tasks, what to present).
