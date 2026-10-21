# Brief: Simon, 1x1 convolutions, SPPF max-pool, SiLU epilogue

Paste this file and `CLAUDE.md` into Claude at the start of a session, or just open Claude Code in the repo on your branch (it reads `CLAUDE.md` itself) and say "read docs/briefs/simon.md".

## Your work (24 1x1 layers = 1.67 GFLOP, 19% of the model, plus SPPF and the activation)

1. **1x1 convolutions** (`src/conv1x1.cpp`). Prof. Low's feedback asked for the 1x1 to get a full model, and they are the second-largest share of the FLOPs. Every C2f block starts (`cv1`) and ends (`cv2`) with one, SPPF has two, and each Detect branch ends with one. Shapes run from 32→32 at 160×160 to 512→256 at 20×20; the full list is in `docs/LAYERS.md` (kind `1x1`).
2. **SPPF max-pool** (`src/sppf.cpp`, layer `model.9`). Three chained 5×5, stride-1, pad-2 max-pools on a 128-channel 20×20 tensor, whose outputs are concatenated with the input (512 channels) and fed to the `model.9.cv2` 1x1 conv.
3. **The SiLU epilogue** (`include/epilogue.h`). Every kernel calls `epilogue_ps()` on its accumulators before storing. A correct AVX2 version already exists, so nobody is waiting on you; making it faster speeds up all three kernels.

Branch: `simon/sppf`.

## The 1x1 conv is a GEMM

In NCHW8c a 1x1 conv is `out[pixel][k] = bias[k] + Σ_c in[pixel][c] · W[k][c]`, a matrix multiply of [pixels × C] by [C × K], which is the Assignment 2 matmul kernel again. Om's stride-1 microkernel (`src/conv3x3_s1.cpp`, `docs/STRIDE1_GUIDE.md`) already works for it if you set R = S = 1: a 2×5 tile of 2 output-channel blocks × 5 pixels gives the 10 independent accumulators needed to hide the 5-cycle FMA latency. Every width (160, 80, 40, 20) divides by 5 and every K/8 (4, 8, 10, 16, 32) is even. Two details: the input has a 1-pixel halo even though a 1x1 conv has pad 0, so start reading at (row + 1, col + 1); and for C = 384 or 512 the weight slab for 2 output blocks is 24 to 32 KB, right at the 32 KB L1 limit, which is a good thing to measure and discuss.

## SPPF details that matter

- **The halo is zeros, but max-pool padding must be -infinity.** The SPPF input comes after SiLU, which can be negative (the real `model.9` input ranges from about -0.28 to 5.9, and 9% of values are negative). A 5×5 window at the border that only sees negative values plus a zero halo would wrongly return 0. Either clamp the window to the image or write -inf into a 2-pixel halo for the pool buffers.
- A 5×5 max is separable: a 1×5 max along each row, then a 5×1 max along each column, which cuts 25 comparisons to 10 per output. With 8 channels per register, each step is one `_mm256_max_ps`.
- If each pool writes straight into its 128-channel slice of the 512-channel NCHW8c buffer that `model.9.cv2` reads, the Concat costs nothing (no copy). That is exactly the "layout of surrounding layers" point from Prof. Low's feedback.
- `scripts/dump_layers.py` already dumps each MaxPool's input and output to `data/tensors/b1/model.9.m*.MaxPool/` (`meta.txt` is `N C H W k stride pad OH OW`). The harness does not time pools yet; adding a pooling mode to `benchmarks/bench_layer.cpp` (or a small `bench_pool.cpp`) is part of your work. The FLOP count does not apply to pooling, so report comparisons per cycle and GB/s instead.

## Faster SiLU

`silu256_ps` computes `x / (1 + exp(-x))` with one `vdivps`, which on Broadwell blocks the divider for about 10 cycles. Replacing it with `_mm256_rcp_ps` plus one Newton-Raphson step (`r = r * (2 - d*r)`) keeps the error well under our 1e-4 tolerance and is a clean, measurable change. Check that `ctest` still passes for everyone's kernels afterwards, since they all call it.

## Your first three tasks

1. Set up git and ece028 with `docs/WORKFLOW.md`, then run `scripts/run_kernels.sh 1x1_v1 1x1` (currently the scalar fallback) and `python scripts/baseline_bench.py`.
2. Write `conv1x1_v2` with the 2×5 tile (register it in `include/kernels.h`, `benchmarks/bench_layer.cpp`, `tests/test_kernels.cpp`). Pass `ctest`, then `scripts/run_kernels.sh 1x1_v2 1x1`.
3. Write the SPPF pool kernel and its benchmark, then the faster SiLU.

## Prompt to start a Claude session with

> I'm Simon, working on the 1x1 conv kernel (src/conv1x1.cpp), the SPPF max-pool (src/sppf.cpp) and the SiLU epilogue (include/epilogue.h) for our 18-645 YOLOv8n project. Read CLAUDE.md and docs/briefs/simon.md, then src/conv3x3_s1.cpp as the template. Help me write conv1x1_v2 with the 2x5 register tile, explain each line so I can present it, and run ctest and run_kernels.sh 1x1_v2 1x1 before telling me it works.
