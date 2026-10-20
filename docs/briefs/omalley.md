# Brief: O'Malley, 3x3 stride-2 convolution

Paste this file and `CLAUDE.md` into Claude at the start of a session, or just open Claude Code in the repo on your branch (it reads `CLAUDE.md` itself) and say "read docs/briefs/omalley.md".

## Your layers (7 layers, 1.27 GFLOP, 15% of the model)

| Layer | C → K | Input H×W | Output | Note |
|---|---|---|---|---|
| `model.0` | 3 → 16 | 640×640 | 320×320 | the stem: only 3 input channels, padded to one 8-channel block |
| `model.1` | 16 → 32 | 320×320 | 160×160 | |
| `model.3` | 32 → 64 | 160×160 | 80×80 | |
| `model.5` | 64 → 128 | 80×80 | 40×40 | |
| `model.7` | 128 → 256 | 40×40 | 20×20 | |
| `model.16` | 64 → 64 | 80×80 | 40×40 | neck (PAN) |
| `model.19` | 128 → 128 | 40×40 | 20×20 | neck (PAN) |

All have pad 1 and are followed by SiLU. The real tensors are in `data/tensors/b1/<layer>/` after setup.

## Your file and branch

`src/conv3x3_s2.cpp` on branch `sherlock/conv3x3-s2`. Today `conv3x3_s2_v1` forwards to the generic scalar conv, so it is already correct and the harness already times it (about 0.6 FLOPs/cycle, roughly 2% of peak). Your job is to replace that.

## Fastest way to a good v1

Om's stride-1 kernel (`src/conv3x3_s1.cpp`, explained in `docs/STRIDE1_GUIDE.md`) uses a 2×5 register tile: 2 output-channel blocks × 5 output pixels = 10 YMM accumulators, enough to hide the 5-cycle FMA latency on 2 FMA ports. The same tile works for you, with one change: output pixel `ow+j` reads input column `2*(ow+j) + s`, so the broadcast address steps by `2 * CB` floats per output pixel instead of `CB`, and the input row is `2*oh + r`. Every one of your output widths (320, 160, 80, 40, 20) divides by 5 and every K/8 is even, so no edge code is needed. Start by copying `ukr_2x5` and changing those two strides, then make the version your own.

## Things specific to your layers (good presentation material)

- **The stem (`model.0`) wastes 5/8 of every FMA.** With C = 3, the 8-channel block is mostly zeros, yet it is 7% of your FLOPs and its 640×640 input is the largest tensor in the network. A specialized stem kernel (for example looping over the 27 real (c, r, s) taps directly, with no channel block) is a clean, explainable win.
- **Stride 2 reads each input pixel fewer times** than stride 1, so your layers have lower arithmetic intensity, and `model.0`/`model.1` move a lot of data. Check whether they are compute- or memory-bound by comparing % of peak at batch 1 vs the input size.
- Only the even input columns land on the "center" tap, so consider whether splitting the input into even and odd columns makes the loads contiguous. Measure before you commit to it.

## Your first three tasks

1. Set up git and ece028 with `docs/WORKFLOW.md`, then run `scripts/run_kernels.sh s2_v1 3x3s2` to see the baseline of your current (scalar) version, and `python scripts/baseline_bench.py` for OpenCV DNN and ONNX Runtime.
2. Write `conv3x3_s2_v2` with the 2×5 tile (register it in `include/kernels.h`, `benchmarks/bench_layer.cpp`, `tests/test_kernels.cpp`). Pass `ctest`, then `scripts/run_kernels.sh s2_v2 3x3s2`.
3. Fill in the analysis for your kernel, following the section headings of `docs/STRIDE1_GUIDE.md` (independent ops, dependency chain, functional units, peak, tile, register allocation and reuse, layout, measured % of peak and why it differs).

## Prompt to start a Claude session with

> I'm O'Malley, working on the 3x3 stride-2 conv kernel in src/conv3x3_s2.cpp for our 18-645 YOLOv8n project. Read CLAUDE.md and docs/briefs/omalley.md, then src/conv3x3_s1.cpp as the template. Help me write conv3x3_s2_v2 with the 2x5 register tile, explain each line so I can present it, and run ctest and run_kernels.sh s2_v2 3x3s2 before telling me it works.
