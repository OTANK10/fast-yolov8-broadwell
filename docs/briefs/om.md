# Brief: Om, 3x3 stride-1 convolution

Paste this file and `CLAUDE.md` into Claude at the start of a session, or open Claude Code in the repo on your branch and say "read docs/briefs/om.md".

## Your layers (32 layers, 5.80 GFLOP, 66% of the model)

20 layers inside the C2f bottlenecks (C = K = 16, 32, 64, 128 at 160², 80², 40², 20²) and 12 in the Detect head (64 or 80 output channels, some with 128 or 256 inputs). All pad 1, all followed by SiLU. Full list: `docs/LAYERS.md`, kind `3x3s1`.

## Your file and branch

`src/conv3x3_s1.cpp` on `om/conv3x3-s1`. The versions are a staircase, one bottleneck per step: `s1_v0` scalar, `s1_v1` AVX2 with one accumulator (latency-bound), `s1_v2` the 2×5 register tile with 10 accumulators on one core, `s1_v3` = v2 + OpenMP. `s1_t<KB>x<OW>` are the same microkernel with other tile shapes for the accumulator experiment (`scripts/tile_sweep.sh`). All pass on all 32 real layers. The design and every number behind it are in `docs/STRIDE1_GUIDE.md`, organized by the Project 2 handout questions.

## Your next tasks

1. On ece028: `bash scripts/machine_info.sh`, `scripts/tile_sweep.sh`, then `scripts/run_kernels.sh` for `s1_v1`, `s1_v2` and `s1_v3`, and `python scripts/baseline_bench.py --batch 1 --threads 1`. These are your first real numbers.
2. Compare single-core % of peak across C = 16, 32, 64, 128. If it drops from C = 64, the weight slab has left L1 (guide section 5); write `s1_v4` that blocks the input-channel loop.
3. Thread scaling of `s1_v3` from 1 thread up to all cores, at batch 1 and 16, and fix first-touch NUMA placement for 20.
4. Stretch: fuse the Bottleneck residual add into the `m.*.cv2` epilogue.

## Prompt to start a Claude session with

> I'm Om, working on the 3x3 stride-1 conv kernel in src/conv3x3_s1.cpp for our 18-645 YOLOv8n project. Read CLAUDE.md, docs/briefs/om.md and docs/STRIDE1_GUIDE.md. Here are my ece028 results: [paste CSV]. Help me explain where the kernel loses performance versus peak and plan s1_v4.
