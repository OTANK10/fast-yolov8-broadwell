# Team guide: start here

This is the one page to read before you touch anything. It says what we are building, what your part is, how to get set up, and how to use Claude with this repo.

## The project in five lines

We take the pretrained YOLOv8n detector, export it to ONNX, and rewrite its convolution layers as fast AVX2 C++ kernels for one machine, ece028 (an Intel Xeon; AVX2 peak 32 FP32 FLOPs/cycle/core). Every kernel is checked against the real outputs of the baseline model on real images, then timed and reported as FLOPs/cycle and % of peak, next to OpenCV DNN and ONNX Runtime on the same layers. The whole structure (data layout, test harness, data dumps, baselines, plotting) is already built and shared. Each of us only writes the inside of our own kernel.

## Who does what

| Person | Kernel | Layers | Share of model | Brief |
|---|---|---|---|---|
| Om | 3×3 stride 1 | 32 | 66% | [docs/briefs/om.md](briefs/om.md) |
| O'Malley | 3×3 stride 2 | 7 | 15% | [docs/briefs/omalley.md](briefs/omalley.md) |
| Simon | 1×1 convs, SPPF max-pool, SiLU activation | 24 + SPPF | 19% | [docs/briefs/simon.md](briefs/simon.md) |

Why this split: the three conv shapes in YOLOv8n are 3×3 stride 1, 3×3 stride 2 and 1×1, and each is a different kernel with its own story for the presentation. Stride 1 is two thirds of the work, so Om takes it. Stride 2 has the fewest layers but the hardest special case (the 3-channel stem layer on the full 640×640 image). Simon's three pieces are all small and share one theme (the work around the convolutions), and the 1×1 conv is what Prof. Low asked for to get a full model. Batch-norm folding is not a task: the ONNX export has already folded it into the weights, so it costs nothing at inference, which was Prof. Low's other point.

## Why you never wait on anyone

All kernels read and write the same data format, defined once in `include/common.h`. Every kernel file already contains a slow but correct version, so the tests and benchmarks pass for everyone today. You replace your own slow version with a fast one, on your own branch, testing on real dumped layers. Nobody hands anything off. The only shared change to coordinate is an edit to `include/common.h` or the scripts, which goes in its own small PR.

## Setup (once, about 15 minutes)

1. Git and the repo on ece028: follow "First-time git setup" in [docs/WORKFLOW.md](WORKFLOW.md). Your branch: `sherlock/conv3x3-s2` or `simon/sppf`.
2. `bash scripts/setup.sh` from the repo root. It creates a Python environment on `/scratch`, downloads YOLOv8n, exports it, dumps every layer's real data, builds with AVX2 and runs the tests. It ends by timing one layer.
3. Every new shell: `source /scratch/$USER/645/venv/bin/activate`.

You can also clone on a laptop to read code and run `ctest` (build with `-DYOLO_ENABLE_BROADWELL=OFF` on a Mac), but only ece028 numbers go in the report.

## Using Claude with this repo

**Claude Code (recommended).** Run `claude` from the repo root on ece028 or your laptop. It reads `CLAUDE.md` automatically, so it already knows the project, the rules, who owns which file and how to test. Three shared commands are included:

| Command | What it does |
|---|---|
| `/check s2_v1 3x3s2` | builds, runs the unit tests, runs your kernel on all your real layers, summarizes FLOPs/cycle and % of peak, compares with the baselines |
| `/new-version conv3x3_s2_v2` | copies your latest version into a new one and registers it in the header, harness and tests |
| `/explain conv3x3_s2_v2` | explains your kernel along the presentation outline, with likely professor questions |

**Claude chat.** Start the conversation by pasting `CLAUDE.md` and your brief, then use the starter prompt at the bottom of your brief. Paste code and results into the chat as you go.

Either way, ask Claude to explain each change until you could present it yourself. The presentation is yours, not Claude's.

## The daily loop

```sh
git pull --rebase origin main          # pick up shared changes
# edit your kernel file
/check <kernel> <kind>                 # inside Claude Code; or in a shell: cmake --build build -j && ctest --test-dir build && scripts/run_kernels.sh <kernel> <kind>
git add -p && git commit -m "s2: ..." && git push
```

Open a PR into `main` once a version passes on all your layers. CI builds and tests every push.

## What each of us brings to the milestone

For your kernel: correctness on every real layer, FLOPs/cycle and % of peak per layer next to OpenCV DNN and ONNX Runtime, and the design explanation (independent operations, dependency chain, functional units, peak, register tile, register reuse, memory layout). `docs/STRIDE1_GUIDE.md` is a finished example of that write-up to copy the structure of. Internal freeze Sat Oct 24, report due Sun Oct 25 AoE, presentation date to confirm on Canvas.
