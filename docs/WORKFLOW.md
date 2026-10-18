# Team workflow

## Why nobody blocks on anybody

Each kernel reads and writes the same blocked tensors defined in `include/common.h`, and every kernel file already contains a correct (slow) version that forwards to `conv_blk_generic`. So O'Malley can tune stride 2, Simon can tune 1×1, SPPF and SiLU, and Om can tune stride 1, all at once, each testing on real dumped yolov8n layers. The only time you need each other is when you change the contract (layout, halo, signature). Do that in a tiny PR, tag both teammates, and merge it before building on it.

The one real dependency is the epilogue: every kernel calls `epilogue_ps()` from `include/epilogue.h`. A correct AVX2 SiLU is already there, so nobody waits on Simon; when he makes it faster, everyone gets it on their next `git pull`.

## Branches

```
main                  always builds, ctest passes, every real layer PASSes
om/conv3x3-s1         Om
sherlock/conv3x3-s2   O'Malley
simon/sppf            Simon (SPPF, SiLU epilogue, 1x1 if we take it on)
```

Work on your branch, push often (ece007 scratch is not backed up), open a PR into `main` when a kernel version is correct on all your layers. GitHub Actions builds and runs `ctest` on every push and PR, with and without AVX2. Keep shared-file changes out of kernel PRs.

## First-time git setup on ece007

```bash
ssh-keygen -t ed25519 -C "andrewid@andrew.cmu.edu"   # accept defaults
cat ~/.ssh/id_ed25519.pub                             # paste into GitHub > Settings > SSH keys
ssh -T git@github.com                                 # should greet you by username
git config --global user.name "Your Name"
git config --global user.email "andrewid@andrew.cmu.edu"

cd /scratch/$USER/645
git clone git@github.com:OTANK10/fast-yolov8-broadwell.git
cd fast-yolov8-broadwell
git switch om/conv3x3-s1        # or: git switch -c sherlock/conv3x3-s2 / simon/sppf
bash scripts/setup_ece007.sh
```

If `~/.ssh` lives on AFS and you get permission errors, run `kinit && aklog` first.

## Daily loop

```bash
source /scratch/$USER/645/venv/bin/activate
git pull --rebase origin main               # pick up teammates' shared changes
# edit src/conv3x3_s1.cpp
cmake --build build -j && ctest --test-dir build                       # random-data checks, 1 s
./build/bench_layer data/tensors/b1/model.4.m.0.cv1 s1_v1              # one real layer, check + time
scripts/run_kernels.sh s1_v1 3x3s1                                     # all real layers before committing
git add -p && git commit -m "s1: ..." && git push -u origin HEAD
```

`cmake --build build --target asm_s1` writes `build/conv3x3_s1.s`. Search it for `vfmadd231ps` to see your inner loop, and for `(%rsp)` inside the FMA loop to catch register spills (v1 has none there).

## Adding a kernel version

1. Write `void conv3x3_s1_v2(...)` in your file with the standard signature.
2. Declare it in `include/kernels.h`.
3. Add `{"s1_v2", conv3x3_s1_v2},` to the table at the top of `benchmarks/bench_layer.cpp`.
4. Add a `check_kernel("s1_v2", ...)` line to `tests/test_kernels.cpp`.
5. `cmake --build build -j && ctest --test-dir build && scripts/run_kernels.sh s1_v2 3x3s1`.

Keep old versions; the presentation is the story from v0 to vN.

## Measurement rules (so all three of us report comparable numbers)

Build with `-DYOLO_ENABLE_BROADWELL=ON` (the setup script does). Use `bench_layer` for kernels and `baseline_bench.py` for baselines, same batch and thread count. Single-thread numbers use core cycles when the kernel lets user code read the perf counter, otherwise TSC cycles; the output says which (`core-cycles` or `tsc-cycles`). TSC ticks at 2.4 GHz but the cores turbo higher, so with TSC a single core can appear to beat its 32 FLOPs/cycle peak. If you see over 100% of peak, the cycle count is the problem, not the kernel. For multi-thread runs we report TSC cycles (wall time), which is what matters for parallel speed. Run on an otherwise idle ece007 (check `top`), pin threads (`run_kernels.sh` sets `OMP_PROC_BIND=close OMP_PLACES=cores`), and use the median over 20 reps when it differs much from the best.

## What goes in git

Code, scripts, docs, and the small result CSVs and PNGs in `results/`. Never the model files, `data/models`, `data/tensors` or `results/raw` (all gitignored; regenerate with the scripts).
