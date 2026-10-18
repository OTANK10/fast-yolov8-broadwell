---
description: Start a new version of my kernel (copy the latest, register it everywhere, verify it builds)
argument-hint: <new kernel function name>   e.g. conv3x3_s2_v2
---
Create kernel version "$ARGUMENTS" following CLAUDE.md:

1. Work out which file it belongs in from the name (conv3x3_s1 -> src/conv3x3_s1.cpp, conv3x3_s2 -> src/conv3x3_s2.cpp, conv1x1 -> src/conv1x1.cpp) and check I am on that owner's branch. Do not touch another owner's file.
2. Copy the newest existing version in that file into a new function with this name and the standard signature. Leave the old version unchanged.
3. Declare it in `include/kernels.h`, add it to the KERNELS table in `benchmarks/bench_layer.cpp` with its short name (e.g. `s2_v2`), and add `check_kernel` lines for it in `tests/test_kernels.cpp` covering the same shapes as the previous version.
4. Build and run `ctest`. Then stop and ask me what optimization this version should try, explaining the options in terms of the bottleneck each one removes.
