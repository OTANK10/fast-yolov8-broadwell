---
description: Build, run the unit tests, and run one kernel on all of its real yolov8n layers
argument-hint: <kernel> <kind> [batch] [threads]   e.g. s2_v2 3x3s2
---
Check the kernel given in "$ARGUMENTS" (kernel name, then kind: 3x3s1, 3x3s2 or 1x1; optional batch and thread count).

1. If `build/` does not exist, configure it: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON` (use OFF if this machine is not x86, and say so).
2. `cmake --build build -j` and `ctest --test-dir build --output-on-failure`. Stop and explain any failure.
3. `scripts/run_kernels.sh $ARGUMENTS`.
4. Report: PASS/FAIL per layer, then a short table of FLOPs/cycle and % of peak per layer, the best and worst layer, and whether the cycle source was `core` or `tsc`. If any layer is above 100% of peak, say the cycle count is inflated by turbo, not that the kernel beats peak.
5. If `results/baseline_b<batch>_t<threads>.csv` exists, compare against OpenCV DNN and ONNX Runtime per layer. Do not invent baseline numbers if it does not exist; tell me to run `python scripts/baseline_bench.py`.
