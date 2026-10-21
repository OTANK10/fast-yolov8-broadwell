#!/usr/bin/env python3
"""Per-layer timing of the two baselines on the same ONNX model and images.

    python scripts/baseline_bench.py --batch 1 --threads 1
    python scripts/baseline_bench.py --batch 16 --threads 20

  OpenCV DNN (primary):   net.getPerfProfile() after each forward -> per-layer ticks
  ONNX Runtime (cross-check): built-in profiler -> per-node microseconds

Writes results/baseline_b<N>_t<T>.csv with, per conv layer, the median time and
the resulting FLOPs/cycle and % of the T-core peak (32 FLOPs/cycle/core, or $PEAK_FPC,
at the base clock read from /proc/cpuinfo), so it lines up with what bench_layer prints.
"""
import argparse, csv, json, os, statistics, sys, time
import numpy as np
import cv2
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from common_py import NOMINAL_HZ, PEAK_FLOPS_PER_CYCLE_PER_CORE, list_images, make_batch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--onnx", default=None)
ap.add_argument("--images", default=None)
a = ap.parse_args()
onnx_path = a.onnx or f"data/models/yolov8n_b{a.batch}.onnx"
layers = list(csv.DictReader(open(f"data/tensors/b{a.batch}/layers.csv")))
x = make_batch(list_images(a.images), a.batch)

# ---------------- OpenCV DNN ----------------
cv2.setNumThreads(a.threads)
net = cv2.dnn.readNetFromONNX(onnx_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ocv = {}
e2e_ocv = []
for i in range(a.reps + 3):
    net.setInput(x)
    t0 = time.perf_counter(); net.forward(); t1 = time.perf_counter()
    if i < 3:
        continue  # warm-up
    e2e_ocv.append(t1 - t0)
    _, per_layer = net.getPerfProfile()
    for name, t in zip(net.getLayerNames(), np.asarray(per_layer).ravel()):
        ocv.setdefault(name, []).append(float(t) / cv2.getTickFrequency() * 1e6)
ocv_med = {k: statistics.median(v) for k, v in ocv.items()}

# ---------------- ONNX Runtime ----------------
so = ort.SessionOptions()
so.enable_profiling = True
so.intra_op_num_threads = a.threads
so.inter_op_num_threads = 1
so.profile_file_prefix = "results/raw/ort_profile"
os.makedirs("results/raw", exist_ok=True)
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
e2e_ort = []
for i in range(a.reps + 3):
    t0 = time.perf_counter(); sess.run(None, {iname: x}); t1 = time.perf_counter()
    if i >= 3:
        e2e_ort.append(t1 - t0)
prof = json.load(open(sess.end_profiling()))
ort_t = {}
nruns = 0
for ev in prof:
    if ev.get("cat") == "Session" and ev.get("name") == "model_run":
        nruns += 1
    if ev.get("cat") == "Node" and ev["name"].endswith("_kernel_time") and nruns > 3:
        ort_t.setdefault(ev["name"][: -len("_kernel_time")], []).append(ev["dur"])
ort_med = {k: statistics.median(v) for k, v in ort_t.items()}


def lookup(table, onnx_name):
    """Match a conv node to the backend's layer name (exact, then by prefix: ORT may fuse
    Conv+Sigmoid+Mul into one node that keeps the Conv's name, OpenCV may rename)."""
    if onnx_name in table:
        return table[onnx_name]
    hits = [v for k, v in table.items() if k.startswith(onnx_name) or onnx_name.startswith(k + "/")]
    return hits[0] if len(hits) == 1 else float("nan")


peak = PEAK_FLOPS_PER_CYCLE_PER_CORE * a.threads
out = f"results/baseline_b{a.batch}_t{a.threads}.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "kind", "owner", "flops", "ocv_us", "ocv_flops_per_cycle", "ocv_pct_peak",
                "ort_us", "ort_flops_per_cycle", "ort_pct_peak"])
    for L in layers:
        fl = float(L["flops"])
        row = [L["dir"], L["kind"], L["owner"], int(fl)]
        for us in (lookup(ocv_med, L["onnx_name"]), lookup(ort_med, L["onnx_name"])):
            fpc = fl / (us * 1e-6 * NOMINAL_HZ) if us == us and us > 0 else float("nan")
            row += [f"{us:.1f}", f"{fpc:.2f}", f"{100 * fpc / peak:.1f}"]
        w.writerow(row)

print(f"end-to-end median, batch {a.batch}, {a.threads} threads:  "
      f"OpenCV DNN {1e3 * statistics.median(e2e_ocv):.1f} ms   ONNX Runtime {1e3 * statistics.median(e2e_ort):.1f} ms")
print("wrote", out)
