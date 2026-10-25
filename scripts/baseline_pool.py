#!/usr/bin/env python3
"""OpenCV DNN time of the SPPF max-pools (model.9), to compare with bench_pool.

    python scripts/baseline_pool.py --batch 1 --threads 1

Same method as baseline_bench.py: net.getPerfProfile() after each forward, median
over --reps runs after 3 warm-ups. Prints every OpenCV layer whose name contains
"MaxPool" or "pool" (OpenCV may rename layers, so check the names it prints), their
sum, and writes results/baseline_pool_b<N>_t<T>.csv. Times are microseconds; cycles
use the Assignment 2 convention (x 3.4 GHz), like bench_pool.
"""
import argparse, csv, os, statistics, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from common_py import list_images, make_batch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--onnx", default=None)
ap.add_argument("--images", default=None)
a = ap.parse_args()
onnx_path = a.onnx or f"data/models/yolov8n_b{a.batch}.onnx"
x = make_batch(list_images(a.images), a.batch)

cv2.setNumThreads(a.threads)
net = cv2.dnn.readNetFromONNX(onnx_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
times = {}
for i in range(a.reps + 3):
    net.setInput(x)
    net.forward()
    if i < 3:
        continue
    _, per_layer = net.getPerfProfile()
    for name, t in zip(net.getLayerNames(), np.asarray(per_layer).ravel()):
        if "pool" in name.lower():
            times.setdefault(name, []).append(float(t) / cv2.getTickFrequency() * 1e6)

if not times:
    sys.exit("no layer with 'pool' in its name; print net.getLayerNames() to find the MaxPool layers")
os.makedirs("results", exist_ok=True)
out = f"results/baseline_pool_b{a.batch}_t{a.threads}.csv"
total = 0.0
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["opencv_layer", "ocv_us", "cycles_at_3.4GHz"])
    for name, v in times.items():
        us = statistics.median(v)
        total += us
        w.writerow([name, f"{us:.2f}", f"{us * 3.4e3:.0f}"])
        print(f"  {name:40s} {us:8.2f} us")
    w.writerow(["total", f"{total:.2f}", f"{total * 3.4e3:.0f}"])
print(f"OpenCV DNN, {len(times)} pool layers, batch {a.batch}, {a.threads} thread(s): {total:.2f} us total")
print("wrote", out)
