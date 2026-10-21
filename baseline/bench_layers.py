#!/usr/bin/env python3
"""Per-layer benchmark of YOLOv8 with OpenCV DNN (CPU).

Uses OpenCV's built-in profiler (net.getPerfProfile) to time every layer,
averaged over N runs. If the `onnx` package is installed, it also reports
output shape, multiply-adds (MACs) and GFLOP/s for each conv layer.

    python bench_layers.py                          # 1 thread, bus640.png, 20 runs
    python bench_layers.py --no-winograd            # direct conv for 3x3 layers
    python bench_layers.py --csv layers_1t.csv      # save every layer, in execution order
    taskset -c 2 python bench_layers.py --top 30    # pinned to one core, show the 30 slowest

Notes:
- OpenCV fuses Conv + SiLU into one layer. The residual Add stays separate.
- 'trlayout.*' layers are OpenCV converting between plain NCHW and its blocked
  internal layout. They are pure overhead; a hand-written pipeline that keeps
  one layout end to end does not need them.
"""
import argparse
import collections
import csv
import re
import sys

import cv2
import numpy as np


def onnx_info(model_path):
    """name -> dict(op, kernel, stride, cin, cout, out_shape, macs). Empty if onnx is missing."""
    try:
        import onnx
        from onnx import numpy_helper, shape_inference
    except ImportError:
        return {}
    m = onnx.load(model_path)
    mi = shape_inference.infer_shapes(m)
    shapes = {}
    for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    inits = {i.name: i for i in m.graph.initializer}
    info = {}
    for n in m.graph.node:
        out = shapes.get(n.output[0], [])
        d = {"op": n.op_type, "out_shape": out, "macs": 0, "kind": n.op_type}
        if n.op_type == "Conv":
            w = numpy_helper.to_array(inits[n.input[1]])
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            s = attrs.get("strides", [1, 1])[0]
            k = w.shape[2]
            d.update(cin=w.shape[1], cout=w.shape[0], k=k, stride=s,
                     macs=int(np.prod(out[1:])) * w.shape[1] * k * k,
                     kind="Conv %dx%d s%d" % (k, k, s))
        info[n.name] = d
    return info


def main():
    ap = argparse.ArgumentParser(description="Per-layer OpenCV DNN benchmark for YOLOv8 ONNX")
    ap.add_argument("--model", default="yolov8n.onnx")
    ap.add_argument("--source", default="bus640.png", help="input image (already 640x640 is best)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1, help="cv2.setNumThreads (default 1; 0 = all cores)")
    ap.add_argument("--no-winograd", action="store_true", help="disable Winograd for 3x3 convs")
    ap.add_argument("--top", type=int, default=20, help="show the N slowest layers")
    ap.add_argument("--csv", default=None, help="write all layers (execution order) to this CSV")
    args = ap.parse_args()

    if args.threads > 0:
        cv2.setNumThreads(args.threads)

    net = cv2.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if args.no_winograd:
        net.enableWinograd(False)

    img = cv2.imread(args.source)
    if img is None:
        sys.exit("ERROR: cannot read %s" % args.source)
    if img.shape[:2] != (args.imgsz, args.imgsz):
        from infer import letterbox
        img = letterbox(img, args.imgsz)[0]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (args.imgsz, args.imgsz), swapRB=True, crop=False)

    for _ in range(args.warmup):
        net.setInput(blob)
        net.forward()

    names = list(net.getLayerNames())
    freq = cv2.getTickFrequency()
    per_run, totals = [], []
    for _ in range(args.runs):
        net.setInput(blob)
        net.forward()
        total, layer_ticks = net.getPerfProfile()
        totals.append(total / freq * 1e3)
        per_run.append(np.array(layer_ticks, dtype=np.float64).ravel() / freq * 1e3)
    times = np.mean(per_run, axis=0)          # ms per layer
    if len(times) != len(names):
        sys.exit("ERROR: %d layer times for %d layer names" % (len(times), len(names)))

    info = onnx_info(args.model)
    rows = []
    for i, (name, t) in enumerate(zip(names, times)):
        d = info.get(name, {})
        kind = d.get("kind") or ("Layout transform" if name.startswith("trlayout")
                                 else re.sub(r"_\d+$", "", name.split("/")[-1]))
        macs = d.get("macs", 0)
        gflops = (2 * macs / 1e9) / (t / 1e3) if macs and t > 0 else 0.0
        block = name.split("/")[1] if name.count("/") >= 2 else "-"
        shape = "x".join(str(v) for v in d.get("out_shape", [])[1:])
        rows.append(dict(idx=i, layer=name, block=block, kind=kind, out=shape,
                         ms=t, macs=macs, gflops=gflops))

    sum_ms = float(times.sum())
    fwd_ms = float(np.mean(totals))
    total_macs = sum(r["macs"] for r in rows)
    print("OpenCV %s | threads=%d | winograd=%s | %d runs after %d warmup | %s"
          % (cv2.__version__, cv2.getNumThreads(), not args.no_winograd, args.runs, args.warmup, args.model))
    print("Forward total %.1f ms (sum of layers %.1f ms), %d layers" % (fwd_ms, sum_ms, len(rows)))
    if total_macs:
        print("Network: %.2f G MACs = %.2f GFLOP -> %.1f GFLOP/s overall"
              % (total_macs / 1e9, 2 * total_macs / 1e9, 2 * total_macs / 1e9 / (fwd_ms / 1e3)))
    else:
        print("(install onnx for shapes, MACs and GFLOP/s:  uv pip install onnx)")

    print("\nBy kernel type")
    print("  %-18s %5s %10s %7s %9s %9s" % ("kind", "count", "ms", "%", "G MACs", "GFLOP/s"))
    agg = collections.OrderedDict()
    for r in rows:
        a = agg.setdefault(r["kind"], [0, 0.0, 0])
        a[0] += 1; a[1] += r["ms"]; a[2] += r["macs"]
    for kind, (n, ms, macs) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        gf = (2 * macs / 1e9) / (ms / 1e3) if macs else 0
        print("  %-18s %5d %10.2f %6.1f%% %9.3f %9s"
              % (kind, n, ms, 100 * ms / sum_ms, macs / 1e9, ("%.1f" % gf) if macs else "-"))

    print("\nBy block (model.N)")
    print("  %-10s %10s %7s %9s" % ("block", "ms", "%", "G MACs"))
    blk = collections.OrderedDict()
    for r in rows:
        b = blk.setdefault(r["block"], [0.0, 0])
        b[0] += r["ms"]; b[1] += r["macs"]
    for b, (ms, macs) in blk.items():
        print("  %-10s %10.2f %6.1f%% %9.3f" % (b, ms, 100 * ms / sum_ms, macs / 1e9))

    print("\nTop %d layers by time" % args.top)
    print("  %-4s %-40s %-14s %-12s %8s %6s %8s %8s"
          % ("#", "layer", "kind", "out CxHxW", "ms", "%", "M MACs", "GFLOP/s"))
    for r in sorted(rows, key=lambda r: -r["ms"])[: args.top]:
        print("  %-4d %-40s %-14s %-12s %8.2f %5.1f%% %8s %8s"
              % (r["idx"], r["layer"][-40:], r["kind"], r["out"], r["ms"], 100 * r["ms"] / sum_ms,
                 ("%.0f" % (r["macs"] / 1e6)) if r["macs"] else "-",
                 ("%.1f" % r["gflops"]) if r["macs"] else "-"))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print("\nwrote %s (%d layers, execution order)" % (args.csv, len(rows)))


if __name__ == "__main__":
    main()