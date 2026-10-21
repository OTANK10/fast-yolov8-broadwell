#!/usr/bin/env python3
"""Export YOLOv8n PyTorch weights (yolov8n.pt) to ONNX for OpenCV's cv2.dnn.

Run this on ANY machine that can install torch + ultralytics (it does not have
to be the target machine). Only the resulting .onnx file needs to be copied to
the target machine, which then needs only numpy + opencv-python-headless.

Usage:
    python export_onnx.py                          # -> yolov8n.onnx (opset 12, 640x640)
    python export_onnx.py --imgsz 320              # -> yolov8n_320.onnx
    python export_onnx.py --no-compat-patch        # skip the OpenCV<4.7 compatibility patch

About the compatibility patch (on by default):
    The exported graph ends with  Mul(boxes[1,4,8400], strides[1,8400]).
    OpenCV >= 4.7 handles that broadcast, but OpenCV <= 4.6 (the newest
    version available for Python 3.6, e.g. RHEL 8 system python) fails with
      "(-215:Assertion failed) start <= (int)shape.size() ... in function 'total'".
    The patch replaces such broadcast constants by an explicitly tiled copy
    (e.g. [1,4,8400]); the math is identical, the file grows by ~130 KB.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def opencv_compat_patch(path):
    """Tile constant operands of elementwise ops to the full shape of the other operand."""
    import numpy as np
    import onnx
    from onnx import numpy_helper, shape_inference

    model = onnx.load(str(path))
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    inits = {i.name: i for i in model.graph.initializer}
    n_patched = 0
    for node in model.graph.node:
        if node.op_type not in ("Mul", "Add", "Sub", "Div") or len(node.input) != 2:
            continue
        for k, (const_name, other_name) in enumerate(((node.input[1], node.input[0]),
                                                      (node.input[0], node.input[1]))):
            if const_name not in inits or other_name not in shapes:
                continue
            const = numpy_helper.to_array(inits[const_name])
            target = shapes[other_name]
            if const.size <= 1 or not target or not all(target) or list(const.shape) == target:
                continue
            if len(const.shape) == len(target) and const.shape[1:] == tuple(target[1:]):
                continue  # already full shape apart from batch
            full = np.ascontiguousarray(np.broadcast_to(const, target)).astype(const.dtype)
            new_name = const_name + "_tiled_" + node.name.strip("/").replace("/", "_")
            model.graph.initializer.append(numpy_helper.from_array(full, new_name))
            idx = 1 if k == 0 else 0
            node.input[idx] = new_name
            n_patched += 1
            print("[compat] %s: constant %s -> %s" % (node.name, tuple(const.shape), tuple(full.shape)))
            break
    if n_patched:
        used = {i for n in model.graph.node for i in n.input}
        unused = [i for i in model.graph.initializer if i.name not in used]
        for i in unused:
            model.graph.initializer.remove(i)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    print("[compat] patched %d node(s)" % n_patched)


def main():
    ap = argparse.ArgumentParser(description="Export YOLOv8 .pt to ONNX for cv2.dnn")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="path/name of .pt weights (auto-downloaded from GitHub if missing)")
    ap.add_argument("--opset", type=int, default=12,
                    help="ONNX opset (12 = safe for every OpenCV 4.x/5.x tested)")
    ap.add_argument("--imgsz", type=int, default=640, help="square input size (multiple of 32)")
    ap.add_argument("--no-simplify", action="store_true", help="disable onnxslim simplification")
    ap.add_argument("--no-compat-patch", action="store_true",
                    help="skip the OpenCV<=4.6 broadcast patch (see module docstring)")
    ap.add_argument("--out", default=None, help="optional output path for the .onnx file")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so --help works without ultralytics

    weights = Path(args.weights)
    if not weights.is_file():
        YOLO(args.weights)  # triggers the GitHub auto-download into the current dir
        weights = Path(Path(args.weights).name)
    stem = weights.stem
    out = Path(args.out) if args.out else Path(
        "%s.onnx" % stem if args.imgsz == 640 else "%s_%d.onnx" % (stem, args.imgsz))

    # ultralytics always writes <weights_stem>.onnx next to the weights; export from a
    # temp copy so an existing .onnx (e.g. the 640 model) is never overwritten.
    tmp = Path(tempfile.mkdtemp(prefix="yolo_export_"))
    try:
        tmp_w = tmp / weights.name
        shutil.copy2(str(weights), str(tmp_w))
        exported = YOLO(str(tmp_w)).export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=not args.no_simplify,
            dynamic=False,   # fixed input shape: required/safest for cv2.dnn
            nms=False,       # keep raw 1x84xN output; NMS is done in infer.py with cv2.dnn.NMSBoxes
        )
        shutil.move(str(exported), str(out))
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    onnx_path = out

    if not args.no_compat_patch:
        opencv_compat_patch(onnx_path)

    # Quick sanity check with OpenCV, if available in this environment.
    try:
        import cv2
        import numpy as np
        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        net.setInput(np.zeros((1, 3, args.imgsz, args.imgsz), np.float32))
        out = net.forward()
        print("[check] OpenCV %s loaded %s, output shape %s" % (cv2.__version__, onnx_path, out.shape))
    except ImportError:
        print("[check] OpenCV not installed here; skipping cv2.dnn check")

    size_mb = onnx_path.stat().st_size / 1e6
    print("[done] %s (%.1f MB). Copy this file to the target machine." % (onnx_path, size_mb))


if __name__ == "__main__":
    sys.exit(main())
