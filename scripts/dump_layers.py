#!/usr/bin/env python3
"""Run the baseline ONNX model on real images and dump every conv layer's
input, weights, bias and outputs, so the C kernels are tested on real data.

    python scripts/dump_layers.py --batch 1                 # 2 sample images shipped with ultralytics
    python scripts/dump_layers.py --batch 16 --images /scratch/$USER/coco/val2017

Writes data/tensors/b<N>/<layer>/{meta.txt,input.bin,weight.bin,bias.bin,conv_out.bin,output.bin}
plus data/tensors/b<N>/layers.csv (one row per conv: kind, owner, shapes, FLOPs).
Use --only 3x3s1 to write tensors for just your layers (batch 16 of everything is ~3 GB).
MaxPool nodes (SPPF) are dumped to data/tensors/b<N>/<name>/{meta.txt,input.bin,output.bin}.
"""
import argparse, csv, os, sys
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from common_py import OWNERS, conv_kind, layer_dirname, list_images, make_batch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--onnx", default=None, help="default data/models/yolov8n_b<batch>.onnx")
ap.add_argument("--images", default=None, help="image file or folder (default: ultralytics sample images)")
ap.add_argument("--out", default=None, help="default data/tensors/b<batch>")
ap.add_argument("--only", default=None, help="comma list of kinds to write tensors for, e.g. 3x3s1 (saves disk at big batch)")
a = ap.parse_args()
onnx_path = a.onnx or f"data/models/yolov8n_b{a.batch}.onnx"
out_root = a.out or f"data/tensors/b{a.batch}"

model = onnx.load(onnx_path)
g = model.graph
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

# Every tensor we want to see becomes an extra graph output.
want = set()
for n in g.node:
    if n.op_type in ("Conv", "MaxPool"):
        want.update([n.input[0], n.output[0]])
        for c in consumers.get(n.output[0], []):
            if c.op_type == "Mul":
                want.add(c.output[0])
existing = {o.name for o in g.output}
for t in sorted(want - existing):
    g.output.append(onnx.helper.make_empty_tensor_value_info(t))

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep every node as exported
sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
x = make_batch(list_images(a.images), a.batch)
names = [o.name for o in sess.get_outputs()]
vals = dict(zip(names, sess.run(names, {sess.get_inputs()[0].name: x})))


def save(d, fname, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(d, fname))


rows = []
os.makedirs(out_root, exist_ok=True)
for n in g.node:
    if n.op_type not in ("Conv", "MaxPool"):
        continue
    attrs = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
    inp, out = vals[n.input[0]], vals[n.output[0]]
    if a.only and n.op_type == "MaxPool" and "pool" not in a.only:
        continue
    d = os.path.join(out_root, layer_dirname(n.name))
    os.makedirs(d, exist_ok=True)
    save(d, "input.bin", inp)

    if n.op_type == "MaxPool":
        k, s, p = attrs["kernel_shape"][0], attrs["strides"][0], attrs["pads"][0]
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{inp.shape[0]} {inp.shape[1]} {inp.shape[2]} {inp.shape[3]} {k} {s} {p} {out.shape[2]} {out.shape[3]}\n")
        save(d, "output.bin", out)
        continue

    w = init[n.input[1]]
    b = init[n.input[2]] if len(n.input) > 2 else np.zeros(w.shape[0], np.float32)
    if attrs.get("group", 1) != 1:
        print("skipping grouped conv", n.name); continue
    stride, pad = attrs["strides"][0], attrs["pads"][0]
    mul = [c for c in consumers.get(n.output[0], []) if c.op_type == "Mul"]
    has_silu = int(bool(mul))
    N, C, H, W = inp.shape
    K, _, R, S = w.shape
    OH, OW = out.shape[2], out.shape[3]
    kind = conv_kind(R, stride) if "dfl" not in n.name else "other"
    if not a.only or kind in a.only.split(","):
        save(d, "weight.bin", w); save(d, "bias.bin", b); save(d, "conv_out.bin", out)
        save(d, "output.bin", vals[mul[0].output[0]] if mul else out)
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{N} {C} {H} {W} {K} {R} {S} {stride} {pad} {OH} {OW} {has_silu}\n")
    else:
        os.remove(os.path.join(d, "input.bin")); os.rmdir(d)
    flops = 2 * N * K * OH * OW * C * R * S
    rows.append(dict(dir=layer_dirname(n.name), onnx_name=n.name, kind=kind, owner=OWNERS[kind],
                     N=N, C=C, H=H, W=W, K=K, R=R, S=S, stride=stride, OH=OH, OW=OW,
                     silu=has_silu, flops=flops))

with open(os.path.join(out_root, "layers.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

tot = sum(r["flops"] for r in rows)
print(f"dumped {len(rows)} conv layers to {out_root}  (total {tot/1e9:.2f} GFLOP for batch {a.batch})")
for kind in ("3x3s1", "3x3s2", "1x1", "other"):
    ks = [r for r in rows if r["kind"] == kind]
    fl = sum(r["flops"] for r in ks)
    print(f"  {kind:6s} {OWNERS[kind]:9s} {len(ks):3d} layers  {fl/1e9:6.2f} GFLOP  ({100*fl/tot:4.1f}%)")
