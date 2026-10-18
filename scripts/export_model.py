"""Download pretrained YOLOv8n and export a static-batch ONNX model.

Usage: python scripts/export_model.py [--batch N]   -> data/models/yolov8n_b<N>.onnx

The Ultralytics exporter fuses each BatchNorm into the preceding Conv's weights
and bias, so the exported graph has no BatchNormalization nodes: every Conv block
is Conv -> Sigmoid -> Mul (x * sigmoid(x) = SiLU). Per-layer tensors are then
extracted by scripts/dump_layers.py.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

root = Path(__file__).resolve().parents[1]
out_dir = root / "data" / "models"
out_dir.mkdir(parents=True, exist_ok=True)
pt = out_dir / "yolov8n.pt"
if not pt.exists():
    YOLO("yolov8n.pt")  # Explicit YOLOv8 Nano; downloads from ultralytics/assets releases
    Path("yolov8n.pt").replace(pt)
model = YOLO(str(pt))
onnx_path = Path(model.export(format="onnx", opset=12, imgsz=a.imgsz, batch=a.batch,
                              dynamic=False, simplify=True))
target = out_dir / f"yolov8n_b{a.batch}.onnx"
onnx_path.replace(target)
print(f"Exported: {target}")
