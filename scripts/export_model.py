"""Download pretrained YOLOv8n and export ONNX. Requires Ultralytics + ONNX.

Usage: python scripts/export_model.py
This only exports the model; the team must still implement a mechanism to
extract per-layer weights/activations/reference outputs for kernel testing.
"""
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
out_dir = root / "data" / "models"
out_dir.mkdir(parents=True, exist_ok=True)
model = YOLO("yolov8n.pt")  # Explicit YOLOv8 Nano; not the newest YOLO.
onnx_path = Path(model.export(format="onnx", imgsz=640, simplify=False))
target = out_dir / "yolov8n.onnx"
if onnx_path.resolve() != target.resolve():
    onnx_path.replace(target)
print(f"Exported: {target}")
