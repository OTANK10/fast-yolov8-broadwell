# YOLOv8n + OpenCV DNN on RHEL 8.10 (CPU, no sudo) — quickstart

The full guide, with troubleshooting, is the "YOLOv8n on CPU with OpenCV DNN — RHEL 8.10 guide" doc.
This file is the command-only version, so it can travel with the files to an offline machine.

Contents of this folder:
- `yolov8n.onnx` (640, ready to use) and `yolov8n_320.onnx` (320, faster). Both are already
  patched for OpenCV <= 4.6.
- `infer.py`: inference with numpy + opencv-python-headless only
- `export_onnx.py`: re-export from yolov8n.pt (needs torch + ultralytics; run on any other machine)
- `requirements-infer.txt`, `requirements-export.txt`
- `bus.jpg`: test image. `bus_det.jpg` is the expected output.

## Path A (recommended): Python 3.12 in $HOME via uv

```bash
mkdir -p ~/.local/bin
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
  | tar -xz -C ~/.local/bin --strip-components=1
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv -p 3.12 ~/yolo312
source ~/yolo312/bin/activate
uv pip install -r requirements-infer.txt        # -> opencv-python-headless 5.0.0.93, numpy 2.5.4
python infer.py --model yolov8n.onnx --source bus.jpg --save bus_out.jpg --no-show
```

## Path B: RHEL system Python 3.6.8

```bash
/usr/libexec/platform-python -m venv ~/yolo36
source ~/yolo36/bin/activate
python -m pip install --upgrade "pip==21.3.1"   # required: pip 9.0.3 can't see the wheels
python -m pip install opencv-python-headless==4.6.0.66 numpy==1.19.5
python infer.py --model yolov8n.onnx --source bus.jpg --save bus_out.jpg --no-show
```

## Expected output on bus.jpg (both paths)

```
  person         0.890  [670.4, 380.6, 809.9, 879.7]
  person         0.883  [221.7, 407.4, 343.8, 856.2]
  person         0.878  [50.7, 397.3, 244.4, 905.4]
  bus            0.843  [31.5, 230.7, 801.4, 775.8]
  person         0.436  [0.4, 549.8, 57.8, 868.3]
```

## More usage

```bash
python infer.py --model yolov8n.onnx --source input.mp4 --save output.mp4 --no-show   # video
python infer.py --model yolov8n.onnx --source 0                                       # webcam (needs a display)
python infer.py --model yolov8n_320.onnx --imgsz 320 --source bus.jpg --no-show       # ~4x faster
python infer.py --model yolov8n.onnx --source bus.jpg --bench 20 --no-show            # benchmark
```

## Offline install (download wheels on another machine)

```bash
# Python 3.6 target
pip download --no-deps -d wheels_py36 --only-binary=:all: \
  --python-version 3.6 --implementation cp --abi cp36m --abi abi3 --abi none \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux2010_x86_64 --platform manylinux1_x86_64 --platform any \
  opencv-python-headless==4.6.0.66 numpy==1.19.5 pip==21.3.1
# on the target:
python -m pip install --no-index --find-links wheels_py36 --upgrade pip
python -m pip install --no-index --find-links wheels_py36 opencv-python-headless==4.6.0.66 numpy==1.19.5

# Python 3.12 target
pip download -d wheels_py312 --only-binary=:all: --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  opencv-python-headless==5.0.0.93 numpy==2.5.4
# on the target:
uv pip install --no-index --find-links wheels_py312 opencv-python-headless==5.0.0.93 numpy==2.5.4
```

## Re-exporting the model (optional, on a machine with internet)

```bash
python3 -m venv ~/yolo-export && source ~/yolo-export/bin/activate
pip install -U pip && pip install -r requirements-export.txt
python export_onnx.py                # -> yolov8n.onnx (opset 12, patched for OpenCV <= 4.6)
python export_onnx.py --imgsz 320    # -> yolov8n_320.onnx
```

Always use `export_onnx.py`. A plain `YOLO(...).export(format="onnx")` file does not load in OpenCV 4.6
(Python 3.6), and `dynamic=True`, `nms=True` and `half=True` break cv2.dnn.
