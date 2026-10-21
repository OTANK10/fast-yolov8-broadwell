# Baseline: YOLOv8n with OpenCV DNN (CPU)

Reference implementation for fast-yolov8-broadwell. Our own kernels must match its
detections and beat its single-thread time.

## Files

| File | What it is |
| --- | --- |
| `yolov8n.onnx` | YOLOv8n, opset 12, fixed 1x3x640x640 input, output 1x84x8400 (patched to load in OpenCV <= 4.6) |
| `yolov8n_320.onnx` | Same weights, 320x320 input, output 1x84x2100 (faster, for debugging) |
| `bus640.png` | Test image: Ultralytics `bus.jpg` letterboxed to 640x640 (grey 114 padding), lossless |
| `infer.py` | Inference with numpy + opencv-python-headless only |
| `export_onnx.py` | Re-export `yolov8n.pt` to ONNX (needs torch + ultralytics) |
| `baseline_opencv_1t_640.txt` | Recorded single-thread benchmark + CPU info |

## Run

    source ~/venvs/yolo312/bin/activate
    taskset -c 2 python infer.py --bench 20 --no-show      # 1 thread (default), pinned to core 2
    python infer.py --threads 0 --bench 20 --no-show        # all cores

Network input = `bus640.png` pixels / 255, BGR -> RGB, NCHW float32 1x3x640x640.
Because the image is already 640x640, no resizing or padding happens.

## Expected detections (640x640 pixel coordinates, conf 0.25, NMS IoU 0.7)

| Class | Conf | x1, y1, x2, y2 |
| --- | --- | --- |
| person | 0.890 | 477.3, 225.5, 560.0, 521.3 |
| person | 0.883 | 211.4, 241.4, 283.7, 507.4 |
| person | 0.878 | 110.0, 235.4, 224.8, 536.5 |
| bus | 0.843 | 98.7, 136.7, 554.9, 459.7 |
| person | 0.436 | 80.3, 325.8, 114.2, 514.5 |

## Baseline (OpenCV 5.0.0, Python 3.12.15, 1 thread, core 2)

- Forward pass: ~181 ms mean at 640 (see `baseline_opencv_1t_640.txt` for this run)
- Work: 4.37 G multiply-adds = 8.74 GFLOP per image; 64 convs (3x3 s1 66%, 1x1 19%, 3x3 s2 15%)
- Note: OpenCV uses Winograd for 3x3 stride-1 convs by default
- Broadwell single-core peak: 32 FLOP/cycle (2 FMA units x 8 floats x 2) -> ~90 ms floor at 3 GHz for direct conv

## Environment setup (RHEL 8.10, no sudo)

    mkdir -p ~/.local/bin
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
      | tar -xz -C ~/.local/bin --strip-components=1
    export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy
    uv python install 3.12 && uv venv -p 3.12 ~/venvs/yolo312
    source ~/venvs/yolo312/bin/activate && uv pip install -r requirements-infer.txt
