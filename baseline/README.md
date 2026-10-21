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
| `bench_layers.py` | Per-layer timing via OpenCV's profiler; with `onnx` installed, adds shapes, MACs, GFLOP/s |
| `export_onnx.py` | Re-export `yolov8n.pt` to ONNX (needs torch + ultralytics) |
| `baseline_opencv_1t_640.txt` | Recorded single-thread benchmark + CPU info |
| `layers_1t_direct.csv` | Per-layer times, 1 thread, Winograd off, execution order |

## Run

    source ~/venvs/yolo312/bin/activate
    taskset -c 2 python infer.py --bench 20 --no-show                  # end to end, 1 thread (default), core 2
    taskset -c 2 python bench_layers.py --no-winograd --csv layers_1t_direct.csv   # per layer, direct conv
    python infer.py --threads 0 --bench 20 --no-show                    # all cores

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

## Baseline (OpenCV 5.0.0, Python 3.12.15, 1 thread, pinned to core 2)

| Measurement | Forward pass |
| --- | --- |
| `bench_layers.py --no-winograd` (direct 3x3 conv) | 172.4 ms |
| `infer.py --bench 20` (OpenCV default, Winograd on) | 181.2 ms mean, 180.5 median, 172.1 min |

- Work: 4.37 G multiply-adds = 8.74 GFLOP per image; 64 convs (3x3 s1 66%, 1x1 19%, 3x3 s2 15% of MACs)
- Broadwell single-core peak: 32 FLOP/cycle (2 FMA units x 8 floats x 2) -> ~90 ms floor at 3 GHz for direct conv
- Winograd gives no measurable gain on this machine, so the direct-conv number is the one to beat

## Per-layer benchmark (OpenCV, 1 thread, direct conv)

Winograd off, mean of 20 runs after 3 warmup, `bus640.png`.
**Forward 172.4 ms, 50.7 GFLOP/s overall; convs alone 155.3 ms at 56.3 GFLOP/s.**

By kernel type:

| Kind | Count | ms | % of time | G MACs | GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv (all) | 64 | 155.31 | 90.1% | 4.372 | 56.3 |
| Layout transform | 45 | 6.53 | 3.8% | – | – |
| Concat | 17 | 3.79 | 2.2% | – | – |
| Transpose | 2 | 1.89 | 1.1% | – | – |
| Softmax | 1 | 1.51 | 0.9% | – | – |
| Resize | 2 | 0.81 | 0.5% | – | – |
| Split | 8 | 0.77 | 0.4% | – | – |
| Sigmoid | 1 | 0.68 | 0.4% | – | – |
| MaxPool | 3 | 0.63 | 0.4% | – | – |
| Add | 8 | 0.42 | 0.2% | – | – |
| Other (Mul, Sub, Slice, Div, Reshape) | 14 | 0.07 | 0.0% | – | – |

By block:

| Block | Role | ms | % of time | G MACs | GFLOP/s |
| --- | --- | ---: | ---: | ---: | ---: |
| layout transforms | OpenCV overhead | 6.53 | 3.8% | – | – |
| model.0 | Conv 3x3 s2 | 6.99 | 4.1% | 0.044 | 12.7 |
| model.1 | Conv 3x3 s2 | 5.61 | 3.3% | 0.118 | 42.1 |
| model.2 | C2f | 11.04 | 6.4% | 0.184 | 33.2 |
| model.3 | Conv 3x3 s2 | 5.02 | 2.9% | 0.118 | 47.0 |
| model.4 | C2f | 14.44 | 8.4% | 0.315 | 43.6 |
| model.5 | Conv 3x3 s2 | 5.58 | 3.2% | 0.118 | 42.3 |
| model.6 | C2f | 9.39 | 5.4% | 0.315 | 67.0 |
| model.7 | Conv 3x3 s2 | 6.46 | 3.7% | 0.118 | 36.5 |
| model.8 | C2f | 5.95 | 3.5% | 0.184 | 61.7 |
| model.9 | SPPF | 3.61 | 2.1% | 0.066 | 36.3 |
| model.10–11 | Upsample + concat | 0.51 | 0.3% | – | – |
| model.12 | C2f | 7.81 | 4.5% | 0.236 | 60.4 |
| model.13–14 | Upsample + concat | 1.22 | 0.7% | – | – |
| model.15 | C2f | 9.40 | 5.5% | 0.236 | 50.2 |
| model.16–17 | Conv 3x3 s2 + concat | 2.89 | 1.7% | 0.059 | 40.8 |
| model.18 | C2f | 5.61 | 3.3% | 0.197 | 70.1 |
| model.19–20 | Conv 3x3 s2 + concat | 3.25 | 1.9% | 0.059 | 36.3 |
| model.21 | C2f | 6.63 | 3.8% | 0.197 | 59.3 |
| model.22 | Detect head + decode | 54.45 | 31.6% | 1.811 | 66.5 |

Top 20 layers by time:

| # | Layer | Kind | Out C×H×W | ms | % | M MACs | GFLOP/s |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 95 | `/model.22/cv3.0/cv3.0.1/conv/Conv` | Conv 3x3 s1, 80→80 | 80×80×80 | 10.01 | 5.8% | 369 | 73.7 |
| 92 | `/model.22/cv3.0/cv3.0.0/conv/Conv` | Conv 3x3 s1, 64→80 | 80×80×80 | 8.07 | 4.7% | 295 | 73.1 |
| 1 | `/model.0/conv/Conv` | Conv 3x3 s2, 3→16 | 16×320×320 | 6.99 | 4.1% | 44 | 12.7 |
| 91 | `/model.22/cv2.0/cv2.0.0/conv/Conv` | Conv 3x3 s1, 64→64 | 64×80×80 | 6.51 | 3.8% | 236 | 72.5 |
| 48 | `/model.7/conv/Conv` | Conv 3x3 s2, 128→256 | 256×20×20 | 6.46 | 3.7% | 118 | 36.5 |
| 94 | `/model.22/cv2.0/cv2.0.1/conv/Conv` | Conv 3x3 s1, 64→64 | 64×80×80 | 6.43 | 3.7% | 236 | 73.4 |
| 2 | `/model.1/conv/Conv` | Conv 3x3 s2, 16→32 | 32×160×160 | 5.61 | 3.3% | 118 | 42.1 |
| 31 | `/model.5/conv/Conv` | Conv 3x3 s2, 64→128 | 128×40×40 | 5.58 | 3.2% | 118 | 42.3 |
| 14 | `/model.3/conv/Conv` | Conv 3x3 s2, 32→64 | 64×80×80 | 5.02 | 2.9% | 118 | 47.0 |
| 114 | `/model.22/cv3.1/cv3.1.0/conv/Conv` | Conv 3x3 s1, 128→80 | 80×40×40 | 3.99 | 2.3% | 147 | 73.9 |
| 68 | `/model.12/cv1/conv/Conv` | Conv 1x1 s1, 384→128 | 128×40×40 | 3.35 | 1.9% | 79 | 47.0 |
| 112 | `/model.19/conv/Conv` | Conv 3x3 s2, 128→128 | 128×20×20 | 3.21 | 1.9% | 59 | 36.7 |
| 113 | `/model.22/cv2.1/cv2.1.0/conv/Conv` | Conv 3x3 s1, 128→64 | 64×40×40 | 3.19 | 1.9% | 118 | 74.0 |
| 0 | `trlayout.0` | Layout transform | – | 2.89 | 1.7% | – | – |
| 90 | `/model.16/conv/Conv` | Conv 3x3 s2, 64→64 | 64×40×40 | 2.82 | 1.6% | 59 | 41.8 |
| 7 | `/model.2/m.0/cv1/conv/Conv` | Conv 3x3 s1, 16→16 | 16×160×160 | 2.78 | 1.6% | 59 | 42.4 |
| 8 | `/model.2/m.0/cv2/conv/Conv` | Conv 3x3 s1, 16→16 | 16×160×160 | 2.77 | 1.6% | 59 | 42.6 |
| 19 | `/model.4/m.0/cv1/conv/Conv` | Conv 3x3 s1, 32→32 | 32×80×80 | 2.72 | 1.6% | 59 | 43.4 |
| 20 | `/model.4/m.0/cv2/conv/Conv` | Conv 3x3 s1, 32→32 | 32×80×80 | 2.71 | 1.6% | 59 | 43.5 |
| 24 | `/model.4/m.1/cv1/conv/Conv` | Conv 3x3 s1, 32→32 | 32×80×80 | 2.71 | 1.6% | 59 | 43.5 |

Takeaways:

- **Wide 3x3 stride-1 convs (64–128 channels) reach ~73 GFLOP/s**, OpenCV's best case. They are mostly in the P3 head, which alone is ~28% of the time.
- **Stride-2 convs and narrow 3x3s (16–32 channels) reach only 36–47 GFLOP/s**, about half that. These are the easiest wins.
- **The first conv (3→16 channels) runs at 12.7 GFLOP/s**, the worst layer: 3 input channels can't fill 8-wide AVX2 registers. It needs its own kernel.
- **~17 ms (10%) is not conv**: 6.5 ms of layout transforms plus concat, transpose and softmax. A single-layout pipeline avoids most of it.

## Environment setup (RHEL 8.10, no sudo)

    mkdir -p ~/.local/bin
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
      | tar -xz -C ~/.local/bin --strip-components=1
    export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy
    uv python install 3.12 && uv venv -p 3.12 ~/venvs/yolo312
    source ~/venvs/yolo312/bin/activate && uv pip install -r requirements-infer.txt
    uv pip install onnx        # optional, for bench_layers.py shapes/MACs/GFLOP/s