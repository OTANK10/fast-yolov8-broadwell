#!/usr/bin/env python3
"""YOLOv8 object detection with OpenCV's DNN module (CPU only).

Needs only:  numpy  +  opencv-python-headless (or opencv-python)
No torch / ultralytics required. Works with Python 3.6+.

Examples:
    python infer.py --model yolov8n.onnx --source bus.jpg --save out.jpg --no-show
    python infer.py --model yolov8n.onnx --source video.mp4 --save out.mp4 --no-show
    python infer.py --model yolov8n.onnx --source 0            # webcam 0, live window
    python infer.py --model yolov8n.onnx --source bus.jpg --bench 20 --no-show
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def letterbox(img, new_size=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to new_size x new_size (same as ultralytics LetterBox)."""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new_size - new_w) / 2.0, (new_size - new_h) / 2.0
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top


class YOLOv8CV:
    def __init__(self, model_path, imgsz=640, conf=0.25, iou=0.7, max_det=300):
        if not os.path.isfile(model_path):
            raise FileNotFoundError("model not found: %s" % model_path)
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def preprocess(self, img_bgr):
        lb, r, padx, pady = letterbox(img_bgr, self.imgsz)
        # scale 1/255, BGR->RGB (swapRB=True), NCHW float32, no crop
        blob = cv2.dnn.blobFromImage(lb, scalefactor=1.0 / 255.0,
                                     size=(self.imgsz, self.imgsz),
                                     mean=(0, 0, 0), swapRB=True, crop=False)
        return blob, r, padx, pady

    def postprocess(self, out, r, padx, pady, orig_shape):
        # out: (1, 84, N) -> (N, 84); rows = [cx, cy, w, h, 80 class scores] (no objectness in v8)
        preds = np.squeeze(out, 0)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        scores_all = preds[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]
        keep = scores > self.conf
        preds, scores, class_ids = preds[keep], scores[keep], class_ids[keep]
        if len(scores) == 0:
            return []

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        # undo letterbox: remove padding, divide by resize ratio -> original image pixels
        x1 = (cx - w / 2 - padx) / r
        y1 = (cy - h / 2 - pady) / r
        bw = w / r
        bh = h / r

        # Class-aware NMS (like ultralytics): offset boxes per class so different
        # classes never suppress each other, then use plain cv2.dnn.NMSBoxes
        # (works on every OpenCV 4.x; NMSBoxesBatched only exists in >= 4.7).
        offset = class_ids.astype(np.float32) * 7680.0
        nms_boxes = np.stack([x1 + offset, y1 + offset, bw, bh], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(), self.conf, self.iou)
        idxs = np.array(idxs).reshape(-1)[: self.max_det]

        H, W = orig_shape[:2]
        dets = []
        for i in idxs:
            bx1 = float(np.clip(x1[i], 0, W))
            by1 = float(np.clip(y1[i], 0, H))
            bx2 = float(np.clip(x1[i] + bw[i], 0, W))
            by2 = float(np.clip(y1[i] + bh[i], 0, H))
            dets.append((int(class_ids[i]), float(scores[i]), bx1, by1, bx2, by2))
        dets.sort(key=lambda d: -d[1])
        return dets

    def __call__(self, img_bgr):
        t0 = time.perf_counter()
        blob, r, padx, pady = self.preprocess(img_bgr)
        t1 = time.perf_counter()
        self.net.setInput(blob)
        try:
            out = self.net.forward()
        except cv2.error as e:
            sys.exit("ERROR: cv2.dnn failed to run the model with OpenCV %s:\n  %s\n"
                     "Hints: export with export_onnx.py (opset 12, dynamic=False, nms=False, FP32, "
                     "compat patch ON - needed for OpenCV <= 4.6), and pass the same --imgsz "
                     "that was used at export." % (cv2.__version__, str(e).strip().splitlines()[-1]))
        t2 = time.perf_counter()
        dets = self.postprocess(out, r, padx, pady, img_bgr.shape)
        t3 = time.perf_counter()
        times = ((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)
        return dets, times


def color_for(cls_id):
    rng = np.random.RandomState(cls_id * 7 + 3)
    return tuple(int(c) for c in rng.randint(64, 256, 3))


def draw(img, dets, names=COCO_NAMES):
    for cls_id, score, x1, y1, x2, y2 in dets:
        c = color_for(cls_id)
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, c, 2)
        name = names[cls_id] if cls_id < len(names) else str(cls_id)
        label = "%s %.2f" % (name, score)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty = p1[1] - 4 if p1[1] - th - 6 > 0 else p1[1] + th + 4
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 2, ty + base - 2), c, -1)
        cv2.putText(img, label, (p1[0] + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img


def try_show(win, img, state):
    """cv2.imshow that degrades gracefully (headless OpenCV or no $DISPLAY)."""
    if not state["show"]:
        return True
    try:
        cv2.imshow(win, img)
        key = cv2.waitKey(1 if state["video"] else 0) & 0xFF
        return key not in (ord("q"), 27)
    except cv2.error as e:
        print("[warn] cv2.imshow unavailable (%s). Continuing without display; "
              "use --no-show to silence this." % str(e).strip().splitlines()[-1])
        state["show"] = False
        return True


def print_dets(dets):
    for cls_id, score, x1, y1, x2, y2 in dets:
        print("  %-14s %.3f  [%.1f, %.1f, %.1f, %.1f]" % (COCO_NAMES[cls_id], score, x1, y1, x2, y2))


def run_image(model, args, state):
    img = cv2.imread(args.source)
    if img is None:
        sys.exit("ERROR: cannot read image %s" % args.source)
    dets, (tp, ti, tq) = model(img)
    print("%s: %dx%d, %d detections | pre %.1f ms, infer %.1f ms, post %.1f ms"
          % (args.source, img.shape[1], img.shape[0], len(dets), tp, ti, tq))
    print_dets(dets)

    if args.bench > 0:
        for _ in range(args.warmup):
            model(img)
        tot, inf = [], []
        for _ in range(args.bench):
            t0 = time.perf_counter()
            _, (tp, ti, tq) = model(img)
            tot.append((time.perf_counter() - t0) * 1000)
            inf.append(ti)
        print("[bench] %d runs after %d warmup @ imgsz %d, threads=%d: "
              "forward mean %.1f ms (min %.1f, median %.1f) | total pipeline mean %.1f ms (%.1f FPS)"
              % (args.bench, args.warmup, args.imgsz, cv2.getNumThreads(), np.mean(inf),
                 np.min(inf), np.median(inf), np.mean(tot), 1000.0 / np.mean(tot)))

    vis = draw(img.copy(), dets)
    if args.save is not None:
        out = args.save or (os.path.splitext(os.path.basename(args.source))[0] + "_det.jpg")
        cv2.imwrite(out, vis)
        print("saved", out)
    try_show("YOLOv8 (OpenCV DNN)", vis, state)


def run_stream(model, args, state):
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video source %r" % (args.source,))
    writer = None
    n, tsum = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets, (tp, ti, tq) = model(frame)
        n += 1
        tsum += tp + ti + tq
        vis = draw(frame, dets)
        msg = "infer %.1f ms (%.1f FPS)" % (ti, 1000.0 / max(tp + ti + tq, 1e-6))
        cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        print("frame %d: %d dets | pre %.1f ms, infer %.1f ms, post %.1f ms" % (n, len(dets), tp, ti, tq))
        if args.save is not None:
            if writer is None:
                out = args.save or "output_det.mp4"
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                if fps <= 0 or fps > 240:
                    fps = 25.0
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (vis.shape[1], vis.shape[0]))
                print("writing", out)
            writer.write(vis)
        if not try_show("YOLOv8 (OpenCV DNN)", vis, state):
            break
    cap.release()
    if writer is not None:
        writer.release()
    if n:
        print("processed %d frames, mean %.1f ms/frame (%.1f FPS)" % (n, tsum / n, 1000.0 * n / tsum))


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 ONNX inference with OpenCV DNN (CPU)")
    ap.add_argument("--model", default="yolov8n.onnx", help="path to exported .onnx")
    ap.add_argument("--source", required=True, help="image path, video path, or webcam index (e.g. 0)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (ultralytics default 0.7)")
    ap.add_argument("--imgsz", type=int, default=640, help="must equal the --imgsz used at export")
    ap.add_argument("--save", nargs="?", const="", default=None,
                    help="save annotated output (optionally give a path)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (headless/SSH)")
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads (0 = OpenCV default)")
    ap.add_argument("--bench", type=int, default=0, help="images only: time N extra runs")
    ap.add_argument("--warmup", type=int, default=3, help="warmup runs before --bench")
    args = ap.parse_args()

    if args.threads > 0:
        cv2.setNumThreads(args.threads)
    print("OpenCV %s, numpy %s, Python %s" % (cv2.__version__, np.__version__, sys.version.split()[0]))

    show = not args.no_show
    if show and sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
            and not os.environ.get("WAYLAND_DISPLAY"):
        print("[info] no $DISPLAY found -> running with --no-show")
        show = False

    model = YOLOv8CV(args.model, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    is_image = args.source.lower().endswith(IMG_EXTS)
    state = {"show": show, "video": not is_image}
    if is_image:
        run_image(model, args, state)
    else:
        run_stream(model, args, state)
    if state["show"]:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
