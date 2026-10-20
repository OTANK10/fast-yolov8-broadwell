"""Shared helpers for the Python scripts (preprocessing, layer naming, ownership)."""
import glob, os
import numpy as np
import cv2

PEAK_FLOPS_PER_CYCLE_PER_CORE = float(os.environ.get("PEAK_FPC", 32.0))


def _base_hz():
    """Base (TSC) clock from the CPU name, e.g. 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz'."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and "GHz" in line:
                return float(line.split("@")[1].strip().replace("GHz", "")) * 1e9
    except (OSError, IndexError, ValueError):
        pass
    return float(os.environ.get("BASE_GHZ", 2.4)) * 1e9


NOMINAL_HZ = _base_hz()

OWNERS = {"3x3s1": "Om", "3x3s2": "O'Malley", "1x1": "Simon", "other": "-"}


def default_images():
    import ultralytics
    d = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    return sorted(glob.glob(os.path.join(d, "*.jpg")))


def list_images(src):
    if not src:
        return default_images()
    if os.path.isdir(src):
        return sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")))
    return [src]


def letterbox(path, size=640):
    """Same preprocessing as Ultralytics: keep aspect, pad with 114, RGB, /255, CHW."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


def make_batch(images, batch, size=640):
    imgs = [letterbox(images[i % len(images)], size) for i in range(batch)]
    return np.ascontiguousarray(np.stack(imgs))


def layer_dirname(onnx_name):
    """'/model.2/m.0/cv1/conv/Conv' -> 'model.2.m.0.cv1' ; detect head keeps its path."""
    parts = [p for p in onnx_name.strip("/").split("/") if p not in ("conv", "Conv")]
    return ".".join(parts).replace("..", ".")


def conv_kind(k, stride):
    if k == 3 and stride == 1:
        return "3x3s1"
    if k == 3 and stride == 2:
        return "3x3s2"
    if k == 1:
        return "1x1"
    return "other"
