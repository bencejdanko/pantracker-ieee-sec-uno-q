"""Host parity of the YOLOv8n XNNPACK exports + python/detector.py decode vs Ultralytics' own predict().

  python parity_yolov8.py            (Vulkan variants: bench/det_parity.py on the board)
Metric: per image with a reference person, IoU of our top box with the best-matching reference box, and whether
our top box matches the reference top box (multi-person scenes can rank differently, esp. int8: scores saturate).
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import detector as D  # noqa: E402


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih + 1e-9)


class RTModel:  # host ExecuTorch runtime with the same .run() interface as infer.Model
    def __init__(self, p):
        from executorch.runtime import Runtime
        self.meth = Runtime.get().load_program(p).load_method("forward")

    def run(self, x):
        return [o.numpy() for o in self.meth.execute([torch.from_numpy(np.ascontiguousarray(x))])]


if __name__ == "__main__":
    from ultralytics import YOLO
    ref = YOLO(os.path.join(HERE, "models", "yolov8n.pt"))
    imgs = sorted(glob.glob(os.path.join(HERE, "..", "..", "data", "otb", "*", "img", "0001.jpg")))
    imgs += sorted(glob.glob(os.path.join(HERE, "calib", "*.jpg")))[::20]
    for v in ("xnn32", "xnn8"):
        det = D.Detector.__new__(D.Detector)
        det.m, det.size, det.last_max = RTModel(os.path.join(HERE, "models", f"yolov8n_{v}_320.pte")), 320, 0.0
        det.anchor_pts, det.strides = D.anchors(320)
        det.bins = np.arange(16, dtype=np.float32)
        matched, top_same, n_ref, n_found = [], 0, 0, 0
        for f in imgs:
            im = cv2.imread(f)
            r = ref.predict(im, imgsz=320, conf=0.25, classes=[0], verbose=False)[0].boxes
            if not len(r):
                continue
            n_ref += 1
            refs = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in r.xyxy.tolist()]
            ours = det.detect(im, 0, 0.25)
            if ours:
                n_found += 1
                matched.append(max(iou(rb, ours[0][:4]) for rb in refs))
                top_same += iou(refs[0], ours[0][:4]) > 0.5
        print(f"{v}: {n_found}/{n_ref} images with a person found; top box vs best-matching ref box IoU "
              f"mean {np.mean(matched):.3f} min {np.min(matched):.3f}; same top person as ref {top_same}/{n_found}")
