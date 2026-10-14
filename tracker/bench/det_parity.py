"""Board-side YOLOv8n check: GPU (vk32/vk16) and int8 detections vs CPU fp32 on the first frame of each OTB sequence.
  python det_parity.py
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from detector import Detector  # noqa: E402


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih + 1e-9)


imgs = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "otb", "*", "img", "0001.jpg")))]
M = os.path.join(ROOT, "models")
ref = Detector(os.path.join(M, "yolov8n_xnn32_320.pte"), threads=4)
refs = [ref.detect(im, 0, 0.25) for im in imgs]
for v in ("xnn8", "vk32", "vk16"):
    d = Detector(os.path.join(M, f"yolov8n_{v}_320.pte"), threads=4)
    m = []
    for im, r in zip(imgs, refs):
        o = d.detect(im, 0, 0.25)
        if r and o:
            m.append(max(iou(rb[:4], o[0][:4]) for rb in r))
    print(f"{v}: {len(m)}/{sum(1 for r in refs if r)} found; matched-box IoU vs xnn32 mean {np.mean(m):.3f} min {np.min(m):.3f}")
