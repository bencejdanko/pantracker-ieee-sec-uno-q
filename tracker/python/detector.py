"""YOLOv8n (COCO-80) detector: raw head outputs from libinfer, DFL decode + NMS in numpy.

The .pte returns box [1,64,A] (DFL logits, 4 sides x 16 bins) and cls [1,80,A] (logits), levels P3/P4/P5
(strides 8/16/32) concatenated, see tracker/export/export_yolov8.py.
"""
import difflib

import cv2
import numpy as np

from infer import Model

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]
ALIASES = {"people": "person", "man": "person", "woman": "person", "me": "person", "phone": "cell phone",
           "mug": "cup", "ball": "sports ball", "plant": "potted plant", "sofa": "couch", "table": "dining table",
           "television": "tv", "monitor": "tv", "bike": "bicycle", "motorbike": "motorcycle"}


class Classes:
    """Prompt text -> COCO class id."""

    def __init__(self):
        self.words = sorted(set(COCO) | set(ALIASES))

    def lookup(self, prompt):
        """-> (class name, class id) or (None, suggestions)."""
        s = " ".join((prompt or "").lower().replace("_", " ").split())
        for c in (s, s[:-1] if s.endswith("s") else None, s[:-2] if s.endswith("es") else None):
            if c:
                c = ALIASES.get(c, c)
                if c in COCO:
                    return c, COCO.index(c)
        return None, difflib.get_close_matches(s, self.words, n=3, cutoff=0.5)


def letterbox(img, s):
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((s, s, 3), 114, np.uint8)
    top, left = (s - nh) // 2, (s - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def anchors(size, strides=(8, 16, 32)):
    pts, st = [], []
    for s in strides:
        n = size // s
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
        pts.append(np.stack([xs.ravel(), ys.ravel()], 1))
        st.append(np.full(n * n, s, np.float32))
    return np.concatenate(pts), np.concatenate(st)


def nms(boxes, scores, iou=0.5):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        o = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][o < iou]
    return keep


class Detector:
    def __init__(self, pte, threads=0, size=320):
        self.m = Model(pte, threads=threads)
        self.size = size
        self.anchor_pts, self.strides = anchors(size)
        self.bins = np.arange(16, dtype=np.float32)
        self.last_max = 0.0  # best score for the requested class on the last frame, even below conf (UI hint)

    def raw(self, bgr):
        img, r, left, top = letterbox(bgr, self.size)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) * (1 / 255.0)
        box, cls = self.m.run(x)
        return box[0], cls[0], (r, left, top)

    def detect(self, bgr, cls_id, conf=0.25):
        """-> list of (x, y, w, h, score) in frame pixels for COCO class cls_id, best first."""
        box, cls, (r, left, top) = self.raw(bgr)
        self.last_max = float(1 / (1 + np.exp(-cls[cls_id].max())))
        logit_thr = np.log(conf / (1 - conf))
        idx = np.nonzero(cls[cls_id] > logit_thr)[0]  # filter on logits before any decode work
        if idx.size == 0:
            return []
        sc = 1 / (1 + np.exp(-cls[cls_id, idx]))
        d = box[:, idx].reshape(4, 16, -1)
        d = np.exp(d - d.max(1, keepdims=True))
        dist = (d * self.bins[None, :, None]).sum(1) / d.sum(1)  # (4, n): left, top, right, bottom in grid units
        a, s = self.anchor_pts[idx], self.strides[idx]
        b = np.stack([a[:, 0] - dist[0], a[:, 1] - dist[1], a[:, 0] + dist[2], a[:, 1] + dist[3]], 1) * s[:, None]
        res = []
        for i in nms(b, sc)[:10]:
            x1, y1, x2, y2 = b[i]
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            res.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(sc[i])))
        return res
