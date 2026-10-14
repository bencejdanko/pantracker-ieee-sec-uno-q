"""LightFC tracker driven by per-block .pte files with a CPU/GPU placement string.

Mirrors LightFC/repo/lib/test/tracker/lightfc.py (sample_target crop, Hanning window, cal_bbox,
map_box_back, clip_box) in numpy. Frames are BGR (OpenCV); the RGB swap the model expects (upstream loads RGB)
happens inside the normalization blob, so no full-frame channel-reversed copy is ever made (~9 ms at 640x480).
"""
import json
import math
import os
import time

import cv2
import numpy as np

from infer import Model

BLOCKS = ("x", "fusion", "head")  # backbone_z follows the backbone_x placement


def parse_placement(s):
    """'x=vk16,fusion=xnn32,head=xnn32' or a single variant 'xnn32' -> dict."""
    if "=" not in s:
        return {b: s for b in BLOCKS}
    d = dict(kv.split("=") for kv in s.split(","))
    return {b: d.get(b, "xnn32") for b in BLOCKS}


def hann1d(sz):
    return 0.5 * (1 - np.cos((2 * math.pi / (sz + 1)) * np.arange(1, sz + 1, dtype=np.float32)))


def sample_target(im, bb, factor, out_sz):
    x, y, w, h = bb
    crop_sz = math.ceil(math.sqrt(w * h) * factor)
    if crop_sz < 1:
        raise ValueError("too small bounding box")
    x1 = round(x + 0.5 * w - crop_sz * 0.5); x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5); y2 = y1 + crop_sz
    x1p, x2p = max(0, -x1), max(x2 - im.shape[1] + 1, 0)
    y1p, y2p = max(0, -y1), max(y2 - im.shape[0] + 1, 0)
    crop = im[y1 + y1p:y2 - y2p, x1 + x1p:x2 - x2p, :]
    crop = cv2.copyMakeBorder(crop, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT)
    return cv2.resize(crop, (out_sz, out_sz)), out_sz / crop_sz


def warp_target(im, bb, factor, out_sz):
    """Same crop as sample_target (square, factor x sqrt(area), zero padding outside the image, bilinear resize to
    out_sz), as one cv2.warpAffine whose cost depends on out_sz only. sample_target pads the whole crop first, which
    costs 2.6 ms for an 80 px box and 8 ms for a 350x450 box on the A53; this is 0.4-0.9 ms. Matches
    cv2.resize's pixel-centre convention; differs only on the last image row/column, which upstream drops
    (off-by-one in its padding)."""
    x, y, w, h = bb
    crop_sz = math.ceil(math.sqrt(w * h) * factor)
    if crop_sz < 1:
        raise ValueError("too small bounding box")
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    s = out_sz / crop_sz
    M = np.array([[s, 0, -(x1 - 0.5) * s - 0.5], [0, s, -(y1 - 0.5) * s - 0.5]], np.float32)
    return cv2.warpAffine(im, M, (out_sz, out_sz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), s


def clip_box(box, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin); x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin); y2 = min(max(margin, y2), H)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


class LightFC:
    def __init__(self, model_dir, placement="xnn32", threads=0, suffix=""):
        self.dir = model_dir
        meta = json.load(open(os.path.join(model_dir, "blocks.json")))
        p = meta["preprocess"]
        self.mean = np.array(p["mean"], np.float32) * 255.0
        self.istd = 1.0 / (np.array(p["std"], np.float32) * 255.0)
        self.z_sz, self.z_fac = p["template"]
        self.x_sz, self.x_fac = p["search"]
        self.feat_sz = p["feat_sz"]
        self.window = (hann1d(self.feat_sz)[:, None] * hann1d(self.feat_sz)[None, :]).astype(np.float32)
        self.threads, self.suffix = threads, suffix
        self.models = {}
        self.state = self.z_feat = self._z_patch = None
        # (bgr - mean) * istd with the BGR->RGB swap, NCHW float32, in one OpenCV call (1.7 ms vs 5.2 ms in numpy)
        self.blob = cv2.dnn.Image2BlobParams()
        self.blob.scalefactor = tuple(self.istd.tolist()) + (0.0,)
        self.blob.mean = tuple(self.mean.tolist()) + (0.0,)
        self.blob.swapRB = True
        self.blob.ddepth = cv2.CV_32F
        self.set_placement(placement)
        self.t = {}

    def _load(self, blk, v):
        key = (blk, v)
        if key not in self.models:
            self.models[key] = Model(os.path.join(self.dir, f"{blk}_{v}{self.suffix}.pte"), threads=self.threads)
        return self.models[key]

    def set_placement(self, placement):
        pl = parse_placement(placement)
        bz = self._load("backbone_z", pl["x"])
        bx = self._load("backbone_x", pl["x"])
        fu = self._load("fusion", pl["fusion"])
        hd = self._load("head", pl["head"])
        self.pl, self.placement = pl, ",".join(f"{k}={v}" for k, v in pl.items())
        self.bz, self.bx, self.fu, self.hd = bz, bx, fu, hd
        if self._z_patch is not None:
            self.z_feat = self.bz.run(self._z_patch)[0].copy()

    def _prep(self, patch_bgr):
        return cv2.dnn.blobFromImageWithParams(patch_bgr, self.blob)

    def initialize(self, bgr, box):
        patch, _ = warp_target(bgr, box, self.z_fac, self.z_sz)
        self._z_patch = self._prep(patch)
        self.z_feat = self.bz.run(self._z_patch)[0].copy()
        self.state = list(map(float, box))

    def track(self, bgr):
        """BGR frame -> (box [x,y,w,h], score). Per-stage ms in self.t."""
        H, W = bgr.shape[:2]
        t0 = time.perf_counter()
        patch, rf = warp_target(bgr, self.state, self.x_fac, self.x_sz)
        x = self._prep(patch)
        t1 = time.perf_counter()
        xf = self.bx.run(x)[0]
        t2 = time.perf_counter()
        fo = self.fu.run(self.z_feat, xf)[0]
        t3 = time.perf_counter()
        score, size, off = self.hd.run(fo)
        t4 = time.perf_counter()
        resp = self.window * score[0, 0]
        idx = int(resp.argmax())
        iy, ix = divmod(idx, self.feat_sz)
        cx = (ix + float(off[0, 0, iy, ix])) / self.feat_sz
        cy = (iy + float(off[0, 1, iy, ix])) / self.feat_sz
        w, h = float(size[0, 0, iy, ix]), float(size[0, 1, iy, ix])
        s = self.x_sz / rf
        cx, cy, w, h = cx * s, cy * s, w * s, h * s
        px, py = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        half = 0.5 * s
        cxr, cyr = cx + (px - half), cy + (py - half)
        self.state = clip_box([cxr - 0.5 * w, cyr - 0.5 * h, w, h], H, W, margin=2)
        t5 = time.perf_counter()
        self.t = {"pre": (t1 - t0) * 1e3, "backbone_x": (t2 - t1) * 1e3, "fusion": (t3 - t2) * 1e3,
                  "head": (t4 - t3) * 1e3, "post": (t5 - t4) * 1e3}
        return list(self.state), float(score[0, 0, iy, ix])
