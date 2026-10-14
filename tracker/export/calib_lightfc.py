"""LightFC int8 PTQ calibration pairs: template/search crops from OTB sequences that are NOT in
the evaluation subset (Car4, Deer, Dog1, Coke, Crossing, Bolt), cropped exactly like the tracker does
(python/lightfc.py sample_target + ImageNet normalization). -> calib/lightfc/*.npz
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from lightfc import sample_target  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "otb")
SEQS = ["Car4", "Deer", "Dog1", "Coke", "Crossing", "Bolt"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def prep(p):
    return ((p.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None].copy()


def read_gt(p):
    return np.array([[float(v) for v in l.replace("\t", ",").replace(" ", ",").split(",") if v][:4]
                     for l in open(p) if l.strip()])


if __name__ == "__main__":
    out = os.path.join(HERE, "calib", "lightfc")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    k = 0
    for s in SEQS:
        imgs = sorted(glob.glob(os.path.join(DATA, s, "img", "*.jpg")))
        gt = read_gt(os.path.join(DATA, s, "groundtruth_rect.txt"))
        n = min(len(imgs), len(gt))
        z_rgb = cv2.imread(imgs[0])[:, :, ::-1]
        z, _ = sample_target(z_rgb, list(gt[0]), 2.0, 128)
        for i in np.linspace(1, n - 1, 11).astype(int):
            rgb = cv2.imread(imgs[i])[:, :, ::-1]
            prev = gt[max(0, i - 1)] + rng.normal(0, 3, 4) * [1, 1, 0, 0]  # small jitter like a real track
            x, _ = sample_target(rgb, list(prev), 4.0, 256)
            np.savez(os.path.join(out, f"{s}_{i:04d}.npz"), z=prep(z), x=prep(x))
            k += 1
    print("calibration pairs:", k)
