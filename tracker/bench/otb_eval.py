"""OTB-subset tracking harness, run ON the board.

  python otb_eval.py --data ~/tracker/data/otb --trackers lightfc:xnn32,lightfc:vk16,kcf,nano --out results/otb

Writes results/otb/<tracker>/<Seq>.txt (x,y,w,h per frame, OTB format) and <Seq>_time.txt (ms per frame).
Metrics are computed on the host by otb_metrics.py.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

SEQS = ["Basketball", "Biker", "BlurCar2", "David3", "Girl2", "Human3", "Jogging-1", "Walking2"]


def read_gt(path):
    rows = []
    for line in open(path):
        line = line.strip().replace("\t", ",").replace(" ", ",")
        if line:
            rows.append([float(v) for v in line.split(",") if v != ""][:4])
    return np.array(rows)


def seq_paths(data, name):
    base, idx = (name.split("-") + [None])[:2]
    d = os.path.join(data, base)
    gt = os.path.join(d, f"groundtruth_rect.{idx}.txt" if idx else "groundtruth_rect.txt")
    imgs = sorted(glob.glob(os.path.join(d, "img", "*.jpg")))
    return imgs, read_gt(gt)


class LightFCWrap:
    def __init__(self, placement, threads):
        from lightfc import LightFC
        self.t = LightFC(os.path.join(ROOT, "models", "lightfc"), placement, threads=threads)

    def init(self, bgr, box):
        self.t.initialize(bgr, list(box))

    def update(self, bgr):
        return self.t.track(bgr)[0]


class CVWrap:
    def __init__(self, kind):
        self.kind = kind

    def init(self, bgr, box):
        if self.kind == "kcf":
            self.t = cv2.TrackerKCF.create() if hasattr(cv2, "TrackerKCF") else cv2.TrackerKCF_create()
        else:
            p = cv2.TrackerNano_Params()
            nd = os.path.join(ROOT, "models", "nanotrack")
            p.backbone = os.path.join(nd, "nanotrack_backbone_sim.onnx")
            p.neckhead = os.path.join(nd, "nanotrack_head_sim.onnx")
            self.t = cv2.TrackerNano.create(p)
        self.box = [float(v) for v in box]
        self.t.init(bgr, tuple(int(round(v)) for v in box))

    def update(self, bgr):
        ok, b = self.t.update(bgr)
        if ok:
            self.box = [float(v) for v in b]
        return self.box


def make(spec, threads):
    if spec.startswith("lightfc:"):
        return LightFCWrap(spec.split(":", 1)[1], threads)
    return CVWrap(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "otb"))
    ap.add_argument("--trackers", default="lightfc:xnn32", help="';'-separated, e.g. 'lightfc:xnn32;kcf;nano'")
    ap.add_argument("--seqs", default=",".join(SEQS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "otb"))
    a = ap.parse_args()
    for spec in a.trackers.split(";"):
        tag = spec.replace(":", "_").replace(",", "+").replace("=", "")
        od = os.path.join(a.out, tag)
        os.makedirs(od, exist_ok=True)
        trk = make(spec, a.threads)
        for name in a.seqs.split(","):
            imgs, gt = seq_paths(a.data, name)
            n = min(len(imgs), len(gt))
            if a.max_frames:
                n = min(n, a.max_frames)
            boxes, times = [list(gt[0])], [0.0]
            trk.init(cv2.imread(imgs[0]), gt[0])
            for i in range(1, n):
                f = cv2.imread(imgs[i])
                t0 = time.perf_counter()
                b = trk.update(f)
                times.append((time.perf_counter() - t0) * 1e3)
                boxes.append([float(v) for v in b])
            np.savetxt(os.path.join(od, f"{name}.txt"), np.array(boxes), fmt="%.2f", delimiter=",")
            np.savetxt(os.path.join(od, f"{name}_time.txt"), np.array(times), fmt="%.3f")
            print(f"{tag:40s} {name:12s} {n:5d} frames  {np.median(times[1:]):7.2f} ms/frame p50", flush=True)


if __name__ == "__main__":
    main()
