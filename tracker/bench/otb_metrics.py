"""OTB metrics on the host: success AUC (IoU thresholds 0..1, 21 steps) and precision @ 20 px,
plus the port check (mean IoU of our trajectories vs the LightFC authors' released raw results).

  python otb_metrics.py --data ../../data/otb --results results/otb --authors <.../otb> --out results/otb_metrics.csv
"""
import argparse
import csv
import glob
import os

import numpy as np

from otb_eval import SEQS, read_gt


def iou(a, b):
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def center_err(a, b):
    ca = a[:, :2] + a[:, 2:4] / 2
    cb = b[:, :2] + b[:, 2:4] / 2
    return np.linalg.norm(ca - cb, axis=1)


def gt_path(data, name):
    base, idx = (name.split("-") + [None])[:2]
    return os.path.join(data, base, f"groundtruth_rect.{idx}.txt" if idx else "groundtruth_rect.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--authors", default=None, help="dir with the authors' <Seq>.txt raw results")
    ap.add_argument("--out", default="results/otb_metrics.csv")
    ap.add_argument("--max-frames", type=int, default=0, help="truncate every trajectory (apples-to-apples drift)")
    a = ap.parse_args()
    thr = np.linspace(0, 1, 21)
    rows = []
    for td in sorted(glob.glob(os.path.join(a.results, "*"))):
        tag = os.path.basename(td)
        all_iou, all_ce, port, times = [], [], [], []
        per_seq = {}
        for name in SEQS:
            rp = os.path.join(td, f"{name}.txt")
            if not os.path.exists(rp):
                continue
            res = np.loadtxt(rp, delimiter=",").reshape(-1, 4)
            gt = read_gt(gt_path(a.data, name))
            n = min(len(res), len(gt))
            if a.max_frames:
                n = min(n, a.max_frames)
            valid = (gt[:n, 2] > 0) & (gt[:n, 3] > 0)
            io = iou(res[:n], gt[:n])[valid]
            ce = center_err(res[:n], gt[:n])[valid]
            all_iou.append(io); all_ce.append(ce)
            tp = os.path.join(td, f"{name}_time.txt")
            if os.path.exists(tp):
                times.append(np.loadtxt(tp)[1:])
            succ = np.mean([np.mean(io > t) for t in thr])
            per_seq[name] = succ
            if a.authors:
                ap_ = os.path.join(a.authors, f"{name.replace('-', '_')}.txt")
                if os.path.exists(ap_):
                    au = read_gt(ap_)
                    m = min(len(au), n)
                    port.append((name, float(np.mean(iou(res[:m], au[:m])))))
        if not all_iou:
            continue
        io, ce = np.concatenate(all_iou), np.concatenate(all_ce)
        auc = float(np.mean([np.mean(io > t) for t in thr]))
        p20 = float(np.mean(ce <= 20))
        tm = np.concatenate(times) if times else np.array([0.0])
        row = {"tracker": tag, "seqs": len(all_iou), "frames": len(io), "success_auc": round(auc, 4),
               "precision_20px": round(p20, 4), "ms_per_frame_p50": round(float(np.median(tm)), 2),
               "ms_per_frame_p95": round(float(np.percentile(tm, 95)), 2)}
        if port:
            row["port_mean_iou_vs_authors"] = round(float(np.mean([p for _, p in port])), 4)
            row["port_per_seq"] = " ".join(f"{n}:{p:.3f}" for n, p in port)
        row["auc_per_seq"] = " ".join(f"{n}:{v:.3f}" for n, v in per_seq.items())
        rows.append(row)
        print(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
