"""Parity of every exported LightFC .pte chain against PyTorch. YOLOv8n: parity_yolov8.py.

  1) host:   python parity.py refs            -> parity/inputs/*.npy + parity/ref/*.npy (PyTorch, ORIGINAL modules)
  2) board:  python bench/parity_run.py       -> parity/out/<pte>/*.npy (libinfer, same inputs)
  3) host:   python parity.py compare         -> parity/parity.csv (max abs diff, box IoU / score-peak agreement)

LightFC refs come from the upstream full model
(backbone -> fusion -> head on the same crops), so per-block chaining is covered.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P = os.path.join(HERE, "parity")
N = 20


def lightfc_refs():
    import export_lightfc as el
    net = el.LightFCNet()
    net.load_state_dict(el.load_state_dict(), strict=True)
    net.eval()
    for m in net.modules():  # upstream test-time path: switch_to_deploy on backbone+head, BN kept as modules
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
    files = sorted(glob.glob(os.path.join(HERE, "calib", "lightfc", "*.npz")))[::3][:N]
    for i, f in enumerate(files):
        d = np.load(f)
        z, x = torch.from_numpy(d["z"]), torch.from_numpy(d["x"])
        with torch.no_grad():
            zf = net.backbone(z)
            out = net.head(net.fusion(zf, net.backbone(x)))
        np.save(f"{P}/inputs/lfc_{i:02d}_z.npy", d["z"])
        np.save(f"{P}/inputs/lfc_{i:02d}_x.npy", d["x"])
        np.save(f"{P}/ref/lfc_{i:02d}_score.npy", out["score_map"].numpy())
        np.save(f"{P}/ref/lfc_{i:02d}_size.npy", out["size_map"].numpy())
        np.save(f"{P}/ref/lfc_{i:02d}_offset.npy", out["offset_map"].numpy())


def compare():
    rows = []
    for od in sorted(glob.glob(f"{P}/out/*")):
        name = os.path.basename(od)
        if name.startswith("lightfc_"):
            d_s, d_sz, d_off, agree = [], [], [], []
            for r in sorted(glob.glob(f"{P}/ref/lfc_*_score.npy")):
                k = os.path.basename(r)[:6]
                if not os.path.exists(os.path.join(od, f"{k}_score.npy")):
                    continue
                rs, os_ = np.load(r), np.load(os.path.join(od, f"{k}_score.npy"))
                d_s.append(np.abs(rs - os_).max())
                d_sz.append(np.abs(np.load(f"{P}/ref/{k}_size.npy") - np.load(os.path.join(od, f"{k}_size.npy"))).max())
                d_off.append(np.abs(np.load(f"{P}/ref/{k}_offset.npy") - np.load(os.path.join(od, f"{k}_offset.npy"))).max())
                agree.append(int(rs.argmax() == os_.argmax()))
            rows.append({"pte": name, "n": len(d_s), "max_abs_diff_score": round(float(np.max(d_s)), 5),
                         "max_abs_diff_size": round(float(np.max(d_sz)), 5),
                         "max_abs_diff_offset": round(float(np.max(d_off)), 5),
                         "peak_agreement": round(float(np.mean(agree)), 3)})
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f"{P}/parity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    for d in ("inputs", "ref", "out"):
        os.makedirs(f"{P}/{d}", exist_ok=True)
    if sys.argv[1] == "refs":
        lightfc_refs()
    else:
        compare()
