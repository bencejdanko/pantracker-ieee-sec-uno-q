"""Board side of the parity check: run every .pte through libinfer on the host-generated inputs.

  python parity_run.py --inputs ~/tracker/parity/inputs --out ~/tracker/parity/out
Writes out/<pte-name>/<case>.npy for tracker/export/parity.py compare.
LightFC variants are run as the full per-block chain (backbone_z, backbone_x -> fusion -> head).
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from infer import Model  # noqa: E402

M = os.path.join(ROOT, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(ROOT, "parity", "inputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "parity", "out"))
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="portable", help="comma-separated substrings to skip (portable = 34 s/frame)")
    a = ap.parse_args()
    variants = sorted({os.path.basename(p)[len("backbone_x_"):-4] for p in glob.glob(os.path.join(M, "lightfc", "backbone_x_*.pte"))})
    for v in variants:
        if (a.only and a.only not in v) or any(k and k in v for k in a.skip.split(",")):
            continue
        L = lambda b: Model(os.path.join(M, "lightfc", f"{b}_{v}.pte"), threads=4)  # noqa: E731
        bz, bx, fu, hd = L("backbone_z"), L("backbone_x"), L("fusion"), L("head")
        od = os.path.join(a.out, f"lightfc_{v}"); os.makedirs(od, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(a.inputs, "lfc_*_z.npy"))):
            k = os.path.basename(f)[:6]
            zf = bz.run(np.load(f))[0].copy()
            xf = bx.run(np.load(f.replace("_z.npy", "_x.npy")))[0].copy()
            fo = fu.run(zf, xf)[0].copy()
            sc, sz, off = hd.run(fo)
            np.save(os.path.join(od, f"{k}_score.npy"), sc); np.save(os.path.join(od, f"{k}_size.npy"), sz)
            np.save(os.path.join(od, f"{k}_offset.npy"), off)
        print("done lightfc", v, flush=True)


if __name__ == "__main__":
    main()
