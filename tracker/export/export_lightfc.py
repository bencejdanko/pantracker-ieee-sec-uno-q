"""Export LightFC (MobileNetV2 variant) per block for ExecuTorch.

Blocks: backbone_z [1,3,128,128]->[1,96,8,8], backbone_x [1,3,256,256]->[1,96,16,16],
        fusion (z_feat, x_feat)->[1,192,16,16], head -> (score[1,1,16,16], size[1,2,16,16], offset[1,2,16,16]).
Files: models/lightfc/{block}_{variant}[_norep].pte and models/lightfc/blocks.json.
`_norep` = train-time graph (repN33 two-branch + unfused BN); default = switch_to_deploy() + BN folded.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "LightFC")
sys.path.insert(0, os.path.join(REPO, "repo"))
sys.path.insert(0, HERE)
from common import lower  # noqa: E402

CKPT = os.path.join(REPO, "extracted/lightfc/output/checkpoints/train/lightfc/"
                    "mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou/lightfc_ep0400.pth.tar")
OUT = os.path.join(HERE, "models", "lightfc")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def upstream_modules():
    """Import only the three model files. The upstream package __init__s pull in the training stack
    (tensorboard, timm, ...), so register bare namespace packages instead of executing them."""
    import importlib
    import types
    root = os.path.join(REPO, "repo")
    for pkg in ("lib", "lib.models", "lib.models.lightfc", "lib.models.lightfc.backbone",
                "lib.models.lightfc.fusion", "lib.models.lightfc.head"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [os.path.join(root, *pkg.split("."))]
            sys.modules[pkg] = m
    bb = importlib.import_module("lib.models.lightfc.backbone.mobilnetv2")
    fu = importlib.import_module("lib.models.lightfc.fusion.ecm")
    hd = importlib.import_module("lib.models.lightfc.head.erh")
    return bb.MobileNetV2, fu.pwcorr_se_scf_sc_iab_sc_concat, hd.repn33_se_center_concat


class LightFCNet(nn.Module):
    """Same submodules/state-dict keys as lib.models.LightFC, without its constructor's
    `if self.train:` bug (bound method => always truthy => tries to load mobilenet_v2 pretrain)."""

    def __init__(self):
        super().__init__()
        MobileNetV2, pwcorr_se_scf_sc_iab_sc_concat, repn33_se_center_concat = upstream_modules()
        self.backbone = MobileNetV2()
        self.fusion = pwcorr_se_scf_sc_iab_sc_concat(num_kernel=64, adj_channel=96)
        self.head = repn33_se_center_concat(inplanes=192, channel=256, feat_sz=16, stride=16, freeze_bn=False)


CLEAN = os.path.join(OUT, "lightfc_net_sd.pt")


def load_state_dict():
    """The authors' checkpoint also pickles training objects from their `lib` package, so
    weights_only=True refuses it. Unpickle with an allow-list (torch/collections/numpy only; any other
    class becomes an inert stub), keep only 'net', and cache it as a plain weights-only file."""
    if os.path.exists(CLEAN):
        return torch.load(CLEAN, map_location="cpu", weights_only=True)
    import pickle
    import types

    class Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("torch", "collections", "numpy", "_codecs", "builtins"):
                if module == "builtins" and name not in ("set", "frozenset", "dict", "list", "tuple", "slice"):
                    return Stub
                return super().find_class(module, name)
            return Stub

    pm = types.SimpleNamespace(Unpickler=Restricted, load=pickle.load, __name__="restricted_pickle")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, pickle_module=pm)["net"]
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    os.makedirs(OUT, exist_ok=True)
    torch.save(sd, CLEAN)
    return sd


def load_net(reparam=True):
    net = LightFCNet()
    net.load_state_dict(load_state_dict(), strict=True)
    net.eval()
    if reparam:
        for m in net.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        fold_bn(net)
    return net


def fold_bn(root):
    for mod in list(root.modules()):
        if isinstance(mod, nn.Sequential):
            for i in range(len(mod) - 1):
                if isinstance(mod[i], nn.Conv2d) and isinstance(mod[i + 1], nn.BatchNorm2d):
                    mod[i] = fuse_conv_bn_eval(mod[i], mod[i + 1])
                    mod[i + 1] = nn.Identity()
    f = getattr(root, "fusion", None)
    if f is not None:
        for c, b in (("conv11", "bn11"), ("conv33", "bn33"), ("conv_up", "bn_up"), ("conv_down", "bn_down")):
            setattr(f, c, fuse_conv_bn_eval(getattr(f, c), getattr(f, b)))
            setattr(f, b, nn.Identity())


class Backbone(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.b = net.backbone

    def forward(self, x):
        return self.b(x)


class Fusion(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.f = net.fusion

    def forward(self, z, x):
        return self.f(z, x)


class Head(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.h = net.head

    def forward(self, x):
        s, sz, off = self.h.get_score_map(x)
        return s, sz, off


def preprocess(rgb_patch):
    x = (rgb_patch.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))


def calib_pairs(n=64):
    """(template, search) crop pairs saved by otb_crops (bench/otb_eval.py --dump-calib)."""
    files = sorted(glob.glob(os.path.join(HERE, "calib", "lightfc", "*.npz")))[:n]
    return [(torch.from_numpy(d["z"]), torch.from_numpy(d["x"])) for d in (np.load(f) for f in files)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="portable,xnn32,vk32,vk16")
    ap.add_argument("--norep", default="portable,xnn32", help="variants also exported without re-param")
    ap.add_argument("--blocks", default="backbone_z,backbone_x,fusion,head")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    z = torch.randn(1, 3, 128, 128)
    x = torch.randn(1, 3, 256, 256)
    calib = calib_pairs()
    meta = {}
    for rep in (True, False):
        net = load_net(reparam=rep)
        with torch.no_grad():
            zf = net.backbone(z)
            xf = net.backbone(x)
            fo = net.fusion(zf, xf)
            ho = Head(net)(fo)
        ex = {"backbone_z": (Backbone(net), (z,)), "backbone_x": (Backbone(net), (x,)),
              "fusion": (Fusion(net), (zf, xf)), "head": (Head(net), (fo,))}
        if rep:
            meta = {k: {"inputs": [list(t.shape) for t in v[1]]} for k, v in ex.items()}
            meta["backbone_z"]["outputs"] = [list(zf.shape)]
            meta["backbone_x"]["outputs"] = [list(xf.shape)]
            meta["fusion"]["outputs"] = [list(fo.shape)]
            meta["head"]["outputs"] = [list(t.shape) for t in ho]
        # per-block calibration inputs (computed through the fp32 chain)
        cal = {"backbone_z": [], "backbone_x": [], "fusion": [], "head": []}
        with torch.no_grad():
            for zc, xc in calib:
                a_, b_ = net.backbone(zc), net.backbone(xc)
                c_ = net.fusion(a_, b_)
                cal["backbone_z"].append((zc,)); cal["backbone_x"].append((xc,))
                cal["fusion"].append((a_, b_)); cal["head"].append((c_,))
        variants = a.variants.split(",") if rep else a.norep.split(",")
        for blk in a.blocks.split(","):
            mod, inp = ex[blk]
            for v in variants:
                if not v:
                    continue
                name = f"{blk}_{v}{'' if rep else '_norep'}.pte"
                try:
                    lower(mod, inp, v, os.path.join(OUT, name), calib=cal[blk] or [inp])
                except Exception as e:
                    print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:300]}")
    meta["preprocess"] = {"mean": MEAN.tolist(), "std": STD.tolist(), "template": [128, 2.0], "search": [256, 4.0],
                          "feat_sz": 16, "stride": 16, "color": "RGB"}
    json.dump(meta, open(os.path.join(OUT, "blocks.json"), "w"), indent=1)
    print(json.dumps(meta))
