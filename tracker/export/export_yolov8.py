"""Export YOLOv8n (COCO-80) for ExecuTorch: yolov8n_{xnn32,xnn8,vk32,vk16}_{S}.pte

Outputs are the RAW head tensors, decoded on the device in numpy (python/detector.py):
  box  [1, 64, A]  DFL logits (4 sides x 16 bins), levels P3/P4/P5 concatenated
  cls  [1, 80, A]  class logits (sigmoid on the device)
Keeping DFL/anchors/sigmoid out of the graph is what makes int8 PTQ work: decoded pixel boxes (0..S) and
probabilities (0..1) never share one quantization scale (that broke the earlier YOLOE int8 export).
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import _NearestUp2, calib_images, lower, xnn_friendly  # noqa: E402

OUT = os.path.join(HERE, "models")


class YoloRaw(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.layers, self.save = det.model, det.save

    def forward(self, img):
        y, x = [], img
        n = len(self.layers)
        for i, m in enumerate(self.layers):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if i == n - 1:
                h = m
                box = torch.cat([h.cv2[k](x[k]).flatten(2) for k in range(h.nl)], 2)
                cls = torch.cat([h.cv3[k](x[k]).flatten(2) for k in range(h.nl)], 2)
                return box, cls
            x = m(x)
            y.append(x if m.i in self.save else None)


def load():
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv
    from ultralytics.utils.torch_utils import fuse_conv_and_bn
    cwd = os.getcwd(); os.chdir(OUT)
    m = YOLO("yolov8n.pt").model.eval()
    os.chdir(cwd)
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():  # fold Conv+BN
        if isinstance(mod, Conv) and hasattr(mod, "bn"):
            mod.conv = fuse_conv_and_bn(mod.conv, mod.bn)
            delattr(mod, "bn")
            mod.forward = mod.forward_fuse
    for i, layer in enumerate(m.model):  # nearest 2x upsample -> deconv (only active inside xnn_friendly)
        if isinstance(layer, nn.Upsample) and layer.mode == "nearest":
            up = _NearestUp2()
            for k in ("f", "i", "type", "np"):
                if hasattr(layer, k):
                    setattr(up, k, getattr(layer, k))
            m.model[i] = up
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="xnn32,xnn8,vk32,vk16")
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    model = YoloRaw(load()).eval()
    s = a.size
    ex = (torch.zeros(1, 3, s, s),)
    with torch.no_grad():
        b, c = model(*ex)
    print(f"YOLOv8n @{s}: box {tuple(b.shape)} cls {tuple(c.shape)}")
    calib = [(x,) for x in calib_images(s)]
    for v in a.variants.split(","):
        try:
            with xnn_friendly(on=v.startswith("xnn")):
                lower(model, ex, v, f"{OUT}/yolov8n_{v}_{s}.pte", calib=calib)
        except Exception as e:
            print(f"  {v}: FAILED {type(e).__name__}: {str(e)[:300]}")
