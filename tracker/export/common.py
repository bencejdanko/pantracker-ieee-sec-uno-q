"""Shared ExecuTorch 1.5.1 lowering helpers for the export scripts."""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

# Adreno 702: maxImageDimension3D = 2048 (vulkaninfo on gatos).
VK_LIMITS = {"texture_limits_x": 2048, "texture_limits_y": 2048, "texture_limits_z": 2048}
VARIANTS = ("portable", "xnn32", "xnn8", "vk32", "vk16")


def quantize_pt2e(model, example, calib, only_modules=()):
    """Int8 PT2E PTQ, per-channel symmetric (XNNPACK quantizer). `only_modules`: quantize just these
    submodules (everything else stays fp32); empty = whole model."""
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    cfg = get_symmetric_quantization_config(is_per_channel=True)
    q = XNNPACKQuantizer()
    if only_modules:
        for name in only_modules:
            q.set_module_name(name, cfg)
    else:
        q.set_global(cfg)
    m = torch.export.export(model, example, strict=False).module()
    m = prepare_pt2e(m, q)
    with torch.no_grad():
        for batch in calib:
            m(*batch)
    return convert_pt2e(m)


def lower(model, example, variant, out_path, calib=None, only_modules=()):
    """Export `model` for one backend variant and write the .pte. Returns byte size."""
    model = model.eval()
    if variant == "xnn8":
        model = quantize_pt2e(model, example, calib or [example], only_modules)
    ep = torch.export.export(model, example, strict=False)
    if variant == "portable":
        parts = []
    elif variant in ("xnn32", "xnn8"):
        parts = [XnnpackPartitioner()]
    elif variant in ("vk32", "vk16"):
        parts = [VulkanPartitioner({**VK_LIMITS, "force_fp16": variant == "vk16"})]
    else:
        raise ValueError(variant)
    edge = to_edge_transform_and_lower(
        ep, partitioner=parts, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    prog = edge.to_executorch()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        prog.write_to_file(f)
    # Delegation summary: how many call_delegate vs. leftover portable ops.
    g = edge.exported_program().graph
    n_del = sum(1 for n in g.nodes if n.op == "call_function" and "call_delegate" in str(n.target))
    n_ops = sum(1 for n in g.nodes if n.op == "call_function" and "call_delegate" not in str(n.target)
                and "getitem" not in str(n.target))
    print(f"  {variant:8s} -> {out_path}  {os.path.getsize(out_path)/1e6:.2f} MB  delegates={n_del} cpu_ops={n_ops}")
    return os.path.getsize(out_path)


# ---- YOLO export helpers (shared by export_yolov8.py) ----

class _NearestUp2(nn.Module):
    """Exact 2x nearest upsample as a depthwise transposed conv with unit 2x2 kernels (XNNPACK-delegable).
    Only active inside xnn_friendly(); Vulkan takes the original op (and wants conv weights as constants)."""
    deconv = False

    def forward(self, x):
        if not _NearestUp2.deconv:
            return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        c = x.shape[1]
        w = torch.ones(c, 1, 2, 2, dtype=x.dtype)
        return torch.nn.functional.conv_transpose2d(x, w, stride=2, groups=c)


def _narrow_chunk(self, chunks, dim=0):
    n = self.shape[dim]
    step = -(-n // chunks)
    return tuple(self.narrow(dim, i, min(step, n - i)) for i in range(0, n, step))


def _narrow_split(self, sizes, dim=0):
    n = self.shape[dim]
    if isinstance(sizes, int):
        sizes = [sizes] * (-(-n // sizes))
    out, o = [], 0
    for sz in sizes:
        sz = min(sz, n - o)
        out.append(self.narrow(dim, o, sz))
        o += sz
    return tuple(out)


class xnn_friendly:
    """During export: chunk/split -> narrow (slice_copy, delegated) instead of split_with_sizes_copy, which
    XNNPACK rejects and which cut the graph into 17 partitions with layout conversions in between."""

    def __init__(self, on=True):
        self.on = on

    def __enter__(self):
        self._c, self._s = torch.Tensor.chunk, torch.Tensor.split
        if self.on:
            torch.Tensor.chunk, torch.Tensor.split = _narrow_chunk, _narrow_split
            _NearestUp2.deconv = True

    def __exit__(self, *a):
        torch.Tensor.chunk, torch.Tensor.split = self._c, self._s
        _NearestUp2.deconv = False


def letterbox(img, s):
    import cv2
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((s, s, 3), 114, np.uint8)
    top, left = (s - nh) // 2, (s - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def to_input(bgr, s):
    x = letterbox(bgr, s)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(x))


def calib_images(s, n=200):
    import cv2
    here = os.path.dirname(os.path.abspath(__file__))
    # demo-camera frames are near-identical (one room), so mix in OTB frames from varied scenes for range coverage
    cam = sorted(glob.glob(os.path.join(here, "calib", "*.jpg")))[::4]
    otb = sorted(glob.glob(os.path.join(here, "..", "..", "data", "otb", "*", "img", "*.jpg")))
    files = (cam + otb[::max(1, len(otb) // (n - len(cam)))])[:n]
    if not files:
        import ultralytics
        files = glob.glob(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "*.jpg"))
    return [to_input(cv2.imread(f), s) for f in files]


