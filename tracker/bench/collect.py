"""Host side: pull results from the board and render the Efficiency Scorecard as markdown.

  python collect.py --pull           # scp ~/tracker/bench/results + ~/tracker/parity/out from gatos
  python collect.py                  # -> bench/SCORECARD.md (committed; raw CSVs stay in results/, gitignored)
"""
import argparse
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
REPO = os.path.dirname(os.path.dirname(HERE))


def read(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def f(v, d=1):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return v or "–"


def g(v):
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return "–"


def pull():
    pw = os.path.join(REPO, ".ssh_pass")
    ssh = f"sshpass -f {pw} ssh arduino@gatos.local"
    os.makedirs(R, exist_ok=True)
    subprocess.run(f"{ssh} 'tar -C ~/tracker/bench -cf - results' | tar -C {HERE} -xf -", shell=True, check=True)
    pd = os.path.join(REPO, "tracker", "export", "parity")
    subprocess.run(f"{ssh} 'tar -C ~/tracker/parity -cf - out' | tar -C {pd} -xf -", shell=True, check=False)


def merged(pattern_prefix, blocks=False):
    """scorecard*.csv (main run + supplementary re-runs); later files override earlier rows with the same key."""
    import glob
    files = sorted(glob.glob(os.path.join(R, pattern_prefix + "*.csv")), key=os.path.getmtime)
    files = [p for p in files if p.endswith("_blocks.csv") == blocks]
    out = {}
    for p in files:
        for r in read(p):
            if not blocks and r.get("status") != "ok" and r["row"] in out:
                continue  # don't let a failed re-run hide a good row
            out[(r["row"], r.get("block")) if blocks else r["row"]] = r
    return list(out.values())


def scorecard(rows):
    out = ["| Row | Config | .pte MB | peak RSS MB | p50 ms | p95 ms | FPS | Mcycles/f | IPC | L2 refill MB/f | bus acc. M/f | GPU busy ms/f | backbone_z ms |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") != "ok":
            out.append(f"| {r['row']} | {r['desc']} | {r.get('status', '')} |" + " |" * 11)
            continue
        out.append(
            f"| {r['row']} | {r['desc']} | {f(r['size_mb'], 2)} | {f(r['peak_rss_mb'], 0)} | {f(r['p50_ms'])} | {f(r['p95_ms'])} | "
            f"{f(r['fps_from_p50'])} | {f(float(r['cycles_per_frame']) / 1e6, 1)} | {f(r['ipc'], 2)} | "
            f"{f(r['l2_refill_mb_per_frame'], 2)} | {f(float(r['bus_access_per_frame']) / 1e6, 2)} | "
            f"{f(r['gpu_busy_ms_per_frame'], 1)} | {f(r.get('backbone_z_p50_ms'), 1)} |"
           )
    return "\n".join(out)


def table(rows, cols):
    if not rows:
        return "_not run_"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args()
    if a.pull:
        pull()
    md = ["# Efficiency Scorecard (measured on gatos)", ""]
    cond = os.path.join(R, "conditions.txt")
    if os.path.exists(cond):
        md += ["Conditions: `" + " · ".join(open(cond).read().split("\n")[:4]) + "`", ""]
    md += ["## Per-model rows (infer_bench, 4 threads; LightFC = backbone_x + fusion + head)", "",
           scorecard(sorted([r for r in merged("scorecard") if r["row"][0] in "LPV"], key=lambda r: ("LPV".index(r["row"][0]), r["row"]))), "",
           "Counters are user-space A53 PMU events (perf_event_paranoid=2), per frame = (N-run − 0-run)/N. "
           "L2 refill MB = refills × 64 B (lower bound on DRAM traffic). GPU busy from DRM fdinfo `drm-engine-gpu`.", ""]
    blocks = [b for b in merged("scorecard", blocks=True) if b["row"][0] in "LPV"]
    if blocks:
        md += ["## Per-block latency", "", table(
            [{"row": b["row"], "block": b["block"], "p50 ms": f(b["p50_ms"], 2), "p95 ms": f(b["p95_ms"], 2),
              "MB": f(b["size_mb"], 2), "Mcyc/f": f(float(b["cycles_per_frame"]) / 1e6, 1),
              "GPU ms/f": f(b["gpu_busy_ms_per_frame"], 1)} for b in blocks],
            ["row", "block", "p50 ms", "p95 ms", "MB", "Mcyc/f", "GPU ms/f"]), ""]
    for name, title in (("otb_metrics.csv", "Tracking accuracy — full sequences"),
                        ("otb300_metrics.csv", "Tracking accuracy — first 300 frames (precision drift)")):
        md += [f"## {title}", "", table(read(os.path.join(R, name)),
                                        ["tracker", "seqs", "frames", "success_auc", "precision_20px", "ms_per_frame_p50",
                                         "port_mean_iou_vs_authors"]), ""]
    md += ["## Parity vs PyTorch", "", table([r for r in read(os.path.join(REPO, "tracker", "export", "parity", "parity.csv")) if r["pte"].startswith("lightfc")],
                                             ["pte", "n", "max_abs_diff_score", "max_abs_diff_size", "max_abs_diff_offset", "peak_agreement"]), ""]
    md += ["## YOLOv8n parity (top-box IoU)", "",
           "Host vs Ultralytics (`export/parity_yolov8.py`): fp32 0.994; int8 finds every person, matched IoU 0.94–0.98. "
           "Board vs CPU fp32 (`bench/det_parity.py`, first frame of the 8 OTB test sequences): int8 0.972, "
           "Vulkan fp32 1.000, Vulkan fp16 0.999. int8 scores saturate near 0.80.", "",
           "## Live system (2026-10-13, read from `/state`, last 60 samples, one person walking, 4 XNNPACK threads)", "",
           "YOLOv8n int8 (SEARCH) + YOLOv8n Vulkan fp16 (TRACK check) + LightFC int8. TRACK 14.4 FPS (capture 15.0 FPS) · "
           "capture → motor command p50 142 / p95 183 ms · tracker step 67 / 77 ms · SEARCH detector step 131 / 163 ms · "
           "GPU check 648 / 661 ms · RPC 6.3 / 9.1 ms · RSS 200 MB · process CPU ~317 % · GPU busy ~60 % · 1,451 frames, "
           "0 exceptions, 0 RPC errors, 73 checks (17 stale and ignored). With 3 threads (same day): tracker step 70 / 76 ms, "
           "detector step 139 / 177 ms, capture → motor 137 / 172 ms, process CPU ~295 %.", ""]
    md += ["## Method notes", "",
           "- PMU counters are user-space only (`perf_event_paranoid=2`, no sudo in the session); GPU-driver/kernel time is not in cycles/frame.",
           "- The `performance`-governor column was not run (needs interactive sudo). All rows: `schedutil`.",
           "- Runs: LightFC 20 warm-up / 200 measured; YOLOv8n 5 / 50; portable (L0) 2 / 10 (34.7 s per frame).",
           "- The cpuss < 50 °C wait is capped at 3 min per row (idle cpuss on this board is ~52–57 °C in this room).",
           "- `portable` rows = no delegate; every op runs on ExecuTorch's CPU kernel library linked into libinfer "
           "(portable kernels, plus optimized kernels where they exist; convolutions are portable).",
           "- Detector: YOLOv8n COCO, raw head outputs (DFL/anchors/sigmoid decoded in numpy), so int8 PTQ keeps boxes intact.",
           "- Calibration: LightFC int8 = 64 crop pairs (first 64 of 66 generated) from OTB Bolt, Car4, Coke, Crossing, Deer, Dog1 (not in the test set); "
           "YOLOv8n int8 = 200 images (50 demo-camera frames + 150 frames from 14 OTB sequences).",
           "- **Vulkan fp16:** ExecuTorch 1.5.1 queried 16-bit features only through `vkGetPhysicalDeviceFeatures2KHR`, which is "
           "NULL on this instance, so `force_fp16` models silently ran fp32 activations. tracker/native/build.sh patches a core-1.1 "
           "fallback; L5 and P3 were re-measured with real fp16 (ETDump: `*_half` shaders): backbone_x 157.3 → 147.0 ms, head 199.2 → 170.9 ms.",
           "- `authors_cuda_reference` = the LightFC authors' released raw OTB trajectories (CUDA, PyTorch), evaluated by the same script.",
           "- Capture → motor command = capture timestamp of the frame that started a motor pulse → `set_motor` RPC return.", ""]
    open(os.path.join(HERE, "SCORECARD.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
