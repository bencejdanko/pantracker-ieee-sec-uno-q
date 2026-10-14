"""Efficiency-scorecard runner, executed ON the board.

For each row: wait until cpuss < 50 °C, then
  A = perf stat (A53 PMU, user-space) -- infer_bench model -w W -n N    (latency JSON + counters + peak RSS)
  B = perf stat ... -- infer_bench model -w 0 -n 0                        (load-only baseline)
  per-frame counters = (A - B) / (W + N)
while sampling GPU devfreq + DRM fdinfo busy time, CPU % of the child, and temperatures.
LightFC rows sum their per-frame blocks (backbone_x + fusion + head); backbone_z is reported separately.

  python matrix.py --rows all --out ../bench/results/scorecard.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from sysstats import GPU_DEVFREQ, gpu_engine_ns, hwmon_temps  # noqa: E402

BENCH = os.path.join(ROOT, "lib", "infer_bench")
M = os.path.join(ROOT, "models")
L = os.path.join(M, "lightfc")
EVENTS = ["cycles", "instructions", "armv8_cortex_a53/l1d_cache_refill/", "armv8_cortex_a53/l2d_cache_refill/",
          "armv8_cortex_a53/bus_access/", "armv8_cortex_a53/mem_access/", "armv8_cortex_a53/l2d_cache/"]
SHORT = ["cycles", "instructions", "l1d_refill", "l2d_refill", "bus_access", "mem_access", "l2d_access"]


def lfc(v, suffix=""):
    return [os.path.join(L, f"{b}_{v}{suffix}.pte") for b in ("backbone_x", "fusion", "head")]


def lfc_mixed(x, f, h):
    return [os.path.join(L, f"backbone_x_{x}.pte"), os.path.join(L, f"fusion_{f}.pte"), os.path.join(L, f"head_{h}.pte")]


# row id -> (description, [per-frame .pte files], optional extra .pte reported separately)
ROWS = {
    "L0": ("LightFC portable fp32, no re-param", lfc("portable", "_norep"), os.path.join(L, "backbone_z_portable_norep.pte")),
    "L1": ("LightFC XNNPACK fp32, no re-param", lfc("xnn32", "_norep"), os.path.join(L, "backbone_z_xnn32_norep.pte")),
    "L2": ("LightFC XNNPACK fp32 + repN33", lfc("xnn32"), os.path.join(L, "backbone_z_xnn32.pte")),
    "L3": ("LightFC XNNPACK int8 PTQ + repN33", lfc("xnn8"), os.path.join(L, "backbone_z_xnn8.pte")),
    "L4": ("LightFC Vulkan fp32 + repN33", lfc("vk32"), os.path.join(L, "backbone_z_vk32.pte")),
    "L5": ("LightFC Vulkan fp16 + repN33", lfc("vk16"), os.path.join(L, "backbone_z_vk16.pte")),
    "P3": ("LightFC x=vk16, fusion=xnn32, head=xnn32", lfc_mixed("vk16", "xnn32", "xnn32"), None),
    "P4": ("LightFC x=xnn32, fusion=vk16, head=vk16", lfc_mixed("xnn32", "vk16", "vk16"), None),
    "V1": ("YOLOv8n XNNPACK fp32 @320", [os.path.join(M, "yolov8n_xnn32_320.pte")], None),
    "V2": ("YOLOv8n XNNPACK int8 @320", [os.path.join(M, "yolov8n_xnn8_320.pte")], None),
    "V3": ("YOLOv8n Vulkan fp32 @320", [os.path.join(M, "yolov8n_vk32_320.pte")], None),
    "V4": ("YOLOv8n Vulkan fp16 @320", [os.path.join(M, "yolov8n_vk16_320.pte")], None),
}


def temp_cpu():
    t = hwmon_temps()
    vals = [v for k, v in t.items() if k.startswith("cpuss")]
    return max(vals) if vals else 0.0


def wait_cool(limit=50.0, timeout=180):
    t0 = time.time()
    while temp_cpu() >= limit and time.time() - t0 < timeout:
        time.sleep(2)
    return temp_cpu()


def gpu_trans_ms():
    """devfreq trans_stat 'time(ms)' column per frequency."""
    out = {}
    try:
        for line in open(f"{GPU_DEVFREQ}/trans_stat").read().splitlines():
            p = line.replace("*", " ").split()
            if p and p[0].rstrip(":").isdigit() and len(p) > 2:
                out[int(p[0].rstrip(":"))] = int(p[-1])
    except OSError:
        pass
    return out


def child_of(ppid):
    """First child pid of ppid (no CONFIG_PROC_CHILDREN on this kernel: scan /proc/*/stat)."""
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]) == ppid:
                    return d
            except (OSError, IndexError, ValueError):
                pass
    return None


def run_one(pte, warm, n, threads, inputs=()):
    """perf stat + infer_bench; returns (bench_json, counters, gpu_busy_s, wall_s)."""
    cmd = ["perf", "stat", "-x", ",", "-e", ",".join(EVENTS), "--", BENCH, pte, "-w", str(warm), "-n", str(n),
           "-t", str(threads)]
    for i in inputs:
        cmd += ["-i", i]
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_max, child = 0, None
    while p.poll() is None:  # sample the infer_bench child's GPU engine time
        try:
            if child is None:
                child = child_of(p.pid)
            if child:
                gpu_max = max(gpu_max, gpu_engine_ns(child))
        except OSError:
            pass
        time.sleep(0.05)
    wall = time.monotonic() - t0
    out, err = p.stdout.read(), p.stderr.read()
    bj = None
    for line in out.splitlines():
        if line.startswith("{"):
            bj = json.loads(line)
    ctr = {}
    for line in err.splitlines():
        f = line.split(",")
        if len(f) > 2:
            for ev, short in zip(EVENTS, SHORT):
                if f[2].startswith(ev.split(":")[0]):
                    try:
                        ctr[short] = int(f[0])
                    except ValueError:
                        ctr[short] = None
    if bj is None:
        raise RuntimeError(f"bench failed: {pte}\n{err[-800:]}")
    return bj, ctr, gpu_max / 1e9, wall


def peak_rss_mb(pte, threads):
    """Separate short run (no perf) so VmHWM belongs to infer_bench itself."""
    p = subprocess.Popen([BENCH, pte, "-w", "3", "-n", "5", "-t", str(threads)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    _, _, ru = os.wait4(p.pid, 0)
    return ru.ru_maxrss / 1024.0, (ru.ru_utime + ru.ru_stime)


def measure(pte, warm, n, threads):
    bj, ca, gpu_s, wall = run_one(pte, warm, n, threads)
    _, cb, _, _ = run_one(pte, 0, 0, threads)
    iters = warm + n
    per = {k: ((ca.get(k) or 0) - (cb.get(k) or 0)) / iters for k in SHORT}
    rss, _ = peak_rss_mb(pte, threads)
    return {"p50_ms": bj["p50_ms"], "p95_ms": bj["p95_ms"], "mean_ms": bj["mean_ms"], "load_ms": bj["load_ms"],
            "size_mb": os.path.getsize(pte) / 1e6, "peak_rss_mb": rss, "gpu_busy_ms_per_frame": 1e3 * gpu_s / iters,
            **{f"{k}_per_frame": v for k, v in per.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="L1,L2,L3,L4,L5,P3,P4,V1,V2,V3,V4,L0")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "scorecard.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    blocks_out = a.out.replace(".csv", "_blocks.csv")
    rows_out = []
    bf = open(blocks_out, "w", newline="")
    bw = None
    for rid in a.rows.split(","):
        desc, files, extra = ROWS[rid]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"{rid}: n/a (missing {', '.join(os.path.basename(m) for m in missing)})", flush=True)
            rows_out.append({"row": rid, "desc": desc, "status": "n/a: not exported"})
            continue
        warm, n = (a.warm, a.n) if "portable" not in files[0] else (2, 10)
        if rid.startswith("V"):
            warm, n = min(warm, 5), min(n, 50)
        tot = {}
        gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
        t_before = wait_cool()
        g0 = gpu_trans_ms()
        try:
            for f in files + ([extra] if extra and os.path.exists(extra) else []):
                r = measure(f, warm, n, a.threads)
                r.update(row=rid, block=os.path.basename(f), governor=gov)
                if bw is None:
                    bw = csv.DictWriter(bf, fieldnames=list(r.keys()))
                    bw.writeheader()
                bw.writerow(r)
                bf.flush()
                print(f"  {rid} {os.path.basename(f):32s} p50 {r['p50_ms']:8.2f} ms  p95 {r['p95_ms']:8.2f}  "
                      f"cyc/f {r['cycles_per_frame']:.3g}  gpu {r['gpu_busy_ms_per_frame']:.2f} ms/f", flush=True)
                if f == extra:
                    tot["backbone_z_p50_ms"] = r["p50_ms"]
                    continue
                for k, v in r.items():
                    if isinstance(v, (int, float)) and k not in ("peak_rss_mb", "load_ms"):
                        tot[k] = tot.get(k, 0) + v
                tot["peak_rss_mb"] = max(tot.get("peak_rss_mb", 0), r["peak_rss_mb"])
        except Exception as e:
            print(f"{rid}: FAILED {e}", flush=True)
            rows_out.append({"row": rid, "desc": desc, "status": f"failed: {str(e)[:200]}"})
            continue
        g1 = gpu_trans_ms()
        tot["gpu_time_at_freq_ms"] = json.dumps({k // 1000000: g1.get(k, 0) - g0.get(k, 0) for k in g1})
        tot["fps_from_p50"] = 1000.0 / tot["p50_ms"] if tot.get("p50_ms") else 0
        tot["ipc"] = tot["instructions_per_frame"] / tot["cycles_per_frame"] if tot.get("cycles_per_frame") else 0
        tot["l2_refill_mb_per_frame"] = tot.get("l2d_refill_per_frame", 0) * 64 / 1e6
        tot.update(row=rid, desc=desc, status="ok", governor=gov, temp_before=t_before, temp_after=temp_cpu(),
                   threads=a.threads, runs=n)
        rows_out.append(tot)
        print(f"{rid} {desc}: p50 {tot['p50_ms']:.2f} ms ({tot['fps_from_p50']:.1f} FPS), size {tot['size_mb']:.2f} MB, "
              f"IPC {tot['ipc']:.2f}, L2 refill {tot['l2_refill_mb_per_frame']:.2f} MB/f", flush=True)
    keys = []
    for r in rows_out:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", a.out, blocks_out)


if __name__ == "__main__":
    main()
