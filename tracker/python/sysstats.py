"""Process/system telemetry for the stats panel and the bench harness."""
import glob
import os
import time

GPU_DEVFREQ = "/sys/class/devfreq/5900000.gpu"
CLK = os.sysconf("SC_CLK_TCK")


def _read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def hwmon_temps():
    out = {}
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        name = _read(f"{d}/name")
        t = _read(f"{d}/temp1_input")
        if name and t:
            out[name] = int(t) / 1000.0
    return out


def gpu_engine_ns(pid="self"):
    """Sum of DRM fdinfo drm-engine-gpu over this process' msm fds (deduplicated by client id)."""
    seen, tot = set(), 0
    for f in glob.glob(f"/proc/{pid}/fdinfo/*"):
        txt = _read(f)
        if "drm-engine-gpu" not in txt:
            continue
        cid = eng = None
        for line in txt.splitlines():
            if line.startswith("drm-client-id:"):
                cid = line.split()[1]
            elif line.startswith("drm-engine-gpu:"):
                eng = int(line.split()[1])
        if cid not in seen and eng is not None:
            seen.add(cid)
            tot += eng
    return tot


class Sampler:
    """Call sample() periodically; returns deltas since the previous call."""

    def __init__(self, pid="self"):
        self.pid = pid
        self.prev = None

    def _snap(self):
        st = _read(f"/proc/{self.pid}/stat").rsplit(")", 1)[-1].split()
        cpu = (int(st[11]) + int(st[12])) / CLK  # utime+stime
        sysl = _read("/proc/stat").splitlines()[0].split()[1:]
        sysv = list(map(int, sysl))
        return time.monotonic(), cpu, sum(sysv), sysv[3] + sysv[4], gpu_engine_ns(self.pid)

    def sample(self):
        now = self._snap()
        out = {"gpu_freq_mhz": int(_read(f"{GPU_DEVFREQ}/cur_freq", "0")) / 1e6,
               "cpu_freq_mhz": [int(_read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "0")) / 1e3
                                for i in range(4)],
               "temps": hwmon_temps(),
               "rss_mb": _rss_mb(self.pid), "hwm_mb": _rss_mb(self.pid, "VmHWM")}
        if self.prev:
            dt = now[0] - self.prev[0]
            out["cpu_pct"] = 100.0 * (now[1] - self.prev[1]) / dt  # 400 % = all 4 cores
            dtot = now[2] - self.prev[2]
            out["sys_cpu_pct"] = 100.0 * (1 - (now[3] - self.prev[3]) / dtot) if dtot else 0.0
            out["gpu_busy_pct"] = 100.0 * (now[4] - self.prev[4]) / (dt * 1e9)
        self.prev = now
        return out


def _rss_mb(pid, key="VmRSS"):
    for line in _read(f"/proc/{pid}/status").splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024.0
    return 0.0
