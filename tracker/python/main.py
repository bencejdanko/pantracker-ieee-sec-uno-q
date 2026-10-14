"""Detect-and-track pan camera on the UNO Q: YOLOv8n (COCO) finds the target, LightFC tracks it.

Threads: capture (V4L2 -> roll-correct -> latest-frame slot), perception (state machine: YOLOv8n in SEARCH,
LightFC in TRACK), verifier (YOLOv8n on the GPU while tracking, own libinfer handle), motor (Bridge keep-alive),
web (Flask). Run: ~/tracker/.venv/bin/python main.py [--port 8000]
"""
import argparse
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

from bridge import Bridge
from control import SETTLED, StepController
from detector import Classes, Detector
from lightfc import LightFC
from sysstats import Sampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.environ.get("TRACKER_MODELS", os.path.join(ROOT, "models"))
CFG_PATH = os.path.join(HERE, "config.json")
log = logging.getLogger("tracker")

IDLE, SEARCH, TRACK, LOST = "IDLE", "SEARCH", "TRACK", "LOST"


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class FrameSlot:
    """Single-slot latest-frame buffer: writers overwrite, readers wait for a newer sequence number."""

    def __init__(self):
        self.cv = threading.Condition()
        self.item = None  # (seq, t_capture, bgr)

    def put(self, item):
        with self.cv:
            self.item = item
            self.cv.notify_all()

    def get_newer(self, seq, timeout=1.0):
        with self.cv:
            self.cv.wait_for(lambda: self.item is not None and self.item[0] > seq, timeout)
            return self.item


class Stat:
    def __init__(self, n=60):
        self.d = deque(maxlen=n)

    def add(self, v):
        self.d.append(v)

    def p50(self):
        return float(np.percentile(self.d, 50)) if self.d else 0.0

    def p95(self):
        return float(np.percentile(self.d, 95)) if self.d else 0.0


class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.running = True
        self.frames = FrameSlot()
        self.vocab = Classes()
        det_path = os.path.join(MODELS, cfg["detector"])
        self.det = Detector(det_path, threads=cfg["threads"], size=cfg["detector_size"])
        # TRACK-time verifier on the GPU (fp16, ~0.6 s per check): slower than the CPU, but it leaves the A53s to the
        # tracker. SEARCH keeps the faster CPU int8 detector.
        ver_path = os.path.join(MODELS, cfg.get("verify_detector") or cfg["detector"])
        try:
            self.ver_det = Detector(ver_path, threads=cfg["threads"], size=cfg["detector_size"])
        except Exception as e:
            log.warning("verify detector %s unavailable (%s); verifying on the CPU detector", ver_path, e)
            ver_path = det_path
            self.ver_det = Detector(det_path, threads=cfg["threads"], size=cfg["detector_size"])
        self.ver_name = os.path.basename(ver_path)
        self.trk = LightFC(os.path.join(MODELS, "lightfc"), cfg["placement"], threads=cfg["threads"])
        # preload every switchable placement: the first Vulkan load compiles shaders (~1.5 s), which must not
        # happen under the app lock in the middle of the demo
        for p in cfg["placements"]:
            try:
                t0 = time.perf_counter()
                self.trk.set_placement(p)
                log.info("preloaded placement %s in %.0f ms", p, (time.perf_counter() - t0) * 1e3)
            except Exception as e:
                log.warning("placement %s unavailable: %s", p, e)
        self.trk.set_placement(cfg["placement"])
        self.ctrl = StepController(cfg)
        self.bridge = Bridge()
        # shared state
        self.state, self.word, self.cls = IDLE, None, None
        self.alert = "No instructions"
        self.box, self.score = None, 0.0
        self.hits, self.last_hit = 0, None
        self.low_since = None
        self.lost_until = 0.0
        self.cmd, self.cmd_frame_t, self.err = 0, 0.0, 0.0
        self.det_best = 0.0  # highest raw detector score on the last SEARCH frame (even below det_conf)
        self.pending_reinit = None
        self.jog_dir, self.jog_until = 0, 0.0
        self.estop = False
        self.verify_req = None  # (frame_bgr, box, track_epoch, t_capture) handed to the verifier
        self.last_pulse_t = 0.0  # monotonic time the last motor pulse started (verifier results older than it are void)
        self.track_epoch = 0    # bumped on every entry to TRACK, so stale verifier results are ignored
        self.verify_misses = 0  # consecutive verifier checks that found no target at all
        self.verify_evt = threading.Event()
        self.last_verify = 0.0
        self.annot = None  # latest annotated JPEG source frame
        self.off_img = self._off_frame()
        self.stats = {"fps": Stat(), "loop_ms": Stat(), "det_ms": Stat(), "trk_ms": Stat(), "verify_ms": Stat(),
                      "glass_to_motor_ms": Stat(), "capture_fps": Stat(), "bridge_ms": Stat()}
        self.stage = {}
        self.sampler = Sampler()
        self.sys = {}
        self.counters = {"frames": 0, "dropped": 0, "exceptions": 0, "reinit": 0, "verify": 0, "verify_lost": 0, "verify_stale": 0, "bridge_err": 0}

    # ---------------- capture ----------------
    def _open_camera(self):
        c = self.cfg
        cap = cv2.VideoCapture(c["camera"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # UVC controls after the stream is configured. The LifeCam drops to 7.5 FPS whenever auto exposure wants more
        # light (any exposure > 15.6 ms at 60 Hz anti-flicker), so the default config pins a manual exposure that
        # keeps 30 FPS and lifts brightness to compensate (bench: 7.5 -> 30 FPS, mean level 135 -> 98).
        ctrls = c.get("camera_ctrls") or {}
        if ctrls:
            args = ["v4l2-ctl", "-d", str(c["camera"])]
            for k, v in ctrls.items():
                args += ["-c", f"{k}={v}"]
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode:
                log.warning("camera controls not applied: %s", (r.stderr or r.stdout).strip())
        return cap

    def capture_loop(self):
        cap = None
        seq, last = 0, time.monotonic()
        while self.running:
            if not self.cfg["camera_on"]:
                # camera switched off from the UI: release the device (the LifeCam LED goes dark)
                if cap is not None:
                    cap.release()
                    cap = None
                    self.frames.put(None)
                if self.annot is not self.off_img:  # also overwrites a frame perception annotated mid-switch
                    self.annot = self.off_img
                time.sleep(0.2)
                continue
            if cap is None:
                cap = self._open_camera()
                last = time.monotonic()
            ok, f = cap.read()
            t = time.monotonic()
            if not ok:
                log.warning("camera read failed; reopening")
                cap.release()
                time.sleep(0.5)
                cap = self._open_camera()
                continue
            th = self.cfg["roll_deg"]
            if abs(th) > 0.05:
                h, w = f.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), th, 1.0)
                f = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR)
            seq += 1
            self.stats["capture_fps"].add(1.0 / max(1e-6, t - last))
            last = t
            self.frames.put((seq, t, f))

    def _off_frame(self):
        img = np.zeros((self.cfg["height"], self.cfg["width"], 3), np.uint8)
        cv2.putText(img, "CAMERA OFF", (self.cfg["width"] // 2 - 110, self.cfg["height"] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (160, 160, 160), 2, cv2.LINE_AA)
        return img

    # ---------------- motor ----------------
    def motor_loop(self):
        period = 1.0 / self.cfg["keepalive_hz"]
        last_sent = None
        while self.running:
            now = time.monotonic()
            with self.lock:
                if self.estop:
                    speed = 0
                elif now < self.jog_until:
                    speed = self.jog_dir * self.cfg["jog_pwm"]
                elif self.state == TRACK and self.cfg.get("follow"):
                    speed = self.ctrl.speed(now)  # nonzero only inside a step pulse
                else:
                    speed = 0
                self.cmd = speed
                # glass-to-motor: capture timestamp of the frame that started the current pulse
                frame_t = self.cmd_frame_t if speed and self.state == TRACK else 0.0
            try:
                t0 = time.monotonic()
                self.bridge.set_motor(speed)
                t1 = time.monotonic()
                self.stats["bridge_ms"].add((t1 - t0) * 1e3)
                if frame_t and frame_t != last_sent:
                    self.stats["glass_to_motor_ms"].add((t1 - frame_t) * 1e3)
                    last_sent = frame_t
            except Exception as e:  # the MCU watchdog stops the motor if we can't reach it
                self.counters["bridge_err"] += 1
                log.debug("bridge: %s", e)
                time.sleep(0.2)
            time.sleep(max(0.0, period - (time.monotonic() - now)))
        try:
            self.bridge.stop()
        except Exception:
            pass

    # ---------------- verifier ----------------
    def verify_loop(self):
        try:
            os.nice(5)  # Linux applies nice() to the calling thread only: verification is low priority
        except OSError:
            pass
        while self.running:
            if not self.verify_evt.wait(0.5):
                continue
            self.verify_evt.clear()
            with self.lock:
                req, cls = self.verify_req, self.cls
            if req is None or cls is None:
                continue
            frame, box, epoch, t_cap = req
            t0 = time.perf_counter()
            dets = self.ver_det.detect(frame, cls, self.cfg["det_conf"])
            self.stats["verify_ms"].add((time.perf_counter() - t0) * 1e3)
            self.counters["verify"] += 1
            with self.lock:
                if self.state != TRACK or epoch != self.track_epoch:
                    continue
                # The camera may have pulsed after the checked frame was captured (a GPU check lags ~1.4 s). A miss is
                # still valid evidence (that still frame had no target) and a hit still resets the strikes; only a
                # re-init is skipped, because the detection's coordinates no longer describe the current scene.
                moved = self.last_pulse_t > t_cap
                if not dets:
                    # nothing of the prompted class anywhere: the tracker has drifted onto background
                    self.verify_misses += 1
                    if self.verify_misses >= self.cfg["verify_strikes"]:
                        self.counters["verify_lost"] += 1
                        self._to(LOST)
                    continue
                self.verify_misses = 0
                if moved:
                    self.counters["verify_stale"] += 1  # re-init check skipped (camera moved since capture)
                    continue
                if max(iou(d[:4], box) for d in dets) < self.cfg["verify_iou"]:
                    # the target is elsewhere: re-init on the detection nearest the current box (not the most
                    # confident one, which could be a different person)
                    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                    d = min(dets, key=lambda d: (d[0] + d[2] / 2 - cx) ** 2 + (d[1] + d[3] / 2 - cy) ** 2)
                    # template from the checked frame itself (the detection's coordinates belong to it); the search
                    # region (4x box) around that old position re-finds the target in the current frame
                    self.pending_reinit = (frame, list(d[:4]))  # frames are never modified in place
                    self.counters["reinit"] += 1

    # ---------------- perception / state machine ----------------
    def set_prompt(self, text):
        word, cls_or_sugg = self.vocab.lookup(text or "")
        with self.lock:
            if not self.cfg["camera_on"]:
                self.alert = "Camera is off"
                return False, self.alert
            if word is None:
                sug = ", ".join(cls_or_sugg) if cls_or_sugg else "nothing close"
                self.alert = f"Unknown word '{text}'. Did you mean: {sug}?"
                return False, self.alert
            self.word, self.cls = word, cls_or_sugg
            self._to(SEARCH)
            return True, word

    def clear_prompt(self):
        with self.lock:
            self.word = self.cls = None
            self._to(IDLE)

    def _to(self, st):
        self.state = st
        self.hits, self.last_hit, self.low_since = 0, None, None
        self.cmd = 0
        self.ctrl.reset()
        if st == IDLE:
            self.alert, self.box = ("No instructions" if self.cfg["camera_on"] else "Camera off"), None
        elif st == SEARCH:
            self.alert, self.box = f"Searching for '{self.word}'", None
        elif st == TRACK:
            self.alert = f"Tracking '{self.word}'"
            self.track_epoch += 1
            self.verify_misses = 0
        elif st == LOST:
            self.alert = "Target lost"
            self.lost_until = time.monotonic() + 1.0

    def perception_loop(self):
        seq, last_t = 0, time.monotonic()
        while self.running:
            item = self.frames.get_newer(seq)
            if item is None or item[0] <= seq:
                continue
            if seq and item[0] > seq + 1:
                self.counters["dropped"] += item[0] - seq - 1
            seq, t_cap, frame = item
            self.counters["frames"] += 1
            t0 = time.monotonic()
            try:
                self._step(frame, t_cap)
            except Exception:
                self.counters["exceptions"] += 1
                log.exception("perception step failed")
                with self.lock:
                    self._to(SEARCH if self.word else IDLE)
            t1 = time.monotonic()
            self.stats["loop_ms"].add((t1 - t0) * 1e3)
            self.stats["fps"].add(1.0 / max(1e-6, t1 - last_t))
            last_t = t1

    def _step(self, frame, t_cap):
        H, W = frame.shape[:2]
        with self.lock:
            st, cls = self.state, self.cls
        if st == LOST and time.monotonic() >= self.lost_until:
            with self.lock:
                self._to(SEARCH)
            st = SEARCH
        if st == SEARCH:
            t0 = time.perf_counter()
            dets = self.det.detect(frame, cls, self.cfg["det_conf"])
            self.stats["det_ms"].add((time.perf_counter() - t0) * 1e3)
            self.det_best = self.det.last_max
            with self.lock:
                if self.state != SEARCH:
                    return self._annotate(frame, [])
                if dets:
                    d = dets[0]
                    cx = d[0] + d[2] / 2
                    if self.last_hit is not None and abs(cx - self.last_hit) <= self.cfg["hit_radius"] * W:
                        self.hits += 1
                    else:
                        self.hits = 1
                    self.last_hit = cx
                    self.box, self.score = list(d[:4]), d[4]
                    if self.hits >= self.cfg["hits_to_track"]:
                        self.trk.initialize(frame, d[:4])
                        self._to(TRACK)
                        self.box = list(d[:4])
                        self.last_verify = time.monotonic()
                else:
                    self.hits, self.last_hit, self.box = 0, None, None
            return self._annotate(frame, dets)
        if st == TRACK:
            with self.lock:
                re = self.pending_reinit
                self.pending_reinit = None
            if re is not None:
                self.trk.initialize(*re)
            t0 = time.perf_counter()
            box, score = self.trk.track(frame)
            self.stats["trk_ms"].add((time.perf_counter() - t0) * 1e3)
            self.stage = dict(self.trk.t)
            now = time.monotonic()
            with self.lock:
                if self.state != TRACK:
                    return self._annotate(frame, [])
                self.box, self.score = box, score
                # blurry frames (motor pulsing or settling) never count toward LOST
                still = not self.cfg.get("follow") or self.ctrl.phase(now) == SETTLED
                if score < self.cfg["lost_score"] and still:
                    self.low_since = self.low_since or now
                    if now - self.low_since >= self.cfg["lost_time"]:
                        self._to(LOST)
                        return self._annotate(frame, [])
                elif score >= self.cfg["lost_score"]:
                    self.low_since = None
                e, started = self.ctrl.update(box, W, t_cap, now) if self.cfg.get("follow") else (
                    (box[0] + box[2] / 2 - W / 2) / (W / 2), False)
                self.err = e
                if started:
                    self.cmd_frame_t = t_cap
                    self.last_pulse_t = now
                if now - self.last_verify >= self.cfg["verify_period"] and still:  # no blurry frames
                    self.last_verify = now
                    self.verify_req = (frame, list(box), self.track_epoch, t_cap)
                    self.verify_evt.set()
            return self._annotate(frame, [])
        return self._annotate(frame, [])

    def _annotate(self, frame, dets):
        img = frame.copy()
        H, W = img.shape[:2]
        cv2.line(img, (W // 2, 0), (W // 2, H), (80, 80, 80), 1)
        for d in dets[1:]:
            x, y, w, h = map(int, d[:4])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 255), 1)
        with self.lock:
            st, box, score, word = self.state, self.box, self.score, self.word
        if box is not None:
            x, y, w, h = map(int, box)
            col = (0, 220, 0) if st == TRACK else (0, 200, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, f"{word} {score:.2f}", (x + 4, max(42, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        fps = self.stats["fps"].p50()
        txt = f"{st}  {fps:4.1f} fps  roll {self.cfg['roll_deg']:+.1f}  [{self.trk.placement}]"
        if st == SEARCH:
            txt += f"  best {self.det_best:.2f}/{self.cfg['det_conf']:.2f}"
        if st == TRACK and self.cfg.get("follow"):
            txt += f"  motor {self.ctrl.phase(time.monotonic())}"
        else:
            txt += "  motor off"
        cv2.rectangle(img, (0, 0), (W, 26), (20, 20, 20), -1)
        cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.annot = img

    # ---------------- control surface (web) ----------------
    def jog(self, direction):
        with self.lock:
            self.jog_dir = int(np.sign(direction))
            self.jog_until = time.monotonic() + 0.25 if direction else 0.0

    def set_estop(self, on):
        with self.lock:
            self.estop = bool(on)
            if on:
                self.jog_until = 0.0
        if on:
            try:
                self.bridge.stop()
            except Exception:
                pass

    def set_camera(self, on):
        with self.lock:
            self.cfg["camera_on"] = bool(on)
            if not on:  # nothing to track without frames: drop the prompt so the motor stays still
                self.word = self.cls = None
                self._to(IDLE)
            else:
                self.alert = "No instructions"
        save_cfg(self.cfg)  # persisted, so the camera stays off across a reboot

    def set_placement(self, placement):
        with self.lock:  # blocks perception between frames
            self.trk.set_placement(placement)
            self.cfg["placement"] = self.trk.placement

    def update_cfg(self, kv, persist=True):
        num = {k for k, v in self.cfg.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        with self.lock:
            for k, v in kv.items():
                if k in num:
                    self.cfg[k] = type(self.cfg[k])(float(v))
                elif k in ("invert", "follow"):
                    self.cfg[k] = bool(v)
        if persist:
            save_cfg(self.cfg)

    def snapshot(self):
        with self.lock:
            s = {"state": self.state, "word": self.word, "alert": self.alert,
                 "box": [round(float(v), 1) for v in self.box] if self.box else None,
                 "score": round(float(self.score), 3), "cmd": self.cmd,
                 "ctrl": self.ctrl.phase(time.monotonic()) if self.state == TRACK and self.cfg.get("follow") else "-",
                 "det_best": round(float(self.det_best), 3),
                 "err": round(float(self.err), 3), "estop": self.estop, "camera_on": self.cfg["camera_on"], "placement": self.trk.placement,
                 "placements": self.cfg["placements"], "detector": self.cfg["detector"], "verify_detector": self.ver_name,
                 "cfg": {k: self.cfg.get(k) for k in ("deadband", "trigger_band", "edge_margin", "big_box", "pulse_pwm", "pulse_k_ms", "pulse_min_ms", "pulse_max_ms", "settle_ms", "settle_frames",
                                                  "invert", "follow", "det_conf",
                                                  "lost_score", "lost_time", "verify_period", "verify_strikes", "roll_deg", "jog_pwm")}}
        s["fps"] = round(self.stats["fps"].p50(), 2)
        s["lat"] = {k: [round(v.p50(), 2), round(v.p95(), 2)] for k, v in self.stats.items() if k != "fps"}
        s["stage_ms"] = {k: round(v, 2) for k, v in self.stage.items()}
        s["sys"] = self.sys
        s["counters"] = dict(self.counters)
        return s

    def stats_loop(self):
        while self.running:
            try:
                self.sys = self.sampler.sample()
            except Exception:
                pass
            time.sleep(0.5)


# step-and-settle defaults for config.json files written before the controller existed
CTRL_DEFAULTS = {"follow": False, "deadband": 0.08, "trigger_band": 0.35, "edge_margin": 0.08, "big_box": 0.6, "pulse_pwm": 25, "pulse_k_ms": 600, "pulse_min_ms": 60,
                 "pulse_max_ms": 400, "settle_ms": 300, "settle_frames": 2, "invert": False, "verify_strikes": 2,
                 "camera_on": True}
OBSOLETE = ("kp", "kd", "min_pwm", "max_pwm")


def load_cfg():
    cfg = {**CTRL_DEFAULTS, **json.load(open(CFG_PATH))}
    for k in OBSOLETE:
        cfg.pop(k, None)
    return cfg


def save_cfg(cfg):
    tmp = CFG_PATH + ".tmp"
    json.dump(cfg, open(tmp, "w"), indent=2)
    os.replace(tmp, CFG_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--prompt", default=None, help="start with this prompt (headless runs)")
    ap.add_argument("--placement", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_cfg()
    if a.placement:
        cfg["placement"] = a.placement
    app = App(cfg)
    for fn in (app.capture_loop, app.perception_loop, app.verify_loop, app.motor_loop, app.stats_loop):
        threading.Thread(target=fn, name=fn.__name__, daemon=True).start()
    if a.prompt:
        app.set_prompt(a.prompt)
    from web import serve
    try:
        serve(app, a.host, a.port)
    finally:
        app.running = False
        time.sleep(0.2)


if __name__ == "__main__":
    main()
