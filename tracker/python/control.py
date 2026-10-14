"""Step-and-settle pan controller.

The motor has no encoder and the camera blurs while it turns, so the vision pipeline is only trusted when the camera
is still. Each correction is one short pulse at a fixed low PWM, its length proportional to the horizontal error;
then the motor stops and the controller waits until (a) settle_ms have passed and (b) the tracker has processed
settle_frames frames *captured after the stop*, before it measures the error again.

    SETTLED --|e| > deadband--> PULSE --pulse_ms elapsed--> SETTLE --time + fresh frames--> SETTLED

Two bands stop hunting when the turntable's smallest step is coarse (target far away, small box): a correction
*starts* only when |e| > trigger_band or the box comes within edge_margin of the frame edge on the side it is
drifting to; it *ends* when |e| < deadband, or when a step overshoots to the other side but lands inside
trigger_band (a little off-centre is accepted instead of chased).
A box wider than big_box x frame (target close) always touches the edges, so the edge rule is ignored and only
|e| > trigger_band moves; wider than 0.9 x frame there is nothing to centre, so the motor never moves.
"""
import math

SETTLED, PULSE, SETTLE = "SETTLED", "PULSE", "SETTLE"


class StepController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.pulse_until = 0.0   # monotonic time the current/last pulse ends
        self.dir = 0             # +1 / -1 while pulsing (invert already applied)
        self.post_frames = 0     # tracker frames captured after the last pulse ended
        self.correcting = 0      # sign of the error being corrected (0 = idle, accepting off-centre)

    def phase(self, now):
        c = self.cfg
        if now < self.pulse_until:
            return PULSE
        if now - self.pulse_until < c["settle_ms"] / 1e3 or self.post_frames < c["settle_frames"]:
            return SETTLE
        return SETTLED

    def speed(self, now):
        """PWM to send right now (called by the motor thread at the keep-alive rate)."""
        return self.dir * int(self.cfg["pulse_pwm"]) if now < self.pulse_until else 0

    def update(self, box, frame_w, t_cap, now):
        """One tracker frame (box = [x, y, w, h]). Returns (e, started_pulse). Only frames captured while still
        can start a pulse."""
        c = self.cfg
        cx = box[0] + box[2] / 2
        e = (cx - frame_w / 2) / (frame_w / 2)
        if t_cap > self.pulse_until:
            self.post_frames += 1
        if self.phase(now) != SETTLED:
            return e, False
        side = 1 if e > 0 else -1
        if box[2] > 0.9 * frame_w:  # fills the frame: nothing to centre
            self.correcting = 0
            return e, False
        m = c.get("edge_margin", 0.08) * frame_w
        big = box[2] > c.get("big_box", 0.6) * frame_w
        near_edge = not big and ((box[0] + box[2] > frame_w - m) if side > 0 else (box[0] < m))
        outside = abs(e) > c.get("trigger_band", 0.35) or (near_edge and abs(e) >= c["deadband"])
        if self.correcting:
            if abs(e) < c["deadband"] or (side != self.correcting and not outside):
                self.correcting = 0  # centred, or overshot but acceptably close: stop here
        elif outside:
            self.correcting = side
        if not self.correcting:
            return e, False
        ms = min(c["pulse_max_ms"], max(c["pulse_min_ms"], c["pulse_k_ms"] * abs(e)))
        self.dir = int(math.copysign(1, e)) * (-1 if c["invert"] else 1)
        self.pulse_until = now + ms / 1e3
        self.post_frames = 0
        return e, True
