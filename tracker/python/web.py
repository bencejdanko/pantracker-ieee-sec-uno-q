"""Flask UI (LAN only): MJPEG feed, /state JSON, prompt, jog, roll, gains, placement, camera on/off, E-STOP."""
import logging
import os
import time

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def serve(app, host, port):
    web = Flask(__name__, static_folder=STATIC)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @web.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @web.get("/state")
    def state():
        return jsonify(app.snapshot())

    @web.get("/vocab")
    def vocab():
        return jsonify(app.vocab.words)

    @web.post("/prompt")
    def prompt():
        ok, msg = app.set_prompt((request.json or {}).get("text", ""))
        return jsonify(ok=ok, msg=msg), (200 if ok else 400)

    @web.post("/clear")
    def clear():
        app.clear_prompt()
        return jsonify(ok=True)

    @web.post("/jog")
    def jog():
        app.jog(int((request.json or {}).get("dir", 0)))
        return jsonify(ok=True)

    @web.post("/estop")
    def estop():
        app.set_estop(bool((request.json or {}).get("on", True)))
        return jsonify(ok=True, estop=app.estop)

    @web.post("/camera")
    def camera():
        app.set_camera(bool((request.json or {}).get("on", True)))
        return jsonify(ok=True, camera_on=app.cfg["camera_on"])

    @web.post("/config")
    def config():
        app.update_cfg(request.json or {})
        return jsonify(ok=True)

    @web.post("/placement")
    def placement():
        p = (request.json or {}).get("placement", "")
        if p not in app.cfg["placements"]:
            return jsonify(ok=False, msg="placement not in the parity-checked list"), 400
        t0 = time.perf_counter()
        app.set_placement(p)
        return jsonify(ok=True, placement=app.trk.placement, switch_ms=(time.perf_counter() - t0) * 1e3)

    @web.get("/frame.jpg")
    def frame():  # raw (roll-corrected, un-annotated) latest frame, for calibration/clip recording
        item = app.frames.item
        if item is None:
            return "no frame", 503
        ok, jpg = cv2.imencode(".jpg", item[2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(jpg.tobytes(), mimetype="image/jpeg")

    @web.get("/stream.mjpg")
    def stream():
        def gen():
            period = 1.0 / app.cfg["stream_fps"]
            last = None
            while app.running:
                img = app.annot
                if img is None or img is last:
                    time.sleep(0.01)
                    continue
                last = img
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(period)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    web.run(host=host, port=port, threaded=True, use_reloader=False)
