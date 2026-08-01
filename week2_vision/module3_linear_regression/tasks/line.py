#!/usr/bin/env python3
"""
Live line-follower tuner: camera + mask side by side, regressed line, sliders.

Run on the Pi:      python3 ~/tuner_app.py
Open on laptop:     http://<pi-ip>:8765   (e.g. http://baseimage.local:8765)

Sliders write /tmp/tune.json (clamped) — the tunable step3_follow_line.py
picks changes up within one frame. Works with or without the flight script
running, so you can also tune the vision on the bench.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = "/camera/nadir"
TUNE_FILE = "/tmp/tune.json"
PORT = 8765
JPEG_QUALITY = 70
STREAM_FPS = 12.0
SAT_MAX = 255


LIMITS = {
    "V_MIN":      (100.0, 255.0),
    "MIN_PIXELS": (20.0, 20000.0),
    "mresid":     (10.0, 400.0),
    "xm":         (0.0, 0.45),
    "mfill":      (0.05, 1.0),
    "KP":         (0.0, 1.5),
    "KI":         (0.0, 0.5),
    "KD":         (0.0, 1.0),
    "MAX_LAT":    (0.02, 0.6),
    "YAW_GAIN":   (0.0, 3.0),
    "MAX_YAW":    (0.05, 1.0),
    "FORWARD_PITCH": (0.0, 0.5),
    "SAT_MAX": (0.0, 255.0),
    "blur":   (0.0, 15.0),
    "erode":  (0.0, 5.0),
    "dilate": (0.0, 5.0),
}
DEFAULTS = {"V_MIN": 200.0, "MIN_PIXELS": 200.0, "mresid": 150.0, "xm": 0.25,
            "mfill": 0.25, "KP": 0.2, "KI": 0.0, "KD": 0.05, "MAX_LAT": 0.15,
            "YAW_GAIN": 0.9, "MAX_YAW": 0.4, "FORWARD_PITCH": 0.1,
            "SAT_MAX": 255.0, "blur": 0.0, "erode": 0.0, "dilate": 0.0}
STEPS = {"V_MIN": 1, "MIN_PIXELS": 10, "mresid": 5, "xm": 0.01, "mfill": 0.01,
         "KP": 0.01, "KI": 0.005, "KD": 0.01, "MAX_LAT": 0.01, "YAW_GAIN": 0.05,
         "MAX_YAW": 0.05, "FORWARD_PITCH": 0.01, "SAT_MAX": 1, "blur": 1, "erode": 1, "dilate": 1}

K3 = np.ones((3, 3), np.uint8)

_lock = threading.Lock()
_latest_jpeg = None
_status = {"ok": False, "why": "waiting for frames", "fps": 0.0}


def read_params():
    p = dict(DEFAULTS)
    try:
        with open(TUNE_FILE) as f:
            for k, v in json.load(f).items():
                if k in LIMITS:
                    lo, hi = LIMITS[k]
                    p[k] = min(max(float(v), lo), hi)
    except Exception:
        pass
    return p


def write_params(changes):
    applied = {}
    try:
        with open(TUNE_FILE) as f:
            d = json.load(f)
    except Exception:
        d = dict(DEFAULTS)
    for k, v in changes.items():
        if k not in LIMITS:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = LIMITS[k]
        v = min(max(v, lo), hi)
        d[k] = v
        applied[k] = v
    with open(TUNE_FILE + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(TUNE_FILE + ".tmp", TUNE_FILE)
    return applied


def _decode(msg):
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    h, w = msg.height, msg.width
    if msg.encoding == "bgr8":
        return buf.reshape(h, w, 3)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"encoding {msg.encoding}")


def analyze(bgr, p):
    h, w = bgr.shape[:2]
    r1 = h // 2
    hsv_v = bgr.max(axis=2)
    hsv_lo = bgr.min(axis=2)
    if p["blur"] >= 1:
        k = int(p["blur"]) | 1
        hsv_v = cv2.GaussianBlur(hsv_v, (k, k), 0)
        hsv_lo = cv2.GaussianBlur(hsv_lo, (k, k), 0)
    mask = ((hsv_v > p["V_MIN"]) & ((hsv_v - hsv_lo) <= p["SAT_MAX"])).astype(np.uint8)
    if int(p["erode"]):
        mask = cv2.erode(mask, K3, iterations=int(p["erode"]))
    if int(p["dilate"]):
        mask = cv2.dilate(mask, K3, iterations=int(p["dilate"]))
    xl = int(w * p["xm"])
    xr = int(w * (1 - p["xm"]))
    region = mask[:r1, xl:xr]
    rows, cols = np.nonzero(region)
    n = cols.size
    band = r1 * (xr - xl)
    d = {"ok": False, "n": n, "r1": r1, "xl": xl, "xr": xr, "why": ""}
    d["mask"] = mask
    if n < p["MIN_PIXELS"]:
        d["why"] = f"too few px ({n})"
        return d
    if n > p["mfill"] * band:
        d["why"] = f"flooded ({100*n/band:.0f}%)"
        return d
    cols = cols + xl
    m, b = np.polyfit(rows.astype(float), cols.astype(float), 1)
    resid = float(np.std(cols - (m * rows + b)))
    d.update(m=float(m), b=float(b), resid=resid)
    if resid > p["mresid"]:
        d["why"] = f"resid {resid:.0f}>{p['mresid']:.0f}"
        return d
    col_at_drone = m * (r1 - 1) + b
    lateral = -(col_at_drone - w / 2) / (w / 2)
    slope = float(m)
    vr = -1 * min(max(p["KP"] * lateral, -p["MAX_LAT"]), p["MAX_LAT"])
    yr = min(max(-p["YAW_GAIN"] * slope, -p["MAX_YAW"]), p["MAX_YAW"])
    d.update(ok=True, angle=float(np.degrees(np.arctan(m))), lateral=float(lateral),
             col_at_drone=float(col_at_drone), vr=float(vr), yr=float(yr))
    return d


def annotate(bgr, a, p):
    out = bgr.copy()
    h, w = out.shape[:2]
    r1 = a["r1"]
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 220, 0), 1)
    for x in range(0, w, 16):
        cv2.line(out, (x, r1), (min(x + 8, w), r1), (0, 230, 230), 1)
    cv2.line(out, (a["xl"], 0), (a["xl"], r1), (120, 120, 120), 1)
    cv2.line(out, (a["xr"], 0), (a["xr"], r1), (120, 120, 120), 1)
    cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), -1)
    if a["ok"]:
        cv2.line(out, (int(a["b"]), 0), (int(a["m"] * r1 + a["b"]), r1), (0, 0, 255), 3)
        cv2.circle(out, (int(a["col_at_drone"]), r1 - 1), 6, (0, 255, 255), -1)
        txt = (f"ang {a['angle']:+.1f}  lat {a['lateral']:+.2f}  "
               f"vr {a['vr']:+.2f}  yaw {a['yr']:+.2f}")
        cv2.putText(out, txt, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
    else:
        cv2.putText(out, f"NO LINE - {a['why']}", (8, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.rectangle(out, (0, 0), (w, 20), (0, 0, 0), -1)
    top = f"n={a['n']} resid={a.get('resid', 0):.0f} VMIN={p['V_MIN']:.0f} SAT={p['SAT_MAX']:.0f} KP={p['KP']:.2f} YG={p['YAW_GAIN']:.2f}"
    cv2.putText(out, top, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


def mask_panel(a, shape):
    h, w = shape[:2]
    r1 = a["r1"]
    mp = cv2.cvtColor(a["mask"] * 255, cv2.COLOR_GRAY2BGR)
    mp[r1:, :] = mp[r1:, :] // 3
    mp[:r1, :a["xl"]] = mp[:r1, :a["xl"]] // 3
    mp[:r1, a["xr"]:] = mp[:r1, a["xr"]:] // 3
    cv2.line(mp, (a["xl"], 0), (a["xl"], r1), (120, 120, 120), 1)
    cv2.line(mp, (a["xr"], 0), (a["xr"], r1), (120, 120, 120), 1)
    cv2.line(mp, (0, r1), (w, r1), (0, 230, 230), 1)
    if a["ok"]:
        cv2.line(mp, (int(a["b"]), 0), (int(a["m"] * r1 + a["b"]), r1), (0, 0, 255), 2)
    cv2.rectangle(mp, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(mp, "MASK", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return mp


class Cam(Node):
    def __init__(self):
        super().__init__("tuner_dashboard")
        self.msg = None
        self.create_subscription(Image, TOPIC, self._cb, 5)

    def _cb(self, m):
        self.msg = m


def vision_loop(cam):
    global _latest_jpeg
    frames = 0
    t0 = time.time()
    while True:
        msg = cam.msg
        if msg is None:
            time.sleep(0.05)
            continue
        cam.msg = None
        try:
            bgr = _decode(msg)
        except Exception as e:
            with _lock:
                _status.update(ok=False, why=str(e))
            time.sleep(0.05)
            continue
        p = read_params()
        a = analyze(bgr, p)
        out = annotate(bgr, a, p)
        out = np.hstack([out, mask_panel(a, bgr.shape)])
        okj, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frames += 1
        el = time.time() - t0
        if el > 2.0:
            fps = frames / el
            frames = 0
            t0 = time.time()
        else:
            fps = _status.get("fps", 0.0)
        with _lock:
            if okj:
                _latest_jpeg = jpg.tobytes()
            _status.update(ok=bool(a["ok"]), why=a.get("why", ""), fps=round(fps, 1),
                           n=int(a["n"]), resid=round(a.get("resid", 0.0), 1),
                           angle=round(a.get("angle", 0.0), 1),
                           lateral=round(a.get("lateral", 0.0), 3))


PAGE = """<!DOCTYPE html>
<html><head><title>line tuner</title>
<style>
body { background:#111; color:#ddd; font-family:monospace; margin:12px; }
#wrap { display:flex; gap:16px; flex-wrap:wrap; }
img { border:1px solid #444; max-width:100%; }
.row { margin:7px 0; }
.row label { display:inline-block; width:130px; }
.row input[type=range] { width:220px; vertical-align:middle; }
.row span { display:inline-block; width:70px; text-align:right; }
#status { margin:8px 0; color:#8f8; }
button { background:#333; color:#ddd; border:1px solid #555; padding:4px 10px; cursor:pointer; }
</style></head><body>
<div id="status">connecting...</div>
<div id="wrap">
  <div><img id="feed" src="/stream" width="980"></div>
  <div id="sliders"></div>
</div>
<button onclick="loadParams()">reload from file</button>
<script>
async function loadParams() {
  const r = await fetch('/params'); const d = await r.json();
  const box = document.getElementById('sliders'); box.innerHTML = '';
  for (const k of Object.keys(d.limits)) {
    const [lo, hi] = d.limits[k]; const step = d.steps[k]; const v = d.values[k];
    const row = document.createElement('div'); row.className = 'row';
    row.innerHTML = `<label>${k}</label>
      <input type="range" min="${lo}" max="${hi}" step="${step}" value="${v}"
        oninput="this.nextElementSibling.textContent=this.value"
        onchange="setParam('${k}', this.value)">
      <span>${v}</span>`;
    box.appendChild(row);
  }
}
async function setParam(k, v) {
  await fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify({[k]: parseFloat(v)})});
}
async function poll() {
  try {
    const r = await fetch('/status'); const s = await r.json();
    document.getElementById('status').textContent =
      (s.ok ? `LINE  ang ${s.angle}  lat ${s.lateral}  resid ${s.resid}  n ${s.n}`
            : `NO LINE  ${s.why}`) + `   |   ${s.fps} fps`;
    document.getElementById('status').style.color = s.ok ? '#8f8' : '#f88';
  } catch (e) {}
  setTimeout(poll, 500);
}
loadParams(); poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html", PAGE.encode())
        elif self.path == "/params":
            body = json.dumps({"values": read_params(), "limits": LIMITS,
                               "steps": STEPS}).encode()
            self._send(200, "application/json", body)
        elif self.path == "/status":
            with _lock:
                body = json.dumps(_status).encode()
            self._send(200, "application/json", body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpg = _latest_jpeg
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                         + jpg + b"\r\n")
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/set":
            n = int(self.headers.get("Content-Length", 0))
            try:
                changes = json.loads(self.rfile.read(n))
                applied = write_params(changes)
                self._send(200, "application/json", json.dumps(applied).encode())
                for k, v in applied.items():
                    print(f"[set] {k} = {v}")
            except Exception as e:
                self._send(400, "text/plain", str(e).encode())
        else:
            self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    rclpy.init()
    cam = Cam()
    ex = SingleThreadedExecutor()
    ex.add_node(cam)
    threading.Thread(target=ex.spin, daemon=True).start()
    threading.Thread(target=vision_loop, args=(cam,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"tuner up: http://<pi-ip>:{PORT}   (topic {TOPIC}, tune file {TUNE_FILE})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass