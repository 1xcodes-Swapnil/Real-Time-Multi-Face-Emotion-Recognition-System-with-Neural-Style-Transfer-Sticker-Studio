"""
dashboard.py  —  MultiFace Integrated Dashboard
Drop this file into your project root alongside ui_app.py and run:

    py -3.11 dashboard.py          (uses your existing TF model, AnimeGAN, etc.)
    python   dashboard.py          (demo/simulation mode if deps missing)

Open:  http://localhost:5050

Integrates with your existing project files:
    face_detector.py · preprocessing.py · grad_cam.py · animegan_inference.py
    models/saved_model/emotion_model.h5
"""

import os, sys, io, base64, glob, time, threading, tempfile, zipfile, json
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, render_template_string, request, jsonify, send_file, abort, Response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ── Dirs (reuse same layout as ui_app.py) ─────────────────────────────────────
STICKER_DIR  = os.path.join(_ROOT, "stickers", "custom")
NOTO_DIR     = os.path.join(_ROOT, "stickers", "noto")
PHOTOS_DIR   = os.path.join(_ROOT, "photos")
VIDEOS_DIR   = os.path.join(_ROOT, "videos")
SAVE_DIR     = os.path.join(_ROOT, "saved_emojis")
for _d in [STICKER_DIR, NOTO_DIR, PHOTOS_DIR, VIDEOS_DIR, SAVE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Optional heavy deps (same try/except pattern as ui_app.py) ─────────────────
try:
    import cv2, numpy as np
    _CV2_OK = True
except Exception as _e:
    print(f"[dash] cv2 unavailable: {_e}"); _CV2_OK = False

try:
    import tensorflow as tf
    _TF_OK = True
except Exception as _e:
    print(f"[dash] TF unavailable: {_e}"); _TF_OK = False

try:
    from animegan_inference import apply_anime_style, is_available as _anime_avail
    _ANIME_OK = _anime_avail()
except Exception as _e:
    print(f"[dash] AnimeGAN: {_e}"); _ANIME_OK = False
    def apply_anime_style(f, **k): return f

try:
    from face_detector  import FaceDetector
    from preprocessing  import preprocess_face, preprocess_face_rgb, EMOTION_LABELS
    from grad_cam       import GradCAM
    _PROJ_OK = True
except Exception as _e:
    print(f"[dash] Project modules: {_e}. Running in demo mode.")
    _PROJ_OK = False
    EMOTION_LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

try:
    import requests as req_lib; _REQ_OK = True
except Exception: _REQ_OK = False

try:
    from PIL import Image, ImageDraw; _PIL_OK = True
except Exception: _PIL_OK = False

# ── Constants matching ui_app.py ──────────────────────────────────────────────
MODEL_PATH = os.path.join(_ROOT, "models", "saved_model", "emotion_model.h5")
CNN_PATH   = os.path.join(_ROOT, "models", "saved_model", "emotion_cnn.h5")
GIPHY_KEY  = os.environ.get("GIPHY_KEY", "jWRZPFqFvThUs6Qb6TCzNawzmdsftmnQ")
VIDEO_FPS  = 15
GIPHY_Q    = {
    "angry":"angry emoji","disgust":"disgusted reaction","fear":"scared emoji",
    "happy":"happy dancing emoji","neutral":"neutral face",
    "sad":"sad crying emoji","surprise":"surprised emoji",
}
NOTO_CODES = {
    "angry":["1f620","1f621"],"disgust":["1f922","1f92e"],
    "fear":["1f628","1f631"],"happy":["1f600","1f601","1f603","1f929"],
    "neutral":["1f610","1f636"],"sad":["1f622","1f62d"],
    "surprise":["1f62e","1f92f"],
}

# ── Shared live state ─────────────────────────────────────────────────────────
_state_lock   = threading.Lock()
_live_state   = {
    "emotion": "neutral", "confidence": 0.0,
    "probs": [0.14,0.02,0.05,0.60,0.12,0.05,0.02],
    "fps": 0, "anime_on": False, "gc_on": False,
}

# ── Background inference thread ───────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running   = True
        self.anime_on  = False
        self.gc_on     = False
        self._smoother = []
        self._fps_times = []

    def run(self):
        if not (_CV2_OK and _PROJ_OK and _TF_OK):
            print("[dash] Inference worker: demo mode (no camera/model)")
            self._sim_loop(); return

        model, gradcam = None, None
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model(np.zeros((1,) + tuple(model.input_shape[1:]), dtype=np.float32), training=False)
            print(f"[dash] Model loaded {model.input_shape}")
        except Exception as e:
            print(f"[dash] Model load failed: {e}")

        if model is not None:
            try:
                cnn = tf.keras.models.load_model(CNN_PATH, compile=False) \
                      if os.path.exists(CNN_PATH) \
                      else (model if model.input_shape[-1] == 1 else None)
                if cnn:
                    gradcam = GradCAM(cnn, "res_conv4")
                    print("[dash] GradCAM ready")
            except Exception as e:
                print(f"[dash] GradCAM: {e}")

        detector = FaceDetector(confidence_threshold=0.5)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        last_pred = 0.0
        last_hmt  = 0.0
        last_hm   = None

        while self.running:
            cap.grab()
            ret, frame = cap.read()
            if not ret: time.sleep(0.01); continue
            frame = cv2.flip(frame, 1)
            now   = time.time()

            faces = detector.detect(frame)
            if now - last_pred > 0.3 and faces and model is not None:
                fx, fy = max(0, faces[0][0]), max(0, faces[0][1])
                fw = min(faces[0][2], frame.shape[1] - fx)
                fh = min(faces[0][3], frame.shape[0] - fy)
                if fw > 0 and fh > 0:
                    crop = frame[fy:fy+fh, fx:fx+fw]
                    try:
                        is_rgb = model.input_shape[-1] == 3
                        img    = preprocess_face_rgb(crop) if is_rgb else preprocess_face(crop)
                        preds  = model(img, training=False)[0].numpy()
                        self._smoother.append(preds)
                        if len(self._smoother) > 8: self._smoother.pop(0)
                        smooth = np.mean(self._smoother, axis=0)
                        idx    = int(np.argmax(smooth))
                        with _state_lock:
                            _live_state["emotion"]    = EMOTION_LABELS[idx]
                            _live_state["confidence"] = float(smooth[idx])
                            _live_state["probs"]      = smooth.tolist()
                    except Exception as e:
                        print(f"[dash] Inference err: {e}")
                    last_pred = now

                    if self.gc_on and gradcam and now - last_hmt > 1.5:
                        try:
                            hm = gradcam.compute(preprocess_face(crop), idx)
                            if hm.max() > 0: last_hm = hm
                            last_hmt = now
                        except: pass

            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t < 1.0]
            with _state_lock:
                _live_state["fps"]      = len(self._fps_times)
                _live_state["anime_on"] = self.anime_on
                _live_state["gc_on"]    = self.gc_on

        cap.release()

    def _sim_loop(self):
        import math, random
        probs = [0.14,0.02,0.05,0.60,0.12,0.05,0.02]
        t = 0
        while self.running:
            probs = [max(0.01, min(0.99, p + random.gauss(0, 0.04))) for p in probs]
            s = sum(probs); probs = [p/s for p in probs]
            idx = probs.index(max(probs))
            with _state_lock:
                _live_state["emotion"]    = EMOTION_LABELS[idx]
                _live_state["confidence"] = probs[idx]
                _live_state["probs"]      = probs[:]
                _live_state["fps"]        = int(24 + 4*math.sin(t))
            t += 0.1; time.sleep(1.0)

_worker = InferenceWorker()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _decode_frame(b64_str):
    if not _CV2_OK: return None
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _apply_shape_mask(img_bgr, shape, size=512):
    if not (_CV2_OK and _PIL_OK): return None
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).convert("RGBA")
    if shape == "circle":
        m = Image.new("L", (size, size), 0)
        ImageDraw.Draw(m).ellipse([0,0,size,size], fill=255)
        pil.putalpha(m)
    elif shape == "square":
        m = Image.new("L", (size, size), 0)
        pad, r = int(size*.04), int(size*.08)
        ImageDraw.Draw(m).rounded_rectangle([pad,pad,size-pad,size-pad], radius=r, fill=255)
        pil.putalpha(m)
    buf = io.BytesIO(); pil.save(buf, "PNG"); buf.seek(0)
    return buf

def _sticker_list():
    pngs = sorted(glob.glob(os.path.join(STICKER_DIR, "*.png")), reverse=True)
    out = []
    for p in pngs:
        sid   = os.path.basename(p)[:-4]
        mtime = os.path.getmtime(p)
        shape = next((s for s in ("circle","square","raw") if s in sid), "circle")
        out.append({"id": sid, "date": datetime.fromtimestamp(mtime).strftime("%b %d %H:%M"), "shape": shape})
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SSE
# ══════════════════════════════════════════════════════════════════════════════
def _sse_stream():
    while True:
        with _state_lock:
            payload = json.dumps(_live_state)
        yield f"data: {payload}\n\n"
        time.sleep(0.4)

@app.route("/api/stream")
def api_stream():
    return Response(_sse_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ══════════════════════════════════════════════════════════════════════════════
# EMOTION STATE
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/emotion")
def api_emotion():
    with _state_lock:
        return jsonify(dict(_live_state))

@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    key = request.json.get("key")
    if key == "anime":
        _worker.anime_on = not _worker.anime_on
        return jsonify({"anime_on": _worker.anime_on})
    elif key == "gcam":
        _worker.gc_on = not _worker.gc_on
        return jsonify({"gc_on": _worker.gc_on})
    return jsonify({"error": "unknown key"}), 400

# ══════════════════════════════════════════════════════════════════════════════
# STICKER API
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/capture", methods=["POST"])
def api_capture():
    data      = request.json or {}
    b64       = data.get("image", "")
    canvas_b64= data.get("canvas", "")
    shape     = data.get("shape", "circle")
    anime     = data.get("anime", False)

    if not b64:
        return jsonify({"ok": False, "error": "No image"})

    ts  = int(time.time() * 1000)
    sid = f"{shape}_{ts}"
    out_path = os.path.join(STICKER_DIR, f"{sid}.png")

    if canvas_b64:
        try:
            raw = base64.b64decode(canvas_b64)
            with open(out_path, "wb") as f: f.write(raw)
            return jsonify({"ok": True, "id": sid})
        except Exception as e:
            pass

    if not (_CV2_OK and _PIL_OK):
        return jsonify({"ok": False, "error": "cv2/PIL not installed on server"})

    frame = _decode_frame(b64)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode image"})

    if anime and _ANIME_OK:
        try:
            out = apply_anime_style(frame, size=512)
            if out is not None:
                frame = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            print(f"[dash/anime] {e}")

    buf = _apply_shape_mask(frame, shape)
    if buf is None:
        return jsonify({"ok": False, "error": "Shape masking failed"})

    with open(out_path, "wb") as f: f.write(buf.getvalue())
    return jsonify({"ok": True, "id": sid})

@app.route("/api/stickers")
def api_stickers():
    return jsonify({"stickers": _sticker_list()})

@app.route("/sticker/<sid>")
def serve_sticker(sid):
    sid  = os.path.basename(sid)
    fmt  = request.args.get("fmt", "png")
    ext  = "webp" if fmt == "webp" else "png"
    path = os.path.join(STICKER_DIR, f"{sid}.{ext}")
    if not os.path.exists(path):
        path = os.path.join(STICKER_DIR, f"{sid}.png")
    if not os.path.exists(path): abort(404)
    return send_file(path, mimetype="image/png",
                     as_attachment=(fmt in ("png","webp")),
                     download_name=f"sticker_{sid}.{ext}")

@app.route("/api/sticker/<sid>", methods=["DELETE"])
def delete_sticker(sid):
    sid = os.path.basename(sid)
    for ext in ("png","webp"):
        p = os.path.join(STICKER_DIR, f"{sid}.{ext}")
        if os.path.exists(p): os.remove(p)
    return jsonify({"ok": True})

@app.route("/api/pack", methods=["POST"])
def api_pack():
    ids   = request.json.get("ids", [])
    paths = [os.path.join(STICKER_DIR, f"{os.path.basename(s)}.png")
             for s in ids if os.path.exists(os.path.join(STICKER_DIR, f"{os.path.basename(s)}.png"))]
    if not paths: return "No stickers found", 404
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths: zf.write(p, os.path.basename(p))
    tmp.close()
    return send_file(tmp.name, mimetype="application/zip",
                     as_attachment=True, download_name="sticker_pack.zip")

@app.route("/api/save_photo", methods=["POST"])
def api_save_photo():
    data  = request.json or {}
    b64   = data.get("image", "")
    fname = os.path.basename(data.get("filename", f"photo_{int(time.time())}.jpg"))
    if not b64: return jsonify({"ok": False})
    try:
        raw = base64.b64decode(b64)
        with open(os.path.join(PHOTOS_DIR, fname), "wb") as f: f.write(raw)
        return jsonify({"ok": True, "saved": fname})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# GIPHY + NOTO PROXIES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/giphy/<emotion>")
def api_giphy(emotion):
    if not _REQ_OK: return jsonify({"url": None})
    offset = int(request.args.get("offset", 0))
    q      = GIPHY_Q.get(emotion.lower(), f"{emotion} emoji")
    try:
        r    = req_lib.get("https://api.giphy.com/v1/gifs/search",
                           params={"q":q,"api_key":GIPHY_KEY,"limit":10,"rating":"g"}, timeout=8)
        data = r.json().get("data", [])
        if not data: return jsonify({"url": None})
        url  = data[offset % len(data)]["images"]["original"]["url"].split("?")[0]
        return jsonify({"url": url})
    except Exception as e:
        return jsonify({"url": None, "error": str(e)})

@app.route("/api/noto/<emotion>")
def api_noto(emotion):
    idx   = int(request.args.get("idx", 0))
    codes = NOTO_CODES.get(emotion.lower(), ["1f610"])
    code  = codes[idx % len(codes)]
    folder = os.path.join(NOTO_DIR, emotion.lower()); os.makedirs(folder, exist_ok=True)
    path   = os.path.join(folder, f"{code}.gif")
    if not os.path.exists(path) or os.path.getsize(path) < 500:
        if not _REQ_OK: abort(503)
        try:
            r = req_lib.get(f"https://fonts.gstatic.com/s/e/notoemoji/latest/{code}/512.gif", timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as f: f.write(r.content)
        except Exception as e: abort(500)
    if not os.path.exists(path): abort(404)
    return send_file(path, mimetype="image/gif")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════
DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MultiFace Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700;800&family=DM+Mono:wght@300;400;500&family=Nunito:wght@400;700;900&family=Fredoka+One&family=Black+Han+Sans&family=Permanent+Marker&family=Dancing+Script:wght@700&display=swap" rel="stylesheet">
<style>
/* ══ RESET & TOKENS ══════════════════════════════════════════════════════════ */
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:   #080a10;
  --bg2:  #0d0f18;
  --bg3:  #131621;
  --bg4:  #191d2e;
  --line: #1e2238;
  --line2:#262b42;
  --hi:   #e94560;
  --hi2:  #5b6af0;
  --hi3:  #00c896;
  --gold: #f0b429;
  --text: #dde1f0;
  --sub:  #5c6080;
  --sub2: #3a3e58;
  --mono: 'DM Mono', monospace;
  --sans: 'Nunito', sans-serif;
  --head: 'Oxanium', sans-serif;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font-family:var(--sans);font-size:13px;line-height:1.4}

/* ══ 3-COL GRID  1 : 2 : 1 ══════════════════════════════════════════════════ */
.shell{display:grid;grid-template-columns:1fr 2fr 1fr;
  grid-template-rows:48px 1fr;height:100vh;overflow:hidden;min-height:0}

/* ══ TOP BAR ════════════════════════════════════════════════════════════════ */
.topbar{
  grid-column:1/-1;grid-row:1;
  display:flex;align-items:center;gap:10px;padding:0 16px;
  background:var(--bg2);border-bottom:1px solid var(--line);
}
.topbar .brand{
  font-family:var(--head);font-size:15px;font-weight:700;
  letter-spacing:.5px;white-space:nowrap;color:var(--text);
  display:flex;align-items:center;gap:8px;
}
.brand-dot{width:7px;height:7px;border-radius:50%;background:var(--hi);
  box-shadow:0 0 8px var(--hi);animation:hb 2s infinite}
@keyframes hb{0%,100%{opacity:1}50%{opacity:.3}}
.topbar-div{width:1px;height:24px;background:var(--line2);margin:0 4px}
.tb-stat{display:flex;flex-direction:column;line-height:1.2}
.tb-stat .v{font-family:var(--mono);font-size:11px;color:var(--hi3)}
.tb-stat .k{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--sub)}
.tb-right{margin-left:auto;display:flex;align-items:center;gap:6px}
.chip{padding:3px 11px;border-radius:20px;font-size:10px;font-weight:700;
  border:1px solid var(--line2);background:var(--bg3);color:var(--sub);
  cursor:pointer;transition:.18s;white-space:nowrap;letter-spacing:.3px}
.chip:hover{border-color:var(--sub2)}
.chip.on-red{background:#e9456014;border-color:var(--hi);color:var(--hi)}
.chip.on-blue{background:#5b6af014;border-color:var(--hi2);color:var(--hi2)}
.chip.on-teal{background:#00c89614;border-color:var(--hi3);color:var(--hi3)}
.rec-blink{width:6px;height:6px;border-radius:50%;background:var(--hi);
  animation:bk 1s infinite;display:none}
.rec-blink.on{display:inline-block}
@keyframes bk{0%,100%{opacity:1}50%{opacity:0}}

/* ══ LEFT SIDEBAR ════════════════════════════════════════════════════════════ */
.left{grid-column:1;grid-row:2;background:var(--bg2);border-right:1px solid var(--line);
  overflow-y:auto;display:flex;flex-direction:column;min-height:0}

/* ══ CAMERA CENTRE ═══════════════════════════════════════════════════════════ */
.centre{grid-column:2;grid-row:2;position:relative;background:#000;overflow:hidden;min-height:0;display:flex;flex-direction:column}
.centre video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
  transform:scaleX(-1);transform-origin:center center}
.centre canvas.vover{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
canvas.vover{pointer-events:none}
.cam-hud{position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(transparent,rgba(8,10,16,.95));
  padding:18px 16px 12px;display:flex;align-items:center;gap:10px}
.hud-spacer{flex:1}
.shape-pills{display:flex;gap:6px}
.sp{padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;cursor:pointer;
  transition:.15s;border:1px solid var(--line2);background:var(--bg3);color:var(--sub)}
.sp.active{border-color:var(--hi);color:var(--hi);background:#e9456012}
.sp:hover{border-color:var(--sub2)}
.shutter{width:52px;height:52px;border-radius:50%;border:3px solid rgba(255,255,255,.5);
  background:transparent;cursor:pointer;position:relative;
  transition:.12s;flex-shrink:0}
.shutter::after{content:'';position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);width:34px;height:34px;border-radius:50%;
  background:rgba(255,255,255,.9);transition:.12s}
.shutter:hover::after{background:var(--hi)}
.shutter:active{transform:scale(.9)}
.cam-info{text-align:right;font-family:var(--mono);font-size:9px;color:var(--sub);line-height:1.7}
.cam-badges{position:absolute;top:12px;left:12px;display:flex;gap:7px}
.cbadge{padding:3px 9px;border-radius:20px;font-size:9px;font-weight:700;
  letter-spacing:.5px;backdrop-filter:blur(10px)}
.cbadge.live{background:rgba(233,69,96,.18);border:1px solid var(--hi);color:var(--hi)}
.cbadge.anime,.cbadge.gcam{display:none}
.cbadge.anime.on,.cbadge.gcam.on{display:inline-block}
.cbadge.anime{background:rgba(91,106,240,.18);border:1px solid var(--hi2);color:var(--hi2)}
.cbadge.gcam{background:rgba(0,200,150,.18);border:1px solid var(--hi3);color:var(--hi3)}
.countdown{position:absolute;inset:0;display:none;align-items:center;
  justify-content:center;background:rgba(0,0,0,.55);backdrop-filter:blur(3px);
  font-family:var(--head);font-size:130px;font-weight:800;color:#fff;
  text-shadow:0 0 50px var(--hi)}
.countdown.on{display:flex;animation:pop .45s ease}
@keyframes pop{from{transform:scale(1.5)}to{transform:scale(1)}}

/* ══ RIGHT SIDEBAR — TABBED ══════════════════════════════════════════════════ */
.right{grid-column:3;grid-row:2;background:var(--bg2);border-left:1px solid var(--line);
  display:flex;flex-direction:column;min-height:0;overflow:hidden}

/* ── Tab bar ── */
.tab-bar{
  display:flex;border-bottom:1px solid var(--line);
  background:var(--bg2);flex-shrink:0;
}
.tab-btn{
  flex:1;padding:11px 6px;font-family:var(--head);font-size:11px;font-weight:700;
  letter-spacing:.4px;text-align:center;cursor:pointer;
  border:none;background:transparent;color:var(--sub);
  border-bottom:2px solid transparent;transition:.15s;
  display:flex;align-items:center;justify-content:center;gap:5px;
}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--hi2);border-bottom-color:var(--hi2)}
.tab-btn .ticon{font-size:13px}

/* ── Tab panels ── */
.tab-panel{display:none;flex:1;overflow-y:auto;flex-direction:column;min-height:0}
.tab-panel.active{display:flex}

/* ══ SHARED SECTION CHROME ═══════════════════════════════════════════════════ */
.sec{padding:12px 13px 10px;border-bottom:1px solid var(--line)}
.sec-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}
.sec-title{font-size:8px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--sub)}
.tag{font-size:8px;font-weight:700;padding:2px 7px;border-radius:10px;letter-spacing:.3px}
.tag-teal{background:#00c89618;border:1px solid var(--hi3);color:var(--hi3)}
.tag-blue{background:#5b6af018;border:1px solid var(--hi2);color:var(--hi2)}
.tag-red{background:#e9456018;border:1px solid var(--hi);color:var(--hi)}

/* ── emotion big label ── */
.emo-hero{display:flex;flex-direction:column;align-items:center;padding:10px 0 6px;gap:4px}
.emo-name{font-family:var(--head);font-size:28px;font-weight:800;color:var(--hi);
  transition:color .3s;text-align:center;line-height:1}
.emo-pct{font-family:var(--mono);font-size:11px;color:var(--sub)}

/* ── bars ── */
.bar-row{display:flex;align-items:center;gap:5px;margin-bottom:4px}
.bar-lbl{width:46px;font-size:9px;color:var(--sub);text-align:right;flex-shrink:0;
  font-family:var(--mono)}
.bar-track{flex:1;height:5px;background:var(--bg4);border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .4s ease}
.bar-pct{width:28px;font-size:9px;color:var(--sub2);text-align:right;font-family:var(--mono)}

/* ── ctrl buttons ── */
.ctrl-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.cbtn{display:flex;align-items:center;gap:5px;padding:7px 9px;border-radius:7px;
  border:1px solid var(--line2);background:var(--bg3);color:var(--sub);
  font-family:var(--sans);font-size:10px;font-weight:700;cursor:pointer;
  transition:.14s;white-space:nowrap}
.cbtn:hover{border-color:var(--sub2);color:var(--text)}
.cbtn.act-red{border-color:var(--hi);background:#e9456012;color:var(--hi)}
.cbtn.act-blue{border-color:var(--hi2);background:#5b6af012;color:var(--hi2)}
.cbtn.act-teal{border-color:var(--hi3);background:#00c89612;color:var(--hi3)}
.cbtn.wide{grid-column:span 2}
.kbd{margin-left:auto;font-size:8px;background:var(--bg4);border:1px solid var(--line2);
  border-radius:3px;padding:1px 4px;font-family:var(--mono);color:var(--sub2)}

/* ── media boxes (giphy/noto) ── */
.media-box{background:var(--bg3);border:1px solid var(--line);border-radius:8px;
  height:124px;display:flex;align-items:center;justify-content:center;overflow:hidden}
.media-box img{max-width:100%;max-height:100%;object-fit:contain}
.media-hint{font-size:10px;color:var(--sub2);text-align:center;padding:8px}

/* ── status ── */
.statusbar{padding:7px 13px;font-family:var(--mono);font-size:9px;color:var(--sub);
  border-top:1px solid var(--line);margin-top:auto;background:var(--bg2);min-height:28px;flex-shrink:0}

/* ══ WEBCAM TAB — extra controls ════════════════════════════════════════════ */
.cam-preview-thumb{
  margin:10px 11px 0;border-radius:9px;overflow:hidden;
  border:1px solid var(--line);aspect-ratio:16/9;background:#000;position:relative;
}
.cam-preview-thumb video{
  width:100%;height:100%;object-fit:cover;transform:scaleX(-1);
}
.cam-preview-thumb .live-dot{
  position:absolute;top:7px;right:8px;
  display:flex;align-items:center;gap:4px;
  background:rgba(8,10,16,.7);border-radius:20px;padding:2px 7px;
  font-size:8px;font-weight:700;color:var(--hi);letter-spacing:.5px;
}
.cam-preview-thumb .live-dot::before{
  content:'';width:5px;height:5px;border-radius:50%;
  background:var(--hi);animation:bk 1s infinite;
}

/* rec timer in webcam tab */
.rec-bar{
  display:flex;align-items:center;justify-content:space-between;
  padding:5px 11px;background:var(--bg3);border-bottom:1px solid var(--line);
  font-family:var(--mono);font-size:9px;color:var(--sub);
}
.rec-bar .rtime{color:var(--hi);font-size:11px;font-weight:700;display:none}
.rec-bar .rtime.on{display:inline}

/* ══ STICKER STUDIO TAB ══════════════════════════════════════════════════════ */
.studio-hd{padding:10px 13px 8px;border-bottom:1px solid var(--line);
  font-family:var(--head);font-size:13px;font-weight:700;
  display:flex;align-items:center;gap:8px;letter-spacing:.3px;flex-shrink:0}

.stk-preview{padding:10px 11px;border-bottom:1px solid var(--line)}
.tool-lbl{font-size:8px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
  color:var(--sub);margin-bottom:6px}
#stkCv{width:100%;border-radius:10px;border:1px solid var(--line2);display:block;
  cursor:crosshair;
  background:repeating-conic-gradient(#ffffff05 0% 25%,transparent 0% 50%) 0 0/12px 12px}

.font-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px}
.fbtn{padding:6px 3px;border-radius:6px;border:1px solid var(--line);background:var(--bg3);
  cursor:pointer;text-align:center;transition:.13s;color:var(--sub);line-height:1.2}
.fbtn:hover{border-color:var(--sub2)}
.fbtn.active{border-color:var(--hi2);background:#5b6af010;color:var(--hi2)}
.fbtn .fsamp{font-size:14px;font-weight:700;display:block;margin-bottom:1px}
.fbtn .fname{font-size:8px;letter-spacing:.4px}

.fx-row{display:flex;gap:4px;flex-wrap:wrap}
.fxtog{padding:4px 9px;border-radius:20px;border:1px solid var(--line);background:var(--bg3);
  color:var(--sub);font-size:9px;font-weight:700;cursor:pointer;transition:.13s}
.fxtog:hover{border-color:var(--sub2)}
.fxtog.on{border-color:var(--gold);background:#f0b42912;color:var(--gold)}

.pal-row{display:flex;gap:5px;flex-wrap:wrap;align-items:center}
.swatch{width:20px;height:20px;border-radius:50%;cursor:pointer;
  transition:.1s;border:2px solid transparent;flex-shrink:0}
.swatch.active{border-color:#fff;transform:scale(1.18)}
.swatch:hover{transform:scale(1.1)}
.swatch-custom{width:26px;height:20px;border-radius:5px;cursor:pointer;
  border:1px solid var(--line);overflow:hidden;flex-shrink:0}
.swatch-custom input[type=color]{width:38px;height:28px;border:none;
  transform:translate(-6px,-4px);cursor:pointer}

.sl-row{display:flex;align-items:center;gap:7px}
.sl-row label{font-size:9px;color:var(--sub);width:48px;flex-shrink:0}
.sl-row input[type=range]{flex:1;accent-color:var(--hi2)}
.sl-row .slv{font-family:var(--mono);font-size:9px;color:var(--sub);width:22px;text-align:right}

.txt-row{display:flex;gap:5px}
.txt-inp{flex:1;background:var(--bg4);border:1px solid var(--line2);border-radius:7px;
  padding:6px 9px;color:var(--text);font-family:var(--sans);font-size:11px;
  outline:none;transition:.14s}
.txt-inp:focus{border-color:var(--hi2)}
.txt-inp::placeholder{color:var(--sub2)}
.add-btn{padding:6px 13px;border-radius:7px;border:none;background:var(--hi2);
  color:#fff;font-family:var(--sans);font-size:11px;font-weight:700;cursor:pointer;transition:.14s}
.add-btn:hover{opacity:.84}

.shape-row{display:flex;gap:5px}
.shbtn{flex:1;padding:7px 0;border-radius:7px;border:1px solid var(--line);
  background:var(--bg3);color:var(--sub);font-size:10px;font-weight:700;
  cursor:pointer;transition:.13s;text-align:center}
.shbtn:hover{border-color:var(--sub2)}
.shbtn.active{border-color:var(--hi);background:#e9456010;color:var(--hi)}

.act-row{display:flex;gap:5px;padding:9px 11px;border-bottom:1px solid var(--line);flex-shrink:0}
.abtn{flex:1;padding:8px 0;border-radius:7px;font-family:var(--sans);font-size:10px;
  font-weight:700;cursor:pointer;transition:.14s;border:1px solid var(--line2);
  background:var(--bg3);color:var(--text)}
.abtn:hover{border-color:var(--sub2)}
.abtn.primary{background:linear-gradient(135deg,var(--hi),#b02040);border:none;
  color:#fff;box-shadow:0 3px 12px rgba(233,69,96,.35)}
.abtn.primary:hover{opacity:.88;transform:translateY(-1px)}
.abtn:disabled{opacity:.4;cursor:not-allowed;transform:none!important}

.gal-hd{padding:9px 11px 5px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
.gal-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;padding:0 11px 10px}
.gcard{aspect-ratio:1;border-radius:7px;background:var(--bg3);border:1px solid var(--line);
  overflow:hidden;cursor:pointer;transition:.14s;position:relative}
.gcard:hover{border-color:var(--hi2);transform:scale(1.04)}
.gcard img{width:100%;height:100%;object-fit:contain;
  background:repeating-conic-gradient(#ffffff04 0% 25%,transparent 0% 50%) 0 0/9px 9px}
.gcard .gdel{position:absolute;top:3px;right:3px;width:15px;height:15px;border-radius:50%;
  background:rgba(233,69,96,.85);color:#fff;font-size:8px;display:none;
  align-items:center;justify-content:center;cursor:pointer;font-weight:700}
.gcard:hover .gdel{display:flex}
.gal-empty{grid-column:span 3;padding:28px;text-align:center;color:var(--sub2);font-size:10px}

.pack-sec{padding:9px 11px;border-top:1px solid var(--line);background:var(--bg2);flex-shrink:0}
.pack-row{display:flex;align-items:baseline;gap:5px;margin-bottom:5px}
.pack-n{font-family:var(--head);font-size:20px;font-weight:800;color:var(--hi2)}
.pack-sub{font-size:9px;color:var(--sub)}
.pack-dl{width:100%;padding:9px;border-radius:8px;border:1px solid var(--hi2);
  background:transparent;color:var(--hi2);font-family:var(--sans);font-size:11px;
  font-weight:700;cursor:pointer;transition:.14s}
.pack-dl:hover{background:#5b6af014}
.pack-dl:disabled{opacity:.35;cursor:not-allowed}

/* sticker tool inner sections */
.stk-sec{padding:9px 11px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:7px}

/* ══ TOAST ═══════════════════════════════════════════════════════════════════ */
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(18px);
  background:var(--bg3);border:1px solid var(--line2);padding:8px 18px;border-radius:30px;
  font-size:11px;opacity:0;transition:.22s;z-index:9999;pointer-events:none;white-space:nowrap}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast.ok{border-color:var(--hi3);color:var(--hi3)}
.toast.err{border-color:var(--hi);color:var(--hi)}

/* scrollbars */
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--line2);border-radius:2px}
</style>
</head>
<body>
<div class="shell">

<!-- ══ TOPBAR ══════════════════════════════════════════════════════════════ -->
<div class="topbar">
  <div class="brand">
    <span class="brand-dot"></span>MultiFace Studio
  </div>
  <div class="topbar-div"></div>
  <div class="tb-stat">
    <div class="v" id="tb-fps">-- fps</div>
    <div class="k">inference</div>
  </div>
  <div class="topbar-div"></div>
  <div class="tb-stat">
    <div class="v" id="tb-emo" style="color:var(--hi)">Neutral</div>
    <div class="k">emotion</div>
  </div>
  <div class="tb-right">
    <span class="rec-blink" id="rec-blink"></span>
    <span id="rec-lbl" style="font-family:var(--mono);font-size:9px;color:var(--hi);display:none">REC</span>
    <div class="chip on-teal">● LIVE</div>
    <div class="chip" id="chip-anime" onclick="toggleFeature('anime')">AnimeGAN OFF</div>
    <div class="chip" id="chip-gcam"  onclick="toggleFeature('gcam')">Grad-CAM</div>
  </div>
</div>

<!-- ══ LEFT SIDEBAR — EMOTION ENGINE ═══════════════════════════════════════ -->
<div class="left">

  <div class="sec">
    <div class="sec-hd"><span class="sec-title">Detected Emotion</span></div>
    <div class="emo-hero">
      <div class="emo-name" id="emo-name">Neutral</div>
      <div class="emo-pct"  id="emo-pct">0% confidence</div>
    </div>
  </div>

  <div class="sec">
    <div class="sec-title" style="margin-bottom:8px">Confidence</div>
    <div id="bars"></div>
  </div>

  <div class="sec">
    <div class="sec-hd">
      <span class="sec-title">Giphy Sticker</span>
      <span class="tag tag-teal">LIVE</span>
    </div>
    <div class="media-box" id="giphy-box"><div class="media-hint">Loads on emotion detect</div></div>
  </div>

  <div class="sec">
    <div class="sec-hd">
      <span class="sec-title">Noto Emoji</span>
      <span class="tag tag-blue">Noto</span>
    </div>
    <div class="media-box" id="noto-box"><div class="media-hint">Press N to load</div></div>
    <button class="cbtn wide" style="margin-top:6px;width:100%;justify-content:center" onclick="nextNoto()">
      ↻ Next Variation <span class="kbd">N</span>
    </button>
  </div>

  <div class="statusbar" id="statusbar">Connecting to inference…</div>
</div>

<!-- ══ CAMERA CENTRE ═══════════════════════════════════════════════════════ -->
<div class="centre">
  <video id="cam-vid" autoplay playsinline muted></video>
  <canvas id="cam-cv" class="vover"></canvas>

  <!-- No-camera overlay (shown when getUserMedia fails) -->
  <div id="no-cam-overlay" style="display:none;position:absolute;inset:0;
    align-items:center;justify-content:center;flex-direction:column;gap:12px;
    background:rgba(8,10,16,.92);z-index:10;">
    <div style="font-size:42px;opacity:.3">📷</div>
    <div style="font-family:var(--head);font-size:15px;font-weight:700;color:var(--sub)">No Camera Found</div>
    <div style="font-size:11px;color:var(--sub2);text-align:center;max-width:260px;line-height:1.6"
         id="no-cam-reason">Check that your browser has camera permission and no other app is using it.</div>
    <button onclick="initCam()" style="margin-top:6px;padding:8px 22px;border-radius:20px;
      border:1px solid var(--hi2);background:transparent;color:var(--hi2);
      font-family:var(--sans);font-size:11px;font-weight:700;cursor:pointer;">
      ↻ Retry
    </button>
  </div>

  <div class="cam-badges">
    <span class="cbadge live" id="badge-live">● LIVE</span>
    <span class="cbadge anime" id="badge-anime">AnimeGAN</span>
    <span class="cbadge gcam"  id="badge-gcam">Grad-CAM</span>
  </div>

  <div class="countdown" id="cd">3</div>

  <div class="cam-hud">
    <div class="shape-pills">
      <div class="sp active" id="sp-circle" onclick="setCamShape('circle',this)">◯ Circle</div>
      <div class="sp"        id="sp-square" onclick="setCamShape('square',this)">▭ Square</div>
      <div class="sp"        id="sp-raw"    onclick="setCamShape('raw',this)">⬜ Full</div>
    </div>
    <div class="hud-spacer"></div>
    <button class="shutter" title="Capture sticker [Space]" onclick="captureSticker()"></button>
    <div class="hud-spacer"></div>
    <div class="cam-info">
      <span id="cam-timer">00:00</span><br>cam time
    </div>
  </div>
</div>

<!-- ══ RIGHT SIDEBAR — TWO TABS ════════════════════════════════════════════ -->
<div class="right">

  <!-- Tab buttons -->
  <div class="tab-bar">
    <button class="tab-btn active" id="tab-webcam-btn" onclick="switchTab('webcam')">
      <span class="ticon">📷</span> Webcam
    </button>
    <button class="tab-btn" id="tab-studio-btn" onclick="switchTab('studio')">
      <span class="ticon">🎭</span> Sticker Studio
    </button>
  </div>

  <!-- ── WEBCAM TAB ───────────────────────────────────────────────────── -->
  <div class="tab-panel active" id="tab-webcam">

    <!-- Mini live preview (mirrors centre cam) -->
    <div class="cam-preview-thumb">
      <video id="cam-thumb" autoplay playsinline muted></video>
      <div class="live-dot">LIVE</div>
    </div>

    <!-- Recording status bar -->
    <div class="rec-bar">
      <span>Recording</span>
      <span class="rtime" id="rec-time-display">00:00</span>
    </div>

    <!-- Feature toggles -->
    <div class="sec">
      <div class="sec-title" style="margin-bottom:8px">Features</div>
      <div class="ctrl-grid">
        <button class="cbtn" id="c-anime" onclick="toggleFeature('anime')">AnimeGAN <span class="kbd">A</span></button>
        <button class="cbtn" id="c-gcam"  onclick="toggleFeature('gcam')">Grad-CAM  <span class="kbd">G</span></button>
      </div>
    </div>

    <!-- Camera actions -->
    <div class="sec">
      <div class="sec-title" style="margin-bottom:8px">Camera Controls</div>
      <div class="ctrl-grid">
        <button class="cbtn" onclick="triggerPhoto()">📷 Photo <span class="kbd">P</span></button>
        <button class="cbtn" onclick="triggerBurst()">💥 Burst <span class="kbd">B</span></button>
        <button class="cbtn wide" id="c-rec" onclick="toggleRec()">⏺ Record <span class="kbd">V</span></button>
        <button class="cbtn" onclick="doSnap()">💾 Snap <span class="kbd">S</span></button>
        <button class="cbtn" onclick="refreshGiphy()">↻ Sticker <span class="kbd">R</span></button>
      </div>
    </div>

    <!-- Shape selector in webcam tab -->
    <div class="sec">
      <div class="sec-title" style="margin-bottom:6px">Capture Shape</div>
      <div class="shape-row">
        <button class="shbtn active" id="ws-circle" onclick="setCamShape('circle',null,'ws-circle')">◯ Circle</button>
        <button class="shbtn"        id="ws-square" onclick="setCamShape('square',null,'ws-square')">▭ Square</button>
        <button class="shbtn"        id="ws-raw"    onclick="setCamShape('raw',null,'ws-raw')">⬜ Full</button>
      </div>
    </div>

    <!-- Quick capture button -->
    <div class="sec">
      <button class="abtn primary" style="width:100%;padding:11px" onclick="captureSticker()">
        ◉ Capture Frame  <span class="kbd" style="background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.2);color:rgba(255,255,255,.7)">Space</span>
      </button>
      <p style="font-size:9px;color:var(--sub2);text-align:center;margin-top:6px">
        Opens in Sticker Studio →
      </p>
    </div>

    <!-- Spacer pushes status to bottom -->
    <div style="flex:1"></div>
    <div class="statusbar" id="statusbar-cam">Ready</div>
  </div>

  <!-- ── STICKER STUDIO TAB ──────────────────────────────────────────── -->
  <div class="tab-panel" id="tab-studio">

    <div class="studio-hd">
      🎭 Sticker Studio
      <span class="tag tag-blue" style="font-size:8px">Creator</span>
    </div>

    <!-- Preview canvas -->
    <div class="stk-preview">
      <div class="tool-lbl">Preview &amp; Edit</div>
      <canvas id="stkCv" width="288" height="288"></canvas>
    </div>

    <!-- Font picker -->
    <div class="stk-sec">
      <div class="tool-lbl">Font Style</div>
      <div class="font-grid" id="font-grid"></div>

      <div>
        <div class="tool-lbl" style="margin-bottom:4px">Text Effects</div>
        <div class="fx-row">
          <div class="fxtog on"  data-fx="shadow"  onclick="toggleFx(this)">Shadow</div>
          <div class="fxtog"     data-fx="stroke"  onclick="toggleFx(this)">Stroke</div>
          <div class="fxtog"     data-fx="glow"    onclick="toggleFx(this)">Glow</div>
          <div class="fxtog"     data-fx="outline" onclick="toggleFx(this)">Outline</div>
          <div class="fxtog"     data-fx="retro"   onclick="toggleFx(this)">Italic</div>
          <div class="fxtog"     data-fx="grunge"  onclick="toggleFx(this)">Grunge</div>
          <div class="fxtog"     data-fx="rainbow" onclick="toggleFx(this)">Rainbow</div>
        </div>
      </div>

      <div>
        <div class="tool-lbl" style="margin-bottom:4px">Text Color</div>
        <div class="pal-row" id="pal-row">
          <div class="swatch-custom" title="Custom">
            <input type="color" id="col-custom" value="#ffffff" onchange="setCustomCol(this.value)">
          </div>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:4px">
        <div class="sl-row">
          <label>Size</label>
          <input type="range" min="10" max="88" value="34" id="sl-size" oninput="slv('sl-size','v-size')">
          <span class="slv" id="v-size">34</span>
        </div>
        <div class="sl-row">
          <label>Stroke</label>
          <input type="range" min="0" max="14" value="3" id="sl-stroke" oninput="slv('sl-stroke','v-stroke')">
          <span class="slv" id="v-stroke">3</span>
        </div>
        <div class="sl-row">
          <label>Opacity</label>
          <input type="range" min="20" max="100" value="100" id="sl-opa" oninput="slv('sl-opa','v-opa')">
          <span class="slv" id="v-opa">100</span>
        </div>
      </div>

      <div class="txt-row">
        <input class="txt-inp" id="txt-in" placeholder="Type sticker text…" maxlength="48"
          onkeydown="if(event.key==='Enter')addText()">
        <button class="add-btn" onclick="addText()">Add</button>
      </div>
      <div style="font-size:8px;color:var(--sub2)">Drag text on preview · click to select</div>
    </div>

    <!-- Shape -->
    <div class="stk-sec">
      <div class="tool-lbl" style="margin-bottom:4px">Sticker Shape</div>
      <div class="shape-row">
        <button class="shbtn active" onclick="setStudioShape('circle',this)">◯ Circle</button>
        <button class="shbtn"        onclick="setStudioShape('square',this)">▭ Square</button>
        <button class="shbtn"        onclick="setStudioShape('raw',this)">⬜ Full</button>
      </div>
    </div>

    <!-- Actions -->
    <div class="act-row">
      <button class="abtn" onclick="clearText()">Clear</button>
      <button class="abtn" onclick="undoText()">↩ Undo</button>
      <button class="abtn primary" id="save-btn" onclick="saveSticker()">Save Sticker</button>
    </div>

    <!-- Gallery -->
    <div class="gal-hd">
      <span class="sec-title">My Stickers</span>
      <span style="font-size:9px;color:var(--sub)" id="gal-count">0 saved</span>
    </div>
    <div class="gal-grid" id="gal-grid">
      <div class="gal-empty">Capture &amp; save stickers to fill your gallery</div>
    </div>

    <!-- Pack download -->
    <div class="pack-sec">
      <div class="pack-row">
        <span class="pack-n" id="pack-n">0</span>
        <span class="pack-sub">stickers saved</span>
      </div>
      <button class="pack-dl" id="pack-dl" disabled>⬇ Download All as ZIP</button>
    </div>

  </div><!-- end #tab-studio -->

</div><!-- end .right -->
</div><!-- end .shell -->

<div class="toast" id="toast"></div>

<script>
/* ════════════════════════════════════════════════════════
   CONSTANTS
════════════════════════════════════════════════════════ */
const EMO_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise'];
const EMO_COLORS = {
  angry:'#E74C3C',disgust:'#2ECC71',fear:'#9B59B6',
  happy:'#F1C40F',neutral:'#7F8C8D',sad:'#3498DB',surprise:'#E67E22'
};
const FONTS = [
  {fam:"'Fredoka One','Nunito',sans-serif",      lbl:'Bubble',  smp:'Bb'},
  {fam:"'Black Han Sans','Impact',sans-serif",   lbl:'Bold',    smp:'Bb'},
  {fam:"'Oxanium','Courier New',monospace",      lbl:'Cyber',   smp:'Bb'},
  {fam:"'Rockwell','Georgia',serif",             lbl:'Slab',    smp:'Bb'},
  {fam:"'Permanent Marker','Comic Sans MS',cursive",lbl:'Graff', smp:'Bb'},
  {fam:"'Dancing Script','Brush Script MT',cursive",lbl:'Script',smp:'Bb'},
];
const PALETTE = ['#ffffff','#111111','#e94560','#5b6af0','#f0b429','#00c896',
                 '#4a9eff','#ff6b35','#ff48c4','#2ecc71'];

/* ════════════════════════════════════════════════════════
   STATE
════════════════════════════════════════════════════════ */
let curEmo = 'neutral';
let animeOn = false, gcamOn = false;
let recActive = false, recTimer = null, recStart = 0, recDispTimer = null, vmr = null, vchunks = [];
let cdActive = false, cdCount = 3, cdTimer = null, burstLeft = 0;
let camStream = null, camStart = Date.now(), fps = 0, lastFpsT = Date.now();
let stkShape = 'circle', capturedCv = null;
let textLayers = [], selColor = '#ffffff', selFont = 0;
let fx = {shadow:true,stroke:false,glow:false,outline:false,retro:false,grunge:false,rainbow:false};
let dragging = null, dragOff = {x:0,y:0}, selLayer = -1;
let savedStickers = [], notoIdx = 0, giphyOff = 0;

const VID  = document.getElementById('cam-vid');
const STKC = document.getElementById('stkCv');
const STKX = STKC.getContext('2d');

/* ════════════════════════════════════════════════════════
   TAB SWITCHER
════════════════════════════════════════════════════════ */
function switchTab(name){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  document.getElementById('tab-'+name+'-btn').classList.add('active');
  if(name==='studio' && capturedCv) redraw();
}

/* ════════════════════════════════════════════════════════
   BUILD FONT GRID
════════════════════════════════════════════════════════ */
function buildFonts(){
  document.getElementById('font-grid').innerHTML = FONTS.map((f,i)=>`
    <div class="fbtn${i===0?' active':''}" onclick="pickFont(${i})" style="font-family:${f.fam}">
      <span class="fsamp" style="font-family:${f.fam}">${f.smp}</span>
      <span class="fname">${f.lbl}</span>
    </div>`).join('');
}
function pickFont(i){
  selFont=i;
  document.querySelectorAll('.fbtn').forEach((b,j)=>b.classList.toggle('active',j===i));
}

/* ════════════════════════════════════════════════════════
   BUILD PALETTE
════════════════════════════════════════════════════════ */
function buildPalette(){
  const row = document.getElementById('pal-row');
  PALETTE.forEach(c=>{
    const s = document.createElement('div');
    s.className = 'swatch'+(c==='#ffffff'?' active':'');
    s.style.cssText = `background:${c};border-color:${c==='#ffffff'?'#555':'transparent'}`;
    s.onclick = ()=>{
      selColor=c;
      row.querySelectorAll('.swatch').forEach(x=>x.classList.remove('active'));
      s.classList.add('active');
    };
    row.insertBefore(s, row.lastElementChild);
  });
}
function setCustomCol(v){ selColor=v; document.querySelectorAll('.swatch').forEach(s=>s.classList.remove('active')); }

/* ════════════════════════════════════════════════════════
   BUILD BARS
════════════════════════════════════════════════════════ */
function buildBars(){
  document.getElementById('bars').innerHTML = EMO_LABELS.map(e=>`
    <div class="bar-row">
      <div class="bar-lbl">${e.slice(0,4)}</div>
      <div class="bar-track"><div class="bar-fill" id="bf-${e}" style="background:${EMO_COLORS[e]}"></div></div>
      <div class="bar-pct" id="bp-${e}">0%</div>
    </div>`).join('');
}

/* ════════════════════════════════════════════════════════
   SSE — real emotion from /api/stream
════════════════════════════════════════════════════════ */
function connectSSE(){
  const es = new EventSource('/api/stream');
  es.onmessage = e=>{
    try{
      const d = JSON.parse(e.data);
      applyEmo(d.emotion, d.confidence, d.probs, d.fps);
    }catch(_){}
  };
  es.onerror = ()=> setTimeout(connectSSE, 3000);
}
function applyEmo(emo, conf, probs, fps_val){
  if(emo !== curEmo){ curEmo=emo; loadGiphy(); }
  const col = EMO_COLORS[emo]||'var(--hi)';
  const nm  = emo[0].toUpperCase()+emo.slice(1);
  document.getElementById('emo-name').textContent = nm;
  document.getElementById('emo-name').style.color = col;
  document.getElementById('emo-pct').textContent  = Math.round(conf*100)+'% confidence';
  document.getElementById('tb-emo').textContent   = nm;
  document.getElementById('tb-emo').style.color   = col;
  document.getElementById('tb-fps').textContent   = (fps_val||0)+' fps';
  if(probs) EMO_LABELS.forEach((e,i)=>{
    const v=probs[i]||0;
    const b=document.getElementById('bf-'+e); if(b) b.style.width=Math.round(v*100)+'%';
    const p=document.getElementById('bp-'+e); if(p) p.textContent=Math.round(v*100)+'%';
  });
  setStatus('Running | '+nm+' | '+(fps_val||0)+' fps');
}

/* ════════════════════════════════════════════════════════
   CAMERA
════════════════════════════════════════════════════════ */
async function initCam(){
  const overlay = document.getElementById('no-cam-overlay');
  const liveBadge = document.getElementById('badge-live');
  overlay.style.display = 'none';
  if(camStream){ camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
  try{
    camStream = await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false
    });
    VID.srcObject = camStream;
    VID.muted = true;
    const thumb = document.getElementById('cam-thumb');
    if(thumb){ thumb.srcObject = camStream; thumb.muted = true; }
    await new Promise((resolve,reject)=>{
      if(VID.readyState>=1){ resolve(); return; }
      VID.onloadedmetadata=resolve; VID.onerror=reject;
      setTimeout(reject,8000);
    });
    await VID.play();
    syncCanvasToVideo();
    camStart=Date.now();
    setStatus('Camera live');
    if(liveBadge) liveBadge.style.display='';
    requestAnimationFrame(camLoop);
    new ResizeObserver(syncCanvasToVideo).observe(document.querySelector('.centre'));
  }catch(e){
    console.error('Camera error:',e);
    overlay.style.display='flex';
    document.getElementById('no-cam-reason').textContent=
      e.name==='NotAllowedError'  ? 'Camera permission denied. Click the camera icon in your address bar, allow access, then click Retry.' :
      e.name==='NotFoundError'    ? 'No camera detected. Plug in a webcam and click Retry.' :
      e.name==='NotReadableError' ? 'Camera in use by another app. Close it and click Retry.' :
      'Error: '+e.message;
    if(liveBadge) liveBadge.style.display='none';
    setStatus('Camera unavailable — '+e.name);
    camStream=null;
  }
}
function syncCanvasToVideo(){
  const cv = document.getElementById('cam-cv');
  const area = document.querySelector('.centre');
  cv.width  = area.clientWidth;
  cv.height = area.clientHeight;
}
function camLoop(){
  fps++; const now=Date.now();
  if(now-lastFpsT>=1000){ fps=0; lastFpsT=now; }
  const el=Math.floor((now-camStart)/1000);
  document.getElementById('cam-timer').textContent=
    String(Math.floor(el/60)).padStart(2,'0')+':'+String(el%60).padStart(2,'0');
  requestAnimationFrame(camLoop);
}

/* ════════════════════════════════════════════════════════
   FEATURE TOGGLES
════════════════════════════════════════════════════════ */
async function toggleFeature(key){
  try{
    const r = await fetch('/api/toggle',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});
    const d = await r.json();
    if(key==='anime'){
      animeOn = d.anime_on;
      document.getElementById('c-anime').classList.toggle('act-blue', animeOn);
      document.getElementById('chip-anime').className='chip'+(animeOn?' on-blue':'');
      document.getElementById('chip-anime').textContent='AnimeGAN '+(animeOn?'ON':'OFF');
      document.getElementById('badge-anime').classList.toggle('on', animeOn);
      toast('AnimeGAN '+(animeOn?'ON':'OFF'));
    } else {
      gcamOn = d.gc_on;
      document.getElementById('c-gcam').classList.toggle('act-teal', gcamOn);
      document.getElementById('chip-gcam').className='chip'+(gcamOn?' on-teal':'');
      document.getElementById('badge-gcam').classList.toggle('on', gcamOn);
      toast('Grad-CAM '+(gcamOn?'ON':'OFF'));
    }
  }catch(e){ toast('Server toggle failed','err'); }
}

/* ════════════════════════════════════════════════════════
   VIDEO RECORDING
════════════════════════════════════════════════════════ */
function toggleRec(){
  if(!recActive){
    recActive=true; recStart=Date.now();
    document.getElementById('c-rec').classList.add('act-red');
    document.getElementById('c-rec').innerHTML='⏹ Stop <span class="kbd">V</span>';
    document.getElementById('rec-blink').classList.add('on');
    document.getElementById('rec-lbl').style.display='inline';
    const rtd = document.getElementById('rec-time-display');
    rtd.classList.add('on');
    setStatus('Recording… press V to stop');
    recDispTimer=setInterval(()=>{
      const s=Math.floor((Date.now()-recStart)/1000);
      rtd.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
    },500);
    recTimer=setInterval(()=>{
      const s=(Date.now()-recStart)/1000;
      if(s>=30) stopRec();
    },500);
    if(camStream){
      vchunks=[];
      try{ vmr=new MediaRecorder(camStream,{mimeType:'video/webm;codecs=vp9'}); }
      catch(e){ vmr=new MediaRecorder(camStream); }
      vmr.ondataavailable=e=>vchunks.push(e.data);
      vmr.onstop=()=>{
        const blob=new Blob(vchunks,{type:'video/webm'});
        const url=URL.createObjectURL(blob);
        const a=document.createElement('a'); a.href=url;
        a.download='emotion_video_'+Date.now()+'.webm'; a.click();
        URL.revokeObjectURL(url); toast('Video downloaded!','ok');
      };
      vmr.start();
    }
  } else stopRec();
}
function stopRec(){
  if(!recActive) return; recActive=false;
  clearInterval(recTimer); clearInterval(recDispTimer);
  document.getElementById('c-rec').classList.remove('act-red');
  document.getElementById('c-rec').innerHTML='⏺ Record <span class="kbd">V</span>';
  document.getElementById('rec-blink').classList.remove('on');
  document.getElementById('rec-lbl').style.display='none';
  const rtd=document.getElementById('rec-time-display');
  rtd.classList.remove('on'); rtd.textContent='00:00';
  if(vmr&&vmr.state!=='inactive') vmr.stop();
  setStatus('Recording saved');
}

/* ════════════════════════════════════════════════════════
   PHOTO / BURST
════════════════════════════════════════════════════════ */
function triggerPhoto(){ if(!cdActive) startCd(3,1); }
function triggerBurst(){ if(!cdActive) startCd(3,3); }
function startCd(secs,total){
  cdActive=true; cdCount=secs; burstLeft=total;
  const ov=document.getElementById('cd');
  ov.textContent=cdCount; ov.classList.add('on');
  setStatus('Photo in '+cdCount+'… smile!');
  cdTimer=setInterval(()=>{
    cdCount--;
    if(cdCount<=0){
      clearInterval(cdTimer); ov.classList.remove('on');
      takePhoto(); burstLeft--;
      if(burstLeft>0) setTimeout(()=>startCd(2,burstLeft),300); else cdActive=false;
    } else {
      ov.textContent=cdCount;
      ov.classList.remove('on'); void ov.offsetWidth; ov.classList.add('on');
    }
  },1000);
}
function takePhoto(){
  if(!camStream){ toast('No camera — allow access and click Retry in the viewer','err'); return; }
  const c=document.createElement('canvas');
  c.width=VID.videoWidth||640; c.height=VID.videoHeight||480;
  const x=c.getContext('2d');
  x.save(); x.scale(-1,1); x.drawImage(VID,-c.width,0); x.restore();
  const b64=c.toDataURL('image/jpeg',.92).split(',')[1];
  fetch('/api/save_photo',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:b64,filename:'photo_'+Date.now()+'.jpg'})}).catch(()=>{});
  const a=document.createElement('a'); a.href=c.toDataURL('image/png');
  a.download='photo_'+Date.now()+'.png'; a.click();
  toast('Photo saved!','ok');
}
function doSnap(){
  if(!camStream){ toast('No camera — allow access and click Retry in the viewer','err'); return; }
  const c=document.createElement('canvas');
  c.width=VID.videoWidth||640; c.height=VID.videoHeight||480;
  const x=c.getContext('2d');
  x.save(); x.scale(-1,1); x.drawImage(VID,-c.width,0); x.restore();
  const a=document.createElement('a'); a.href=c.toDataURL('image/png');
  a.download='snap_'+Date.now()+'.png'; a.click();
  toast('Snap saved!','ok');
}

/* ════════════════════════════════════════════════════════
   SHAPE SYNC
════════════════════════════════════════════════════════ */
function setCamShape(s, camPillBtn, webcamBtnId){
  stkShape=s;
  // Sync camera HUD pills
  document.querySelectorAll('.sp').forEach(b=>b.classList.remove('active'));
  const pill = document.getElementById('sp-'+s);
  if(pill) pill.classList.add('active');
  // Sync webcam tab buttons
  ['ws-circle','ws-square','ws-raw'].forEach(id=>{
    const el=document.getElementById(id);
    if(el) el.classList.toggle('active',id==='ws-'+s);
  });
  // Sync studio tab buttons
  document.querySelectorAll('#tab-studio .shbtn').forEach(b=>
    b.classList.toggle('active', b.textContent.toLowerCase().includes(s)));
  redraw();
}
function setStudioShape(s,btn){
  stkShape=s;
  // Studio buttons
  document.querySelectorAll('#tab-studio .shbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  // Webcam tab buttons
  ['ws-circle','ws-square','ws-raw'].forEach(id=>{
    const el=document.getElementById(id);
    if(el) el.classList.toggle('active',id==='ws-'+s);
  });
  // Camera HUD pills
  ['circle','square','raw'].forEach(x=>{
    const el=document.getElementById('sp-'+x);
    if(el) el.classList.toggle('active',x===s);
  });
  redraw();
}

/* ════════════════════════════════════════════════════════
   CAPTURE FRAME INTO STICKER EDITOR
════════════════════════════════════════════════════════ */
function captureSticker(){
  if(!camStream){
    switchTab('webcam');
    toast('No camera — allow access and click Retry','err');
    document.getElementById('no-cam-overlay').style.display='flex';
    return;
  }
  const c=document.createElement('canvas');
  c.width=VID.videoWidth||640; c.height=VID.videoHeight||480;
  const x=c.getContext('2d');
  x.save(); x.scale(-1,1); x.drawImage(VID,-c.width,0); x.restore();
  capturedCv=c; textLayers=[]; selLayer=-1;
  redraw();
  // Auto-switch to studio tab
  switchTab('studio');
  toast('Frame captured — add text then save!','ok');
  setStatus('Sticker captured → add text in studio');
}

/* ════════════════════════════════════════════════════════
   STICKER CANVAS RENDER
════════════════════════════════════════════════════════ */
function redraw(){
  const W=STKC.width, H=STKC.height;
  STKX.clearRect(0,0,W,H);
  for(let r=0;r<H;r+=12) for(let c=0;c<W;c+=12){
    STKX.fillStyle=(Math.floor(r/12)+Math.floor(c/12))%2===0?'#191d2e':'#131621';
    STKX.fillRect(c,r,12,12);
  }
  if(!capturedCv) return;
  STKX.save();
  if(stkShape==='circle'){
    STKX.beginPath(); STKX.arc(W/2,H/2,W/2-2,0,Math.PI*2); STKX.clip();
  } else if(stkShape==='square'){
    const p=7; STKX.beginPath(); STKX.roundRect(p,p,W-p*2,H-p*2,14); STKX.clip();
  }
  STKX.drawImage(capturedCv,0,0,W,H);
  STKX.restore();
  if(stkShape==='circle'){
    STKX.beginPath(); STKX.arc(W/2,H/2,W/2-2,0,Math.PI*2);
    STKX.strokeStyle='rgba(255,255,255,.10)'; STKX.lineWidth=2; STKX.stroke();
  }
  if(selLayer>=0 && selLayer<textLayers.length){
    const l=textLayers[selLayer];
    STKX.save();
    STKX.strokeStyle='rgba(91,106,240,.6)'; STKX.lineWidth=1.5; STKX.setLineDash([4,3]);
    STKX.strokeRect(l.x-70, l.y-l.size*.6-4, 140, l.size+8);
    STKX.restore();
  }
  textLayers.forEach(drawLayer);
}

function drawLayer(l){
  const {text,fam,size,col,x,y,effects,sw,opa}=l;
  STKX.save();
  STKX.globalAlpha = opa/100;
  const italic = effects.retro ? 'italic ' : '';
  STKX.font = `${italic}${Math.round(size)}px ${fam}`;
  STKX.textAlign='center'; STKX.textBaseline='middle';
  if(effects.glow){ STKX.shadowColor=col; STKX.shadowBlur=22; }
  if(effects.shadow){
    STKX.shadowOffsetX=3; STKX.shadowOffsetY=3;
    STKX.shadowBlur = effects.glow ? 22 : 8;
    STKX.shadowColor = effects.glow ? col : 'rgba(0,0,0,.8)';
  }
  if(effects.grunge){
    for(let i=0;i<4;i++){
      STKX.fillStyle='rgba(0,0,0,.18)';
      STKX.fillText(text, x+(Math.random()*6-3), y+(Math.random()*6-3));
    }
  }
  if(effects.outline){
    STKX.strokeStyle=inv(col); STKX.lineWidth=(sw||3)*3.5;
    STKX.lineJoin='round'; STKX.strokeText(text,x,y);
  }
  if(effects.stroke && sw>0){
    STKX.strokeStyle=inv(col); STKX.lineWidth=sw*2;
    STKX.lineJoin='round'; STKX.strokeText(text,x,y);
  }
  STKX.shadowColor='transparent'; STKX.shadowBlur=0;
  STKX.shadowOffsetX=0; STKX.shadowOffsetY=0;
  STKX.strokeStyle='rgba(0,0,0,.5)'; STKX.lineWidth=sw||3;
  STKX.lineJoin='round'; STKX.strokeText(text,x,y);
  if(effects.rainbow){
    const grad=STKX.createLinearGradient(x-size*2,y,x+size*2,y);
    ['#e94560','#f0b429','#00c896','#5b6af0','#ff48c4'].forEach((c,i,a)=>
      grad.addColorStop(i/(a.length-1),c));
    STKX.fillStyle=grad;
  } else {
    STKX.fillStyle=col;
  }
  STKX.fillText(text,x,y);
  STKX.restore();
}
function inv(hex){
  try{
    const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
    return r+g+b>382?'#000':'#fff';
  }catch{return '#000';}
}

/* ════════════════════════════════════════════════════════
   TEXT TOOLS
════════════════════════════════════════════════════════ */
function addText(){
  const txt=document.getElementById('txt-in').value.trim();
  if(!txt){ toast('Type some text first','err'); return; }
  const size = +document.getElementById('sl-size').value;
  const sw   = +document.getElementById('sl-stroke').value;
  const opa  = +document.getElementById('sl-opa').value;
  textLayers.push({
    text:txt, fam:FONTS[selFont].fam, size, col:selColor,
    x:STKC.width/2,
    y:STKC.height*(0.72 + textLayers.length*0.10),
    effects:{...fx}, sw, opa
  });
  selLayer = textLayers.length-1;
  document.getElementById('txt-in').value='';
  redraw(); toast('Text added — drag to reposition','ok');
}
function toggleFx(el){
  const f=el.dataset.fx; fx[f]=!fx[f];
  el.classList.toggle('on',fx[f]); redraw();
}
function clearText(){ textLayers=[]; selLayer=-1; redraw(); toast('Text cleared'); }
function undoText(){ if(textLayers.length){textLayers.pop(); selLayer=-1; redraw();} }
function slv(id,vid){ document.getElementById(vid).textContent=document.getElementById(id).value; }

/* ── drag & select ── */
STKC.addEventListener('mousedown',e=>{
  const r=STKC.getBoundingClientRect(),sx=STKC.width/r.width,sy=STKC.height/r.height;
  const cx=(e.clientX-r.left)*sx, cy=(e.clientY-r.top)*sy;
  let hit=false;
  for(let i=textLayers.length-1;i>=0;i--){
    const l=textLayers[i];
    if(Math.abs(cx-l.x)<90 && Math.abs(cy-l.y)<l.size*.7+6){
      dragging=i; selLayer=i; dragOff={x:cx-l.x,y:cy-l.y}; hit=true; break;
    }
  }
  if(!hit){ selLayer=-1; }
  redraw();
});
STKC.addEventListener('mousemove',e=>{
  if(dragging===null) return;
  const r=STKC.getBoundingClientRect(),sx=STKC.width/r.width,sy=STKC.height/r.height;
  textLayers[dragging].x=(e.clientX-r.left)*sx-dragOff.x;
  textLayers[dragging].y=(e.clientY-r.top)*sy-dragOff.y;
  redraw();
});
STKC.addEventListener('mouseup', ()=>{ dragging=null; });
STKC.addEventListener('mouseleave',()=>{ dragging=null; });

/* ════════════════════════════════════════════════════════
   SAVE STICKER → /api/capture
════════════════════════════════════════════════════════ */
async function saveSticker(){
  if(!capturedCv){ toast('Capture a frame first!','err'); return; }
  const btn=document.getElementById('save-btn');
  btn.textContent='Saving…'; btn.disabled=true;

  const canvasB64=STKC.toDataURL('image/png').split(',')[1];
  const fc=document.createElement('canvas');
  fc.width=VID.videoWidth||640; fc.height=VID.videoHeight||480;
  const fx2=fc.getContext('2d'); fx2.save(); fx2.scale(-1,1); fx2.drawImage(VID,-fc.width,0); fx2.restore();
  const frameB64=fc.toDataURL('image/jpeg',.88).split(',')[1];

  let savedId = null;
  try{
    const res=await fetch('/api/capture',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:frameB64, canvas:canvasB64, shape:stkShape, anime:animeOn})
    });
    const data=await res.json();
    if(data.ok) savedId=data.id;
  }catch(_){}

  const displayUrl=STKC.toDataURL('image/png');
  savedStickers.push({id:savedId||String(Date.now()), dataURL:displayUrl,
    date:new Date().toLocaleTimeString()});
  renderGallery();
  const a=document.createElement('a'); a.href=displayUrl;
  a.download='sticker_'+Date.now()+'.png'; a.click();
  toast('Sticker saved!','ok');
  setStatus('Sticker saved + downloaded');
  btn.textContent='Save Sticker'; btn.disabled=false;
}

/* ════════════════════════════════════════════════════════
   GALLERY
════════════════════════════════════════════════════════ */
function renderGallery(){
  const g=document.getElementById('gal-grid');
  document.getElementById('gal-count').textContent=savedStickers.length+' saved';
  document.getElementById('pack-n').textContent=savedStickers.length;
  document.getElementById('pack-dl').disabled=savedStickers.length===0;
  if(!savedStickers.length){
    g.innerHTML='<div class="gal-empty">Capture &amp; save stickers to fill your gallery</div>'; return;
  }
  g.innerHTML=savedStickers.map((s,i)=>`
    <div class="gcard" onclick="loadIntoEditor(${i})">
      <img src="${s.dataURL}" alt="">
      <div class="gdel" onclick="event.stopPropagation();delSticker(${i})">✕</div>
    </div>`).join('');
}
function delSticker(i){
  const s=savedStickers[i];
  fetch('/api/sticker/'+s.id,{method:'DELETE'}).catch(()=>{});
  savedStickers.splice(i,1); renderGallery(); toast('Deleted');
}
function loadIntoEditor(i){
  const img=new Image();
  img.onload=()=>{
    const c=document.createElement('canvas');
    c.width=img.naturalWidth; c.height=img.naturalHeight;
    c.getContext('2d').drawImage(img,0,0);
    capturedCv=c; textLayers=[]; selLayer=-1; redraw();
    toast('Loaded into editor','ok');
  };
  img.src=savedStickers[i].dataURL;
}

/* ════════════════════════════════════════════════════════
   DOWNLOAD PACK
════════════════════════════════════════════════════════ */
async function downloadPack(){
  if(!savedStickers.length) return;
  try{
    const sl=await fetch('/api/stickers').then(r=>r.json());
    if(sl.stickers&&sl.stickers.length){
      const res=await fetch('/api/pack',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ids:sl.stickers.map(s=>s.id)})});
      const blob=await res.blob();
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a'); a.href=url;
      a.download='sticker_pack.zip'; a.click();
      URL.revokeObjectURL(url); toast('ZIP downloaded!','ok'); return;
    }
  }catch(_){}
  savedStickers.forEach((s,i)=>setTimeout(()=>{
    const a=document.createElement('a'); a.href=s.dataURL;
    a.download='sticker_'+(i+1)+'.png'; a.click();
  },i*200));
  toast('Downloading '+savedStickers.length+' stickers…','ok');
}
document.getElementById('pack-dl').onclick=downloadPack;

/* ════════════════════════════════════════════════════════
   GIPHY + NOTO
════════════════════════════════════════════════════════ */
let gTimer=null;
async function loadGiphy(){
  clearTimeout(gTimer);
  gTimer=setTimeout(async()=>{
    try{
      const r=await fetch(`/api/giphy/${curEmo}?offset=${giphyOff}`);
      const d=await r.json();
      if(d.url){
        const box=document.getElementById('giphy-box'); box.innerHTML='';
        const img=document.createElement('img'); img.src=d.url;
        img.style='max-width:100%;max-height:100%;object-fit:contain'; box.appendChild(img);
      }
    }catch(_){}
  },700);
}
function refreshGiphy(){ giphyOff++; loadGiphy(); setStatus('Refreshing sticker…'); }

async function loadNoto(){
  try{
    const box=document.getElementById('noto-box'); box.innerHTML='';
    const img=document.createElement('img');
    img.src=`/api/noto/${curEmo}?idx=${notoIdx}&_=${Date.now()}`;
    img.style='max-width:100%;max-height:100%;object-fit:contain';
    box.appendChild(img);
  }catch(_){}
}
function nextNoto(){ notoIdx++; loadNoto(); }

/* ════════════════════════════════════════════════════════
   LOAD SERVER STICKERS INTO GALLERY ON START
════════════════════════════════════════════════════════ */
async function loadServerStickers(){
  try{
    const d=await fetch('/api/stickers').then(r=>r.json());
    if(d.stickers&&d.stickers.length){
      d.stickers.forEach(s=>savedStickers.push(
        {id:s.id, dataURL:`/sticker/${s.id}`, date:s.date, server:true}));
      renderGallery();
    }
  }catch(_){}
}

/* ════════════════════════════════════════════════════════
   STATUS / TOAST
════════════════════════════════════════════════════════ */
function setStatus(m){
  document.getElementById('statusbar').textContent=m;
  const sc=document.getElementById('statusbar-cam');
  if(sc) sc.textContent=m;
}
let _tt=null;
function toast(msg,type=''){
  const el=document.getElementById('toast');
  el.textContent=msg; el.className='toast show'+(type?' '+type:'');
  clearTimeout(_tt); _tt=setTimeout(()=>el.className='toast',2800);
}

/* ════════════════════════════════════════════════════════
   KEYBOARD SHORTCUTS
════════════════════════════════════════════════════════ */
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  const k=e.key.toLowerCase();
  if(k==='a')      toggleFeature('anime');
  else if(k==='g') toggleFeature('gcam');
  else if(k==='v') toggleRec();
  else if(k==='p') triggerPhoto();
  else if(k==='b') triggerBurst();
  else if(k==='s') doSnap();
  else if(k==='r') refreshGiphy();
  else if(k==='n') nextNoto();
  else if(k===' '){ e.preventDefault(); captureSticker(); }
});

/* ════════════════════════════════════════════════════════
   BOOT
════════════════════════════════════════════════════════ */
buildFonts();
buildPalette();
buildBars();
redraw();
initCam();
loadServerStickers();
connectSSE();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("    MultiFace Studio — dashboard.py   ")
    print(f"  Inference:  {'TF + project modules' if (_TF_OK and _PROJ_OK) else 'DEMO MODE (no model/cam found)'}")
    print(f"  AnimeGAN:   {'available (GPU)' if _ANIME_OK else 'not available'}")
    print(f"  cv2/PIL:    {'OK' if _CV2_OK else 'missing — sticker server-side disabled'}")
    print()
    print("  Open:  http://localhost:5050")
    print()
    _worker.start()
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)