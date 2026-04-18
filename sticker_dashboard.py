"""
sticker_dashboard.py
Flask-based sticker dashboard — serves the HTML UI and handles
sticker creation, listing, downloading, and ZIP pack generation.

Run:
    python sticker_dashboard.py
Then open:
    http://localhost:5050
"""
import os, sys, json, base64, io, glob, time
from flask import Flask, request, jsonify, send_file, render_template_string

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

STICKER_DIR = os.path.join(_ROOT, "stickers", "custom")
os.makedirs(STICKER_DIR, exist_ok=True)

# ── Try importing project modules ─────────────────────────────────────────────
try:
    import cv2, numpy as np
    from sticker_generator import StickerGenerator
    gen = StickerGenerator()
    _CV2_OK = True
except Exception as e:
    print(f"[Dashboard] cv2/sticker: {e}"); _CV2_OK = False; gen = None

try:
    from animegan_inference import apply_anime_style, is_available
    _ANIME_OK = is_available()
except:
    _ANIME_OK = False
    def apply_anime_style(f, **k): return f

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anime Sticker Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d0d14;--bg2:#13131f;--bg3:#1a1a2c;
  --border:#2a2a42;--accent:#7c6dfa;--accent2:#fa6d9f;
  --green:#4dffa0;--text:#e8e8ff;--muted:#6b6b8e;
}
body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

/* ── Layout ── */
.app{display:grid;grid-template-columns:380px 1fr;min-height:100vh;gap:0}
.panel-left{background:var(--bg2);border-right:1px solid var(--border);padding:24px;
  display:flex;flex-direction:column;gap:20px;overflow-y:auto}
.panel-right{padding:28px;overflow-y:auto}

/* ── Header ── */
.logo{font-size:22px;font-weight:800;letter-spacing:-0.5px}
.logo span{background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.badge{font-size:11px;background:var(--bg3);border:1px solid var(--border);
  color:var(--muted);padding:3px 10px;border-radius:20px;display:inline-block;margin-top:4px}

/* ── Camera ── */
.cam-wrap{position:relative;border-radius:16px;overflow:hidden;
  background:#000;border:1px solid var(--border)}
#cam{width:100%;display:block;transform:scaleX(-1)}
.cam-overlay{position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(transparent,rgba(0,0,0,.7));padding:14px;
  display:flex;justify-content:space-between;align-items:center}
.cam-status{font-size:12px;color:var(--muted)}
.cam-status.live{color:var(--green)}
#anime-pill{font-size:11px;padding:3px 10px;border-radius:20px;
  background:var(--bg3);border:1px solid var(--border);color:var(--muted);cursor:pointer;
  transition:.2s}
#anime-pill.on{background:#7c6dfa22;border-color:var(--accent);color:var(--accent)}

/* ── Controls ── */
.section-title{font-size:11px;font-weight:600;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--muted);margin-bottom:8px}

.shape-row{display:flex;gap:8px}
.shape-btn{flex:1;padding:10px 0;border-radius:10px;border:1px solid var(--border);
  background:var(--bg3);color:var(--muted);font-family:'Outfit',sans-serif;
  font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
.shape-btn.active{border-color:var(--accent);color:var(--accent);background:#7c6dfa15}
.shape-btn:hover{border-color:var(--muted)}

.capture-btn{width:100%;padding:14px;border-radius:12px;border:none;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;font-family:'Outfit',sans-serif;font-size:16px;font-weight:700;
  cursor:pointer;transition:.2s;letter-spacing:0.3px}
.capture-btn:hover{opacity:.88;transform:translateY(-1px)}
.capture-btn:active{transform:translateY(0)}

.hint{font-size:12px;color:var(--muted);text-align:center;line-height:1.6}

/* ── Pack builder ── */
.pack-bar{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:14px}
.pack-count{font-size:28px;font-weight:800;color:var(--accent)}
.pack-label{font-size:12px;color:var(--muted);margin-top:2px}
.pack-btn{width:100%;margin-top:12px;padding:11px;border-radius:10px;border:1px solid var(--accent);
  background:transparent;color:var(--accent);font-family:'Outfit',sans-serif;
  font-size:14px;font-weight:600;cursor:pointer;transition:.15s}
.pack-btn:hover{background:#7c6dfa18}
.pack-btn:disabled{opacity:.35;cursor:not-allowed}

/* ── Dashboard right ── */
.dash-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px}
.dash-title{font-size:26px;font-weight:800}
.dash-actions{display:flex;gap:10px}
.action-btn{padding:9px 18px;border-radius:10px;font-family:'Outfit',sans-serif;
  font-size:13px;font-weight:600;cursor:pointer;transition:.15s;border:1px solid var(--border)}
.action-btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.action-btn.secondary{background:var(--bg3);color:var(--text)}
.action-btn:hover{opacity:.82}

.filter-row{display:flex;gap:8px;margin-bottom:20px}
.filter-chip{padding:6px 14px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg3);color:var(--muted);font-size:12px;cursor:pointer;transition:.15s}
.filter-chip.active{border-color:var(--accent);color:var(--accent);background:#7c6dfa15}

.sticker-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:16px}

.sticker-card{background:var(--bg3);border:1px solid var(--border);border-radius:14px;
  overflow:hidden;transition:.2s;cursor:pointer;position:relative}
.sticker-card:hover{border-color:var(--accent);transform:translateY(-3px)}
.sticker-card.selected{border-color:var(--accent2);box-shadow:0 0 0 2px var(--accent2)}

.sticker-img-wrap{background:repeating-conic-gradient(#ffffff08 0% 25%,transparent 0% 50%)
  0 0 / 16px 16px;aspect-ratio:1;display:flex;align-items:center;justify-content:center}
.sticker-img-wrap img{width:90%;height:90%;object-fit:contain}

.sticker-info{padding:10px 12px;display:flex;justify-content:space-between;align-items:center}
.sticker-date{font-size:11px;color:var(--muted)}
.sticker-actions{display:flex;gap:6px}
.icon-btn{width:28px;height:28px;border-radius:8px;border:1px solid var(--border);
  background:var(--bg2);color:var(--muted);display:flex;align-items:center;
  justify-content:center;cursor:pointer;font-size:13px;transition:.15s}
.icon-btn:hover{border-color:var(--accent);color:var(--accent)}
.sel-check{position:absolute;top:8px;right:8px;width:22px;height:22px;
  border-radius:50%;background:var(--accent2);color:#fff;display:none;
  align-items:center;justify-content:center;font-size:11px;font-weight:700}
.sticker-card.selected .sel-check{display:flex}

.empty{text-align:center;padding:80px 20px;color:var(--muted)}
.empty-icon{font-size:48px;margin-bottom:12px;opacity:.3}
.empty-text{font-size:15px}

/* ── Toast ── */
.toast{position:fixed;bottom:30px;right:30px;background:var(--bg3);border:1px solid var(--border);
  padding:14px 20px;border-radius:12px;font-size:13px;color:var(--text);
  transform:translateY(80px);opacity:0;transition:.3s;z-index:999;max-width:300px}
.toast.show{transform:translateY(0);opacity:1}
.toast.success{border-color:var(--green);color:var(--green)}
.toast.error{border-color:#fa6d6d;color:#fa6d6d}

/* ── Progress ── */
.progress-bar{height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2));
  width:0;border-radius:3px;transition:.4s}
.progress-wrap{background:var(--border);border-radius:3px;overflow:hidden;margin-top:8px}

@media(max-width:768px){
  .app{grid-template-columns:1fr}
  .panel-left{border-right:none;border-bottom:1px solid var(--border)}
}
</style>
</head>
<body>
<div class="app">

<!-- ── LEFT: Camera Panel ── -->
<div class="panel-left">
  <div>
    <div class="logo">Anime <span>Sticker</span> Studio</div>
    <div class="badge">Custom Sticker Creator</div>
  </div>

  <div class="cam-wrap">
    <video id="cam" autoplay playsinline muted></video>
    <canvas id="canvas" style="display:none"></canvas>
    <div class="cam-overlay">
      <span class="cam-status live" id="cam-status">● Live</span>
      <span id="anime-pill" onclick="toggleAnime()">AnimeGAN OFF</span>
    </div>
  </div>

  <div>
    <div class="section-title">Sticker shape</div>
    <div class="shape-row">
      <button class="shape-btn active" onclick="setShape('circle',this)">Circle</button>
      <button class="shape-btn" onclick="setShape('square',this)">Square</button>
      <button class="shape-btn" onclick="setShape('raw',this)">Full</button>
    </div>
  </div>

  <button class="capture-btn" onclick="capture()">Capture Sticker</button>

  <div class="hint">
    Strike a pose, make a face, hold an object —<br>
    anything goes. No labels. Just your expression.
  </div>

  <div class="pack-bar">
    <div class="pack-count" id="sel-count">0</div>
    <div class="pack-label">stickers selected for pack</div>
    <div class="progress-wrap"><div class="progress-bar" id="pack-progress"></div></div>
    <button class="pack-btn" id="pack-btn" disabled onclick="downloadPack()">
      Download as ZIP Pack
    </button>
  </div>
</div>

<!-- ── RIGHT: Dashboard ── -->
<div class="panel-right">
  <div class="dash-header">
    <div class="dash-title">My Stickers</div>
    <div class="dash-actions">
      <button class="action-btn secondary" onclick="selectAll()">Select all</button>
      <button class="action-btn secondary" onclick="clearSel()">Clear</button>
      <button class="action-btn primary" onclick="downloadPack()">Download Pack</button>
    </div>
  </div>

  <div class="filter-row">
    <div class="filter-chip active" onclick="filterBy('all',this)">All</div>
    <div class="filter-chip" onclick="filterBy('circle',this)">Circle</div>
    <div class="filter-chip" onclick="filterBy('square',this)">Square</div>
    <div class="filter-chip" onclick="filterBy('raw',this)">Full</div>
  </div>

  <div class="sticker-grid" id="grid">
    <div class="empty">
      <div class="empty-icon">🎭</div>
      <div class="empty-text">No stickers yet.<br>Strike a pose and capture!</div>
    </div>
  </div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
const API = '';
let shape = 'circle';
let animeOn = false;
let selected = new Set();
let allStickers = [];
let activeFilter = 'all';
let stream = null;
const vid = document.getElementById('cam');
const canvas = document.getElementById('canvas');

// ── Camera init ──────────────────────────────────────────────────────────────
async function initCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    vid.srcObject = stream;
    document.getElementById('cam-status').textContent = '● Live';
  } catch(e) {
    document.getElementById('cam-status').textContent = 'No camera';
    document.getElementById('cam-status').className = 'cam-status';
    toast('Camera not accessible', 'error');
  }
}

function setShape(s, btn) {
  shape = s;
  document.querySelectorAll('.shape-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}

function toggleAnime() {
  animeOn = !animeOn;
  const pill = document.getElementById('anime-pill');
  pill.textContent = animeOn ? 'AnimeGAN ON' : 'AnimeGAN OFF';
  pill.className = 'on' ? animeOn : '';
  if(animeOn) pill.classList.add('on'); else pill.classList.remove('on');
}

// ── Capture ──────────────────────────────────────────────────────────────────
async function capture() {
  if(!stream){ toast('No camera available','error'); return; }
  canvas.width = vid.videoWidth || 640;
  canvas.height = vid.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.save(); ctx.scale(-1,1); ctx.drawImage(vid,-canvas.width,0); ctx.restore();
  const b64 = canvas.toDataURL('image/jpeg',0.92).split(',')[1];
  const btn = document.querySelector('.capture-btn');
  btn.textContent = 'Processing...'; btn.disabled = true;
  try {
    const res = await fetch('/api/capture', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image:b64, shape, anime:animeOn})
    });
    const data = await res.json();
    if(data.ok) {
      toast('Sticker created!','success');
      await loadStickers();
    } else {
      toast(data.error||'Error creating sticker','error');
    }
  } catch(e){ toast('Server error','error'); }
  btn.textContent = 'Capture Sticker'; btn.disabled = false;
}

// ── Load stickers ─────────────────────────────────────────────────────────────
async function loadStickers() {
  const res = await fetch('/api/stickers');
  const data = await res.json();
  allStickers = data.stickers || [];
  renderGrid();
}

function filterBy(f, el) {
  activeFilter = f;
  document.querySelectorAll('.filter-chip').forEach(c=>c.classList.remove('active'));
  el.classList.add('active');
  renderGrid();
}

function renderGrid() {
  const grid = document.getElementById('grid');
  let items = allStickers;
  if(activeFilter !== 'all') items = items.filter(s=>s.shape===activeFilter);
  if(!items.length) {
    grid.innerHTML = `<div class="empty">
      <div class="empty-icon">🎭</div>
      <div class="empty-text">No stickers yet.<br>Strike a pose and capture!</div>
    </div>`;
    return;
  }
  grid.innerHTML = items.map(s=>`
    <div class="sticker-card${selected.has(s.id)?' selected':''}"
         id="card-${s.id}" onclick="toggleSelect('${s.id}')">
      <div class="sticker-img-wrap">
        <img src="/sticker/${s.id}" alt="sticker" loading="lazy">
      </div>
      <div class="sticker-info">
        <span class="sticker-date">${s.date}</span>
        <div class="sticker-actions" onclick="event.stopPropagation()">
          <div class="icon-btn" title="Download PNG" onclick="downloadOne('${s.id}','png')">↓</div>
          <div class="icon-btn" title="Download WebP" onclick="downloadOne('${s.id}','webp')">W</div>
          <div class="icon-btn" title="Delete" onclick="deleteOne('${s.id}')">✕</div>
        </div>
      </div>
      <div class="sel-check">✓</div>
    </div>`).join('');
  updatePackBar();
}

// ── Selection ─────────────────────────────────────────────────────────────────
function toggleSelect(id) {
  if(selected.has(id)) selected.delete(id);
  else selected.add(id);
  renderGrid();
}
function selectAll() {
  allStickers.forEach(s=>selected.add(s.id));
  renderGrid();
}
function clearSel() {
  selected.clear(); renderGrid();
}
function updatePackBar() {
  document.getElementById('sel-count').textContent = selected.size;
  document.getElementById('pack-btn').disabled = selected.size === 0;
  const pct = Math.min(100, selected.size * 8);
  document.getElementById('pack-progress').style.width = pct+'%';
}

// ── Download ──────────────────────────────────────────────────────────────────
function downloadOne(id, fmt) {
  const link = document.createElement('a');
  link.href = `/sticker/${id}?fmt=${fmt}`;
  link.download = `sticker_${id}.${fmt}`;
  link.click();
}

async function downloadPack() {
  if(!selected.size){ toast('Select stickers first','error'); return; }
  const btn = document.getElementById('pack-btn');
  btn.textContent = 'Generating...'; btn.disabled = true;
  try {
    const res = await fetch('/api/pack', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ids:[...selected]})
    });
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href  = url; link.download = 'anime_sticker_pack.zip'; link.click();
    URL.revokeObjectURL(url);
    toast(`Downloaded pack with ${selected.size} stickers`,'success');
  } catch(e){ toast('Pack error','error'); }
  btn.textContent = 'Download as ZIP Pack';
  updatePackBar();
}

async function deleteOne(id) {
  await fetch(`/api/sticker/${id}`, {method:'DELETE'});
  selected.delete(id);
  await loadStickers();
  toast('Deleted','success');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type='') {
  const el = document.getElementById('toast');
  el.textContent = msg; el.className = 'toast show '+(type||'');
  setTimeout(()=>el.className='toast', 3000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
initCam();
loadStickers();
setInterval(loadStickers, 5000);
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/capture', methods=['POST'])
def api_capture():
    if not _CV2_OK:
        return jsonify({"ok":False,"error":"cv2 not available"})
    try:
        data  = request.json
        b64   = data['image']
        shape = data.get('shape','circle')
        anime = data.get('anime', False)

        # Decode image
        img_bytes = base64.b64decode(b64)
        arr   = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"ok":False,"error":"Could not decode image"})

        # Apply AnimeGAN if requested and available
        if anime and _ANIME_OK:
            try:
                out = apply_anime_style(frame, size=512)
                if out is not None:
                    frame = cv2.resize(out, (frame.shape[1], frame.shape[0]))
            except Exception as e:
                print(f"[Anime] {e}")

        _, png_path, _ = gen.make(frame, shape=shape)
        return jsonify({"ok": True, "path": png_path})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

@app.route('/api/stickers')
def api_stickers():
    pngs = gen.load_saved() if gen else []
    stickers = []
    for p in pngs:
        fname = os.path.basename(p)
        sid   = fname.replace('.png','')
        mtime = os.path.getmtime(p)
        from datetime import datetime
        date  = datetime.fromtimestamp(mtime).strftime('%b %d, %H:%M')
        # Guess shape from filename (future: store metadata)
        shape = 'circle'
        stickers.append({"id":sid,"date":date,"shape":shape,"path":p})
    return jsonify({"stickers": stickers})

@app.route('/sticker/<sid>')
def serve_sticker(sid):
    fmt  = request.args.get('fmt','png')
    ext  = 'webp' if fmt=='webp' else 'png'
    path = os.path.join(STICKER_DIR, f"{sid}.{ext}")
    if not os.path.exists(path):
        path = os.path.join(STICKER_DIR, f"{sid}.png")
    if not os.path.exists(path):
        return "Not found", 404
    mime = 'image/webp' if ext=='webp' else 'image/png'
    return send_file(path, mimetype=mime,
                     as_attachment=(fmt in ('png','webp')),
                     download_name=f"sticker_{sid}.{ext}")

@app.route('/api/pack', methods=['POST'])
def api_pack():
    ids      = request.json.get('ids',[])
    paths    = []
    for sid in ids:
        p = os.path.join(STICKER_DIR, f"{sid}.png")
        if os.path.exists(p): paths.append(p)
    if not paths:
        return "No stickers found", 404
    zip_path = gen.make_pack(paths)
    return send_file(zip_path, mimetype='application/zip',
                     as_attachment=True, download_name='anime_sticker_pack.zip')

@app.route('/api/sticker/<sid>', methods=['DELETE'])
def delete_sticker(sid):
    for ext in ['png','webp']:
        p = os.path.join(STICKER_DIR, f"{sid}.{ext}")
        if os.path.exists(p): os.remove(p)
    return jsonify({"ok":True})

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n Anime Sticker Studio")
    print("--------------------------")
    print(" Open: http://localhost:5050")
    print(" Strike a pose and capture!\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
