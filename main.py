"""
main_app.py — All-in-one emotion recognition app

Layout:
  ┌──────────────────┬──────────────┬──────────────┐
  │   WEBCAM FEED    │   STICKER    │    EMOJI     │
  │  face detection  │ Giphy/Noto   │  MediaPipe   │
  │  emotion label   │ animated GIF │  Noto avatar │
  │  GradCAM / Anime │              │              │
  └──────────────────┴──────────────┴──────────────┘

Controls:
    A      toggle AnimeGAN cartoon
    G      toggle GradCAM heatmap
    E      capture emoji (press while facing camera)
    N      next emoji variation
    R      refresh sticker
    S      save emoji PNG
    SPACE  pause/resume sticker animation
    Q/ESC  quit
"""
import sys, os, cv2, numpy as np, time
import threading, glob, urllib.request, io, requests
from PIL import Image, ImageSequence
import mediapipe as mp

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from face_detector import FaceDetector
from preprocessing import preprocess_face, preprocess_face_rgb, EMOTION_LABELS


class EmotionSmoother:
    """Rolling average over last N frames — kills flickering."""
    def __init__(self, window=8):
        self.window  = window
        self.history = []

    def update(self, probs):
        self.history.append(probs.copy())
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

    def reset(self):
        self.history = []
from grad_cam      import GradCAM
from display_utils import draw_face_box, draw_fps

try:
    from animegan_inference import apply_anime_style, is_available as anime_available
    ANIME_READY = anime_available()
except Exception:
    ANIME_READY = False
    apply_anime_style = None

MODEL_PATH  = os.path.join(_ROOT, "models", "saved_model", "emotion_model.h5")
LAST_CONV   = "res_conv4"
STICKER_DIR = os.path.join(_ROOT, "stickers")
SAVE_DIR    = os.path.join(_ROOT, "saved_emojis")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STICKER_DIR, exist_ok=True)
PHOTOS_DIR = os.path.join(_ROOT, "photos")
VIDEOS_DIR = os.path.join(_ROOT, "videos")
VIDEO_FPS  = 15   # target fps for recorded video
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Panel sizes
CAM_W, CAM_H   = 480, 360
PANEL_W        = 280
PANEL_H        = CAM_H
TOTAL_W        = CAM_W + PANEL_W * 2
TOTAL_H        = CAM_H + 50   # extra for bottom bar

GIPHY_API_KEY  = "jWRZPFqFvThUs6Qb6TCzNawzmdsftmnQ"

# ── Emotion styling ────────────────────────────────────────────────────────────
ECOLORS = {
    "angry":(30,30,200),   "disgust":(20,140,20), "fear":(130,0,200),
    "happy":(20,200,20),   "neutral":(120,120,120),"sad":(200,100,20),
    "surprise":(20,150,240),
}
EBGS = {
    "angry":(230,210,210), "disgust":(210,235,210),"fear":(230,210,250),
    "happy":(210,248,210), "neutral":(235,235,235),"sad":(210,225,255),
    "surprise":(210,235,255),
}

# ── Noto emoji codes ───────────────────────────────────────────────────────────
NOTO_CODES = {
    "angry":   ["1f620","1f621","1f624","1f92c"],
    "disgust": ["1f922","1f92e","1f915"],
    "fear":    ["1f628","1f630","1f631"],
    "happy":   ["1f600","1f601","1f603","1f604","1f606","1f929","1f973"],
    "neutral": ["1f610","1f611","1f636"],
    "sad":     ["1f622","1f625","1f62d","1f614"],
    "surprise":["1f62e","1f62f","1f635","1f92f"],
}
NOTO_BASE = "https://fonts.gstatic.com/s/e/notoemoji/latest/{code}/512.gif"

GIPHY_QUERIES = {
    "angry":   ["angry emoji","mad reaction"],
    "disgust": ["disgusted emoji","eww reaction"],
    "fear":    ["scared emoji","afraid reaction"],
    "happy":   ["happy emoji dancing","celebration sticker"],
    "neutral": ["neutral face emoji","meh sticker"],
    "sad":     ["sad emoji crying","crying sticker"],
    "surprise":["surprised emoji","shocked reaction"],
}

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
             379,378,400,377,152,148,176,149,150,136,172,58,132,93,
             234,127,162,21,54,103,67,109]


# ══════════════════════════════════════════════════════════════════════════════
# GIF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def gif_frames(src, size):
    try:
        gif = Image.open(src if isinstance(src,(str,os.PathLike)) else io.BytesIO(src))
        out = []
        for f in ImageSequence.Iterator(gif):
            dur  = f.info.get("duration",80)
            rgba = f.convert("RGBA").resize(size,Image.LANCZOS)
            arr  = np.array(rgba)
            bg   = np.full_like(arr[:,:,:3],255)
            a    = arr[:,:,3:4]/255.0
            bgr  = cv2.cvtColor((arr[:,:,:3]*a+bg*(1-a)).astype(np.uint8),
                                cv2.COLOR_RGB2BGR)
            out.append((bgr,max(20,dur)))
        return out
    except Exception as e:
        print(f"[GIF] {e}")
        return []


def gif_best_frame(src, size):
    frames = gif_frames(src, size)
    if not frames:
        return None
    return frames[len(frames)//2][0]


def local_gifs(emotion):
    d = os.path.join(STICKER_DIR, emotion.lower())
    os.makedirs(d, exist_ok=True)
    return sorted(glob.glob(os.path.join(d,"*.gif")) +
                  glob.glob(os.path.join(d,"*.GIF")))


def ensure_noto(emotion, idx=0):
    codes  = NOTO_CODES.get(emotion.lower(),["1f610"])
    code   = codes[idx % len(codes)]
    folder = os.path.join(STICKER_DIR, emotion.lower())
    os.makedirs(folder, exist_ok=True)
    path   = os.path.join(folder, f"{code}.gif")
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return path
    try:
        print(f"[Noto] Downloading {emotion} {code}...")
        urllib.request.urlretrieve(NOTO_BASE.format(code=code), path)
        return path if os.path.getsize(path) > 500 else None
    except Exception as e:
        print(f"[Noto] {e}"); return None


def fetch_giphy(emotion, offset=0):
    qs    = GIPHY_QUERIES.get(emotion.lower(),[f"{emotion} emoji"])
    query = qs[offset % len(qs)]
    try:
        r = requests.get("https://api.giphy.com/v1/gifs/search",
            params={"q":query,"api_key":GIPHY_API_KEY,"limit":10,
                    "rating":"g","lang":"en"},
            timeout=10)
        r.raise_for_status()
        data = r.json().get("data",[])
        if not data:
            print(f"[Giphy] No results for '{query}'")
            return None
        # Use original GIF URL (not mp4/webp)
        item    = data[offset % len(data)]
        gif_url = item["images"]["original"]["url"]
        # Strip query params that can cause issues
        gif_url = gif_url.split("?")[0]
        print(f"[Giphy] Fetching: {gif_url}")
        gr = requests.get(gif_url, timeout=15,
                          headers={"User-Agent":"Mozilla/5.0"})
        gr.raise_for_status()
        # Verify it is actually a GIF
        if gr.content[:3] != b"GIF":
            print(f"[Giphy] Not a GIF ({gr.content[:10]}), trying direct download url")
            dl_url = item["images"]["original"]["url"]
            gr = requests.get(dl_url, timeout=15)
            gr.raise_for_status()
        print(f"[Giphy] OK {query} ({len(gr.content)//1024}KB)")
        return gr.content
    except Exception as e:
        print(f"[Giphy] Error: {e}"); return None


# ══════════════════════════════════════════════════════════════════════════════
# STICKER PLAYER
# ══════════════════════════════════════════════════════════════════════════════
class StickerPlayer:
    def __init__(self, w, h):
        self.W = w; self.H = h
        self.frames=[]; self.fidx=0; self.tick=time.time()
        self.emotion=None; self.offset=0; self.paused=False
        self.raw=None; self._lock=threading.Lock(); self._busy=False
        self._placeholder("Detecting\nemotion...")

    def _placeholder(self, msg):
        ph = np.full((self.H,self.W,3),(240,240,240),dtype=np.uint8)
        cv2.circle(ph,(self.W//2,self.H//2-20),55,(210,210,210),-1)
        font=cv2.FONT_HERSHEY_SIMPLEX
        for i,ln in enumerate(msg.split("\n")):
            (tw,_),_=cv2.getTextSize(ln,font,0.5,1)
            cv2.putText(ph,ln,((self.W-tw)//2,self.H//2+50+i*22),
                        font,0.5,(130,130,130),1)
        with self._lock:
            self.frames=[(ph,500)]; self.fidx=0

    def load(self, emotion, refresh=False):
        if self._busy: return
        if emotion==self.emotion and not refresh: return
        self.emotion=emotion; self._busy=True
        self._placeholder(f"Loading\n{emotion}...")
        def _work():
            # ── GIPHY ONLY for sticker panel ──────────────────────────────
            raw = fetch_giphy(emotion, self.offset)
            if raw:
                frames = gif_frames(raw, (self.W, self.H))
                if frames: self.raw = raw
            else:
                # Giphy failed — check local cache only (no Noto here)
                local = local_gifs(emotion)
                frames = gif_frames(local[self.offset%len(local)],
                                    (self.W,self.H)) if local else []
            if frames:
                with self._lock:
                    self.frames=frames; self.fidx=0
            else:
                self._placeholder(f"Giphy failed\nPress R to retry")
            self._busy=False  # always release lock
        threading.Thread(target=_work,daemon=True).start()

    def refresh(self):
        self.offset += 1
        e = self.emotion
        self.emotion = None   # reset so load() doesn't skip
        self._busy   = False  # unblock in case previous load stalled
        self.load(e, refresh=True)

    def get_frame(self):
        with self._lock:
            if not self.frames:
                return np.full((self.H,self.W,3),240,dtype=np.uint8)
            bgr,dur=self.frames[self.fidx]
            if not self.paused and (time.time()-self.tick)*1000>=dur:
                self.fidx=(self.fidx+1)%len(self.frames)
                self.tick=time.time()
            return bgr.copy()


# ══════════════════════════════════════════════════════════════════════════════
# EMOJI COMPOSER
# ══════════════════════════════════════════════════════════════════════════════
def get_landmarks(face_bgr):
    try:
        mp_face=mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=True,max_num_faces=1,
                              refine_landmarks=True,
                              min_detection_confidence=0.5) as mesh:
            rgb=cv2.cvtColor(face_bgr,cv2.COLOR_BGR2RGB)
            res=mesh.process(rgb)
            if not res.multi_face_landmarks: return None
            lms=res.multi_face_landmarks[0].landmark
            return [(l.x,l.y) for l in lms]
    except AttributeError:
        pass
    try:
        BaseOptions=mp.tasks.BaseOptions
        FL=mp.tasks.vision.FaceLandmarker
        FLO=mp.tasks.vision.FaceLandmarkerOptions
        RM=mp.tasks.vision.RunningMode
        mp_path=os.path.join(_ROOT,"face_landmarker.task")
        if not os.path.exists(mp_path):
            print("[Emoji] Downloading MediaPipe model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task",mp_path)
        opts=FLO(base_options=BaseOptions(model_asset_path=mp_path),
                 running_mode=RM.IMAGE,num_faces=1,
                 min_face_detection_confidence=0.5)
        with FL.create_from_options(opts) as lm:
            img=mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(face_bgr,cv2.COLOR_BGR2RGB))
            res=lm.detect(img)
            if not res.face_landmarks: return None
            return [(l.x,l.y) for l in res.face_landmarks[0]]
    except Exception as e:
        print(f"[Emoji] Landmark error: {e}"); return None


def compose_emoji(face_bgr, lms, noto_bgr, emotion, conf, W, H):
    """Show Noto emoji cleanly on colored background — no face overlay."""
    key   = emotion.lower()
    bg    = EBGS.get(key,(235,235,235))
    color = ECOLORS.get(key,(100,100,100))
    S     = min(W, H-40)

    # Clean background circle
    canvas = np.full((S,S,3),(255,255,255),dtype=np.uint8)
    cv2.circle(canvas,(S//2,S//2),S//2-4,bg,-1)
    cv2.circle(canvas,(S//2,S//2),S//2-4,tuple(max(0,c-30) for c in bg),2)

    # Noto emoji centered and large
    pad    = int(S * 0.06)
    es     = S - 2*pad
    emoji  = cv2.resize(noto_bgr,(es,es),interpolation=cv2.INTER_LANCZOS4)
    canvas[pad:pad+es, pad:pad+es] = emoji

    # Resize to panel width
    result = cv2.resize(canvas,(W,H-40),interpolation=cv2.INTER_LANCZOS4)

    # Label bar
    bar   = np.full((40,W,3),color,dtype=np.uint8)
    lbl   = f"{emotion.capitalize()}  {conf:.0%}"
    (tw,_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.putText(bar,lbl,((W-tw)//2,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return np.vstack([result,bar])


# ══════════════════════════════════════════════════════════════════════════════
# CARTOON THREAD
# ══════════════════════════════════════════════════════════════════════════════
class CartoonThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._in=None; self._out=None
        self._lk=threading.Lock(); self._ev=threading.Event()
        self.running=True
    def submit(self,f):
        with self._lk: self._in=f.copy()
        self._ev.set()
    def result(self):
        with self._lk: return self._out.copy() if self._out is not None else None
    def run(self):
        while self.running:
            self._ev.wait(1); self._ev.clear()
            with self._lk: f=self._in
            if f is None: continue
            try:
                out=apply_anime_style(f,size=512)
                out=cv2.resize(out,(f.shape[1],f.shape[0]))
                with self._lk: self._out=out
            except Exception as e: print(f"[Anime] {e}")
    def stop(self): self.running=False; self._ev.set()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    try:
        import tensorflow as tf
        m=tf.keras.models.load_model(MODEL_PATH)
        m.predict(np.zeros((1,48,48,1),dtype=np.float32),verbose=0)
        print("[App] Model loaded.")
        return m
    except Exception as e:
        print(f"[App] {e} — demo mode"); return None


def predict(model, face_bgr):
    if model is None:
        p=np.random.dirichlet(np.ones(len(EMOTION_LABELS)))
        return EMOTION_LABELS[np.argmax(p)], float(p.max()), p
    try:
        is_rgb = model.input_shape[-1] == 3
    except: is_rgb = False
    img   = preprocess_face_rgb(face_bgr) if is_rgb else preprocess_face(face_bgr)
    # Use direct model call — 10x faster than model.predict() for single images
    import tensorflow as tf
    preds = model(img, training=False)[0].numpy()
    return EMOTION_LABELS[np.argmax(preds)], float(preds.max()), preds


# ══════════════════════════════════════════════════════════════════════════════
# PANEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def build_sticker_panel(sf, emotion, conf, loading, paused):
    W,H=PANEL_W,PANEL_H
    panel=np.full((H,W,3),(18,18,18),dtype=np.uint8)

    # Header
    cv2.putText(panel,"STICKER (Giphy)",(W//2-62,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)

    # Sticker frame
    sh=H-60
    sf_r=cv2.resize(sf,(W-8,sh))
    panel[30:30+sh,4:4+W-8]=sf_r
    cv2.rectangle(panel,(3,29),(W-4,30+sh),(60,60,60),1)

    # Emotion bar
    by=30+sh+4
    color=ECOLORS.get(emotion.lower(),(100,100,100))
    cv2.rectangle(panel,(4,by),(W-4,by+22),color,-1)
    lbl=f"{emotion.capitalize()} {conf:.0%}"
    (tw,_),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    cv2.putText(panel,lbl,((W-tw)//2,by+16),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # Status
    parts=[]
    if loading: parts.append("Loading...")
    if paused:  parts.append("PAUSED")
    cv2.putText(panel," ".join(parts),(6,H-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.33,(140,140,140),1)
    return panel


def build_emoji_panel(emoji_frame, generating, has_emoji):
    W,H=PANEL_W,PANEL_H
    panel=np.full((H,W,3),(18,18,18),dtype=np.uint8)

    # Header
    cv2.putText(panel,"EMOJI (Noto)",(W//2-52,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)

    if emoji_frame is not None:
        ef=cv2.resize(emoji_frame,(W-8,H-30))
        panel[28:28+H-30,4:W-4]=ef
    else:
        msg="Press E to" if not generating else "Generating..."
        msg2="generate" if not generating else "please wait"
        cv2.circle(panel,(W//2,H//2-20),55,(35,35,35),-1)
        cv2.putText(panel,msg,(W//2-55,H//2+35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,160,160),1)
        cv2.putText(panel,msg2,(W//2-45,H//2+58),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,160,160),1)

    cv2.putText(panel,"E=capture N=next S=save",(4,H-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.33,(120,120,120),1)
    return panel


def build_bottom_bar(anime_on, gradcam_on, emotion, conf):
    bar=np.full((50,TOTAL_W,3),(15,15,15),dtype=np.uint8)
    color=ECOLORS.get(emotion.lower(),(100,100,100))

    # Emotion
    lbl=f"Emotion: {emotion.capitalize()}  {conf:.0%}"
    cv2.putText(bar,lbl,(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    cv2.putText(bar,lbl,(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,1)

    # Controls
    ctrl=f"A=Anime({'ON' if anime_on else 'OFF'})  G=GradCAM({'ON' if gradcam_on else 'OFF'})  R=Sticker  E=Emoji  N=Next  S=Save  P=Photo  B=Burst(3)  V=Video  Q=Quit"
    cv2.putText(bar,ctrl,(CAM_W+10,32),cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,160),1)
    return bar


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    model    = load_model()
    detector = FaceDetector(confidence_threshold=0.5)
    # Auto-detect correct GradCAM layer based on model architecture
    gradcam_layer = None
    if model:
        try:
            input_shape = model.input_shape
            if input_shape[-1] == 1:
                # CNN grayscale model — use res_conv4
                gradcam_layer = "res_conv4"
            else:
                # MobileNetV2 — use last conv layer inside base model
                for layer in reversed(model.layers):
                    if hasattr(layer, 'layers'):
                        # Find last conv layer in MobileNetV2 sub-model
                        for sub in reversed(layer.layers):
                            if 'conv' in sub.name and len(sub.output_shape) == 4:
                                gradcam_layer = sub.name
                                break
                        break
                if not gradcam_layer:
                    gradcam_layer = "res_conv4"
            print(f"[GradCAM] Using layer: {gradcam_layer}")
        except Exception as eg:
            gradcam_layer = "res_conv4"
            print(f"[GradCAM] Layer detection failed: {eg}")
    # GradCAM setup — works with CNN only (grayscale 48x48)
    # Try dedicated CNN model first, fall back to main model if it is CNN
    grad_cam     = None
    gradcam_model= None
    try:
        is_rgb = model.input_shape[-1] == 3 if model else False
        if is_rgb:
            # MobileNetV2 is loaded — try loading CNN separately for GradCAM
            if os.path.exists(CNN_MODEL_PATH):
                import tensorflow as tf2
                gradcam_model = tf2.keras.models.load_model(CNN_MODEL_PATH, compile=False)
                grad_cam = GradCAM(gradcam_model, "res_conv4")
                print("[GradCAM] Using dedicated CNN model for GradCAM")
            else:
                print("[GradCAM] No CNN model found — save CNN as emotion_cnn.h5 for GradCAM")
                print("[GradCAM]   copy models/saved_model/emotion_model_cnn.h5 models/saved_model/emotion_cnn.h5")
        else:
            # Main model IS CNN — use it directly
            gradcam_model = model
            grad_cam = GradCAM(model, "res_conv4")
            print("[GradCAM] Using main CNN model")
    except Exception as eg:
        print(f"[GradCAM] Init failed: {eg}")
    sticker  = StickerPlayer(PANEL_W,PANEL_H)
    sticker.load("neutral")

    cartoon  = CartoonThread()
    if ANIME_READY: cartoon.start()

    # MediaPipe live mesh
    face_mesh=None; mp_draw=None; mp_styles=None; mp_face_mod=None
    try:
        mp_face_mod=mp.solutions.face_mesh
        face_mesh=mp_face_mod.FaceMesh(max_num_faces=1,refine_landmarks=True,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
        mp_draw=mp.solutions.drawing_utils
        mp_styles=mp.solutions.drawing_styles
        print("[App] MediaPipe ready.")
    except AttributeError:
        print("[App] MediaPipe new API.")

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened():
        print("[App] Cannot open webcam"); return

    print("\n"+"="*55)
    print("  A=AnimeGAN  G=GradCAM  R=Sticker  E=Emoji")
    print("  N=Next emoji  S=Save  SPACE=Pause  Q=Quit")
    print("="*55+"\n")

    anime_on=False; gradcam_on=False
    cur_emotion="neutral"; cur_conf=0.0
    last_pred=0.0; last_req=0.0; prev_time=time.time()
    smoother = EmotionSmoother(window=8)
    last_cartoon=None; CART_INT=1.5

    # Photo booth state
    countdown_active = False
    countdown_start  = 0.0
    countdown_secs   = 3       # 3-2-1 countdown
    burst_count      = 0       # group burst shots remaining
    burst_total      = 0

    # Video recorder state
    video_writer        = None
    video_recording     = False
    video_start         = 0.0
    video_path          = None
    video_frames_written= 0
    last_heatmap=None          # cached heatmap — recompute every 1.5s not every frame
    last_heatmap_time=0.0
    HEATMAP_INTERVAL=1.5
    gc_busy=False              # prevents stacking GradCAM threads

    # Emoji state
    emoji_frame=None; emoji_idx=0; generating=False

    # Emoji generation in background
    emoji_lock=threading.Lock()

    def generate_emoji(crop, emotion, conf, idx):
        nonlocal emoji_frame, generating
        generating=True
        try:
            lms=get_landmarks(crop)
            if lms is None:
                print("[Emoji] No landmarks — face camera directly")
                generating=False; return
            # ── NOTO ONLY for emoji panel ─────────────────────────────────
            path = ensure_noto(emotion, idx)
            if path is None:
                print("[Emoji] Could not download Noto emoji")
                generating=False; return
            noto = gif_best_frame(path, (512,512))
            if noto is None:
                print("[Emoji] Failed to read Noto GIF")
                generating=False; return
            print(f"[Emoji] Composing with Google Noto emoji...")
            ef = compose_emoji(crop, lms, noto, emotion, conf, PANEL_W, PANEL_H)
            with emoji_lock:
                emoji_frame=ef
            print("[Emoji] Done!")
        except Exception as e:
            print(f"[Emoji] Error: {e}")
        generating=False

    while True:
        cap.grab()
        ret,frame=cap.read()
        if not ret: break

        # Flip horizontally to remove mirror effect
        frame=cv2.flip(frame,1)

        now=time.time()
        raw=frame.copy()
        display=frame.copy()
        faces=detector.detect(frame)

        # Live mesh
        if face_mesh and mp_draw:
            try:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    for fl in res.multi_face_landmarks:
                        mp_draw.draw_landmarks(
                            display,fl,mp_face_mod.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
            except Exception: pass

        # Predict every 0.3s — faster response, smoother sync
        if now-last_pred>0.3 and faces:
            fx,fy=max(0,faces[0][0]),max(0,faces[0][1])
            fw=min(faces[0][2],frame.shape[1]-fx)
            fh=min(faces[0][3],frame.shape[0]-fy)
            if fw>0 and fh>0:
                crop=frame[fy:fy+fh,fx:fx+fw]
                e,c,preds=predict(model,crop)
                last_pred=now

                # Smooth predictions over last 8 frames
                smooth = smoother.update(np.array(preds))
                smooth_idx = int(np.argmax(smooth))
                e    = EMOTION_LABELS[smooth_idx]
                c    = float(smooth[smooth_idx])
                preds = smooth.tolist()

                cur_conf=c
                if e.lower()!=cur_emotion.lower():
                    cur_emotion=e; sticker.load(e)

                # GradCAM — skip if busy, run inline but quick (already fast enough)
                if gradcam_on and grad_cam:
                    if now - last_heatmap_time > HEATMAP_INTERVAL:
                        last_heatmap_time = now
                        try:
                            hm = grad_cam.compute(preprocess_face(crop),
                                                  int(np.argmax(preds)))
                            if hm.max() > 0:
                                last_heatmap = hm
                        except Exception as e2:
                            print(f"[GradCAM] {e2}")

            # Apply cached heatmap outside prediction block so it shows every frame
            if gradcam_on and last_heatmap is not None and faces:
                try:
                    fx2,fy2=max(0,faces[0][0]),max(0,faces[0][1])
                    fw2=min(faces[0][2],display.shape[1]-fx2)
                    fh2=min(faces[0][3],display.shape[0]-fy2)
                    if fw2>0 and fh2>0:
                        hm=cv2.resize(last_heatmap,(fw2,fh2))
                        roi=display[fy2:fy2+fh2,fx2:fx2+fw2]
                        display[fy2:fy2+fh2,fx2:fx2+fw2]=cv2.addWeighted(
                            roi,.4,hm,.6,0)
                except Exception as e2: print(f"[GradCAM overlay] {e2}")

        # Face boxes
        for i,(x,y,w,h) in enumerate(faces):
            fx,fy=max(0,x),max(0,y)
            fw=min(w,display.shape[1]-fx)
            fh=min(h,display.shape[0]-fy)
            if fw>0 and fh>0:
                draw_face_box(display,fx,fy,fw,fh,cur_emotion,cur_conf)

        # AnimeGAN — submit often, show cartoon when ready, live feed otherwise
        if ANIME_READY and anime_on:
            if now-last_req>CART_INT:
                cartoon.submit(raw); last_req=now
            r=cartoon.result()
            if r is not None and r.shape==raw.shape:
                last_cartoon=r
            if last_cartoon is not None:
                # Show cartoon at full quality
                display=last_cartoon.copy()
            # Always redraw face boxes on top
            for x,y,w,h in faces:
                fx,fy=max(0,x),max(0,y)
                fw2=min(w,display.shape[1]-fx)
                fh2=min(h,display.shape[0]-fy)
                if fw2>0 and fh2>0:
                    draw_face_box(display,fx,fy,fw2,fh2,cur_emotion,cur_conf)

        fps=1.0/max(now-prev_time,1e-9); prev_time=now
        draw_fps(display,fps)

        # Resize webcam
        cam=cv2.resize(display,(CAM_W,CAM_H))

        # Sticker panel
        sf=sticker.get_frame()
        sp=build_sticker_panel(sf,cur_emotion,cur_conf,sticker._busy,sticker.paused)

        # Emoji panel
        with emoji_lock:
            ef=emoji_frame.copy() if emoji_frame is not None else None
        ep=build_emoji_panel(ef,generating,emoji_frame is not None)

        # Bottom bar
        bb=build_bottom_bar(anime_on,gradcam_on,cur_emotion,cur_conf)

        # Combine: [webcam | sticker | emoji]
        top=np.hstack([cam,sp,ep])
        combined=np.vstack([top,bb])

        # ── Countdown overlay ──────────────────────────────────────────────
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = countdown_secs - int(elapsed)
            if remaining > 0:
                num = str(remaining)
                fs, thick = 5.0, 12
                (tw,th),_ = cv2.getTextSize(num,cv2.FONT_HERSHEY_SIMPLEX,fs,thick)
                tx,ty = (CAM_W-tw)//2, (CAM_H+th)//2
                cv2.putText(combined,num,(tx+4,ty+4),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),thick+4,cv2.LINE_AA)
                cv2.putText(combined,num,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),thick,cv2.LINE_AA)
                cx,cy=CAM_W//2,CAM_H//2
                cv2.ellipse(combined,(cx,cy),(80,80),0,0,int(360*(elapsed/countdown_secs)),(100,220,255),6)
            else:
                # Save CLEAN frame — anime or raw, no UI overlays at all
                if last_cartoon is not None:
                    shot = cv2.resize(last_cartoon.copy(),(CAM_W,CAM_H))
                else:
                    shot = raw.copy()
                # Flash on screen
                white = np.ones_like(combined)*255
                cv2.imshow("Emotion Recognition — Sticker — Emoji",
                    cv2.addWeighted(combined,0.3,white,0.7,0))
                cv2.waitKey(80)
                ts = int(time.time())
                if burst_total>0:
                    fname=os.path.join(PHOTOS_DIR,f"group_{burst_total-burst_count+1}of{burst_total}_{ts}.png")
                else:
                    fname=os.path.join(PHOTOS_DIR,f"anime_photo_{ts}.png")
                cv2.imwrite(fname,shot)
                print(f"[Photo] Saved -> {fname}")
                burst_count-=1
                if burst_count>0:
                    countdown_start=time.time(); countdown_secs=2
                else:
                    countdown_active=False; countdown_secs=3
                    print("[Photo] All shots done — check photos/ folder")

        # ── Video recording indicator ───────────────────────────────────────
        if video_recording and video_writer is not None:
            elapsed_v = time.time()-video_start
            # Throttle: only write a frame when real time says we should
            # VIDEO_FPS=15 means write one frame every 1/15 = 0.0667s
            expected_frames = int(elapsed_v * VIDEO_FPS)
            if expected_frames > video_frames_written:
                # Record CLEAN anime or raw frame — no emotion labels/boxes/UI
                if last_cartoon is not None:
                    vframe = cv2.resize(last_cartoon.copy(),(CAM_W,CAM_H))
                else:
                    vframe = raw.copy()
                video_writer.write(vframe)
                video_frames_written += 1
            if int(elapsed_v*2)%2==0:
                cv2.circle(combined,(20,20),8,(0,0,255),-1)
            cv2.putText(combined,f"REC {elapsed_v:.0f}s",(34,26),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2,cv2.LINE_AA)
            if elapsed_v>=30:
                video_writer.release(); video_writer=None; video_recording=False
                print(f"[Video] Auto-stopped — {video_frames_written} frames — saved to {video_path}")

        cv2.imshow("Emotion Recognition — Sticker — Emoji",combined)
        key=cv2.waitKey(1)&0xFF

        if key in (ord("q"),27): break
        elif key==ord("a"):
            if ANIME_READY:
                anime_on=not anime_on
                print(f"[App] AnimeGAN: {'ON' if anime_on else 'OFF'}")
            else: print("[App] AnimeGAN not available")
        elif key==ord("g"):
            gradcam_on=not gradcam_on
            print(f"[App] GradCAM: {'ON' if gradcam_on else 'OFF'}")
        elif key==ord("r"):
            sticker.refresh(); print("[App] Sticker refreshed")
        elif key==ord(" "):
            sticker.paused=not sticker.paused
            print(f"[App] Sticker {'paused' if sticker.paused else 'resumed'}")
        elif key==ord("e"):
            if not generating and faces:
                fx,fy=max(0,faces[0][0]),max(0,faces[0][1])
                fw=min(faces[0][2],frame.shape[1]-fx)
                fh=min(faces[0][3],frame.shape[0]-fy)
                if fw>0 and fh>0:
                    crop=frame[fy:fy+fh,fx:fx+fw]
                    print(f"[Emoji] Generating {cur_emotion}...")
                    threading.Thread(target=generate_emoji,
                        args=(crop,cur_emotion,cur_conf,emoji_idx),
                        daemon=True).start()
            elif not faces:
                print("[Emoji] No face detected")
        elif key==ord("n"):
            if not generating and faces:
                emoji_idx+=1
                fx,fy=max(0,faces[0][0]),max(0,faces[0][1])
                fw=min(faces[0][2],frame.shape[1]-fx)
                fh=min(faces[0][3],frame.shape[0]-fy)
                if fw>0 and fh>0:
                    crop=frame[fy:fy+fh,fx:fx+fw]
                    threading.Thread(target=generate_emoji,
                        args=(crop,cur_emotion,cur_conf,emoji_idx),
                        daemon=True).start()
        elif key==ord("s"):
            with emoji_lock:
                if emoji_frame is not None:
                    fname=os.path.join(SAVE_DIR,
                        f"emoji_{cur_emotion}_{int(time.time())}.png")
                    cv2.imwrite(fname,emoji_frame)
                    print(f"[App] Saved -> {fname}")
                else:
                    print("[App] Press E first to generate emoji")

        # ── P  = Photo booth (single anime shot, 3-2-1 countdown) ────────
        elif key==ord("p"):
            if not countdown_active:
                countdown_active = True
                countdown_start  = time.time()
                countdown_secs   = 3
                burst_count      = 1
                burst_total      = 0
                print("[Photo] Countdown started — smile!")

        # ── B  = Burst mode (3 group photos, 2s apart) ───────────────────
        elif key==ord("b"):
            if not countdown_active:
                countdown_active = True
                countdown_start  = time.time()
                countdown_secs   = 3
                burst_count      = 3
                burst_total      = 3
                print("[Photo] Burst mode — 3 shots incoming!")

        # ── V  = Start / stop video recording ────────────────────────────
        elif key==ord("v"):
            if not video_recording:
                ts   = int(time.time())
                video_path = os.path.join(VIDEOS_DIR, f"emotion_video_{ts}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(video_path, fourcc, 15, (CAM_W, CAM_H))
                video_recording      = True
                video_start          = time.time()
                video_frames_written = 0
                print(f"[Video] Recording started -> {video_path}")
            else:
                if video_writer: video_writer.release(); video_writer=None
                video_recording = False
                print(f"[Video] Stopped — saved to {video_path}")

    if ANIME_READY: cartoon.stop()
    if face_mesh: face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[App] Closed.")


if __name__=="__main__":
    run()