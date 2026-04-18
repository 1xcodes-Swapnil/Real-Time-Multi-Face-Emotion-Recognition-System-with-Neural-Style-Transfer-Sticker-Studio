"""
webcam_app.py
Real-time multi-face emotion recognition.

Controls:
  Q / ESC  quit
  A        toggle cartoon style on/off
  G        toggle Grad-CAM overlay
  1        hayao   (OpenCV cartoon)
  2        shinkai (neural CartoonGAN)
  3        hosoda  (OpenCV cartoon)
  4        paprika (neural CartoonGAN)
"""
import argparse
import sys
import os
import time
import threading
import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_ROOT) == "app":
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from face_detector        import FaceDetector
from preprocessing        import preprocess_face, tight_crop, EMOTION_LABELS
from grad_cam             import GradCAM
from animegan_inference   import apply_anime_style, is_available
from display_utils        import draw_face_box, draw_fps, make_info_panel

MODEL_PATH  = os.path.join(_ROOT, "models", "saved_model", "emotion_model.h5")
LAST_CONV   = "block3_conv2"   # updated for new residual model
WINDOW_NAME = "MultiFace Emotion Recognition"


# ── Temporal smoother — averages predictions over last N frames ────────────────
class EmotionSmoother:
    """
    Keeps a rolling average of raw softmax probabilities over the last
    WINDOW frames. Eliminates frame-to-frame flickering without lag.
    """
    def __init__(self, num_classes=7, window=8):
        self.window  = window
        self.history = []   # list of prob arrays

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(probs.copy())
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)


# ── Model ──────────────────────────────────────────────────────────────────────
def load_emotion_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        model.predict(np.zeros((1, 48, 48, 1), dtype=np.float32), verbose=0)
        print(f"[App] Model loaded.")
        return model
    except Exception as e:
        print(f"[App] Model error: {e} — DEMO mode.")
        return None


def predict_emotion(model, face_bgr):
    if model is None:
        probs = np.random.dirichlet(np.ones(len(EMOTION_LABELS)))
        idx   = int(np.argmax(probs))
        return EMOTION_LABELS[idx], float(probs[idx]), probs
    img   = preprocess_face(face_bgr)   # now includes CLAHE + anti-alias blur
    preds = model.predict(img, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return EMOTION_LABELS[idx], float(preds[idx]), preds


# ── Background cartoon thread ──────────────────────────────────────────────────
class CartoonThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.input_frame  = None
        self.output_frame = None
        self.lock         = threading.Lock()
        self.event        = threading.Event()
        self.running      = True

    def request(self, frame):
        with self.lock:
            self.input_frame = frame.copy()
        self.event.set()

    def get_result(self):
        with self.lock:
            return self.output_frame.copy() if self.output_frame is not None else None

    def run(self):
        while self.running:
            self.event.wait(timeout=1.0)
            self.event.clear()
            with self.lock:
                frame = self.input_frame.copy() if self.input_frame is not None else None
            if frame is None:
                continue
            try:
                result = apply_anime_style(frame, size=512)
                result = cv2.resize(result, (frame.shape[1], frame.shape[0]))
                with self.lock:
                    self.output_frame = result
            except Exception as e:
                print(f"[CartoonGAN] Thread error: {e}")

    def stop(self):
        self.running = False
        self.event.set()


# ── Main loop ──────────────────────────────────────────────────────────────────
def run(cam_index=0, video_path=None, use_anime=True, use_gradcam=True):

    detector      = FaceDetector(confidence_threshold=0.5)
    model         = load_emotion_model()
    grad_cam      = GradCAM(model, LAST_CONV) if (model and use_gradcam) else None

    source = video_path if video_path else cam_index
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[App] Cannot open source: {source}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cartoon_thread = CartoonThread()
    cartoon_thread.start()

    print(f"\n{'='*60}")
    print("Q/ESC=quit  A=toggle live/cartoon  G=GradCAM")
    print(f"{'='*60}\n")

    prev_time        = time.time()
    anime_on         = use_anime
    gradcam_on       = use_gradcam
    last_cartoon     = None
    request_sent     = False
    last_render_time = 0.0
    RENDER_INTERVAL  = 2.0
    smoothers        = {}   # face_id → EmotionSmoother

    while True:
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame     = frame.copy()
        faces         = detector.detect(frame)
        emotions_data = []

        for face_id, (x, y, w, h) in enumerate(faces):
            fx, fy = max(0, x), max(0, y)
            fw     = min(w, frame.shape[1] - fx)
            fh     = min(h, frame.shape[0] - fy)
            if fw <= 0 or fh <= 0:
                continue

            # Tight crop — removes neck/background padding
            face_crop = tight_crop(frame, fx, fy, fw, fh, pad_frac=0.10)
            if face_crop.size == 0:
                face_crop = frame[fy:fy+fh, fx:fx+fw]

            emotion, confidence, preds = predict_emotion(model, face_crop)

            # Temporal smoothing per face — kills flickering
            if face_id not in smoothers:
                smoothers[face_id] = EmotionSmoother(window=8)
            smooth_probs = smoothers[face_id].update(preds)
            smooth_idx   = int(np.argmax(smooth_probs))
            emotion      = EMOTION_LABELS[smooth_idx]
            confidence   = float(smooth_probs[smooth_idx])

            emotions_data.append({"id": face_id+1, "emotion": emotion,
                                   "confidence": confidence})

            # GradCAM overlay
            if gradcam_on and grad_cam is not None:
                try:
                    img_arr = preprocess_face(face_crop)
                    heatmap = grad_cam.compute(img_arr, class_idx=smooth_idx)
                    hm      = cv2.resize(heatmap, (fw, fh))
                    roi     = frame[fy:fy+fh, fx:fx+fw]
                    frame[fy:fy+fh, fx:fx+fw] = cv2.addWeighted(roi, 0.55, hm, 0.45, 0)
                    cv2.putText(frame, "GradCAM", (fx, fy+fh+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                except Exception as e:
                    print(f"[GradCAM] Error: {e}")

            draw_face_box(frame, fx, fy, fw, fh, emotion, confidence)

        # ── Cartoon thread ─────────────────────────────────────────────────
        now = time.time()
        if request_sent:
            new_result = cartoon_thread.get_result()
            if new_result is not None:
                rh, rw = new_result.shape[:2]
                if rh == raw_frame.shape[0] and rw == raw_frame.shape[1]:
                    last_cartoon = new_result.copy()
                request_sent = False

        if not request_sent and (now - last_render_time) >= RENDER_INTERVAL:
            cartoon_thread.request(raw_frame.copy())
            request_sent     = True
            last_render_time = now

        # ── Display ────────────────────────────────────────────────────────
        if anime_on and last_cartoon is not None:
            display_frame = last_cartoon.copy()
            for i, (x, y, w, h) in enumerate(faces):
                fx, fy = max(0, x), max(0, y)
                fw2 = min(w, display_frame.shape[1] - fx)
                fh2 = min(h, display_frame.shape[0] - fy)
                if fw2 > 0 and fh2 > 0 and i < len(emotions_data):
                    draw_face_box(display_frame, fx, fy, fw2, fh2,
                                  emotions_data[i]["emotion"],
                                  emotions_data[i]["confidence"])
            cv2.putText(display_frame, "AnimeGAN2  [A=live]",
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        else:
            display_frame = frame
            label = "Live  [A=cartoon]" if not anime_on else "AnimeGAN2  [warming up...]"
            cv2.putText(display_frame, label,
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time = curr_time
        draw_fps(display_frame, fps)

        if emotions_data:
            panel = make_info_panel(emotions_data, panel_h=display_frame.shape[0])
            display_frame = np.hstack([display_frame, panel])

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            anime_on = not anime_on
            print(f"[App] {'Cartoon' if anime_on else 'Live feed'}")
        elif key == ord("g"):
            gradcam_on = not gradcam_on
            print(f"[App] GradCAM: {'ON' if gradcam_on else 'OFF'}")


    cartoon_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[App] Closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",        type=int, default=0)
    parser.add_argument("--video",      type=str, default=None)
    parser.add_argument("--no-anime",   action="store_true")
    parser.add_argument("--no-gradcam", action="store_true")
    args = parser.parse_args()
    run(cam_index=args.cam, video_path=args.video,
        use_anime=not args.no_anime, use_gradcam=not args.no_gradcam)