# MultiFace Emotion Recognition System

A real-time multi-face emotion detection desktop application with animated stickers, Google Noto emoji generation, AnimeGAN2 neural style transfer, Grad-CAM explainability, photo booth, video recording, and a clean Tkinter dark-theme UI.

---

## Features

- **Real-time emotion detection** — detects 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from live webcam
- **Multi-face support** — detects and classifies multiple faces simultaneously
- **Grad-CAM heatmap** — visualises which facial regions influenced the prediction (red = important, blue = ignored)
- **AnimeGAN2 cartoon** — neural anime style transfer on live webcam feed, GPU accelerated (RTX 3050)
- **Giphy sticker panel** — animated GIF matching detected emotion, auto-refreshes on emotion change
- **Google Noto emoji** — personalised emoji using MediaPipe 468-point facial landmarks
- **MediaPipe face mesh** — live facial landmark overlay in webcam
- **Tkinter dark UI** — clean navy dark theme with per-emotion colour bars, works on any Python version
- **Photo booth** — 3-2-1 countdown captures clean anime photo, saved to `photos/`
- **Burst mode** — 3 automatic group shots 2 seconds apart
- **Video recording** — real-time video capture at correct 15fps, saved to `videos/`

---

## UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  ✦ Emotion Recognition              87%  [Happy badge]  │
├──────────────────────────┬──────────────────────────────┤
│                          │  EMOTION CONFIDENCE          │
│   LIVE FEED              │  ████████████ Happy    87%   │
│                          │  ███          Neutral  18%   │
│   webcam + face box      │  █            Sad       5%   │
│   GradCAM heatmap        │  ...                         │
│   AnimeGAN cartoon       ├──────────────┬───────────────┤
│                          │  STICKER     │  EMOJI        │
│  [AnimeGAN] [Grad-CAM]   │  Giphy GIF   │  Noto emoji   │
│                 [Snap]   │  [↻ Refresh] │  [↻ Next]     │
├──────────────────────────┴──────────────┴───────────────┤
│  Running                                       FPS: 28  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project/
├── ui_app.py                ← Tkinter dark-theme UI (main launcher, no PyQt5 needed)
├── main_app.py              ← OpenCV combined app (backup/fallback)
├── webcam_app.py            ← Standalone webcam with emotion + AnimeGAN
├── train_model.py           ← Training script (CNN + MobileNetV2)
├── emotion_model.py         ← Model architectures (CNN Residual + MobileNetV2)
├── face_detector.py         ← OpenCV DNN face detector + Haar fallback
├── grad_cam.py              ← Grad-CAM heatmap (Keras 3 compatible)
├── animegan_inference.py    ← AnimeGAN2 PyTorch wrapper (auto GPU/CPU)
├── preprocessing.py         ← Face preprocessing + EMOTION_LABELS
├── display_utils.py         ← OpenCV drawing helpers
├── download_models.py       ← Downloads DNN face detector weights
├── face_landmarker.task     ← MediaPipe face landmarker model
├── requirements.txt
├── README.md
│
├── backup/                  ← Fallback apps for presentation
│   ├── combined_app.py
│   ├── sticker_app.py
│   └── Emoji generator
│
├── models/
│   └── saved_model/
│       ├── emotion_model.h5     ← Trained emotion model
│       └── class_indices.json   ← Class label mapping
│
├── model/
│   └── face_paint_512_v2.pt     ← AnimeGAN2 weights
│
├── animegan2-pytorch/           ← AnimeGAN2 repo (cloned)
│
├── dataset/
│   └── train/emotion/           ← FER2013 training images (6 classes)
│
├── stickers/                    ← Auto-downloaded Giphy/Noto GIFs
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── fear/
│   ├── surprise/
│   ├── neutral/
│   └── disgust/
│
├── saved_emojis/                ← Saved emoji + snapshot PNG outputs
├── photos/                      ← Photo booth captures (anime_photo_*.png, group_*of*.png)
├── videos/                      ← Video recordings (emotion_video_*.avi)
└── logs/                        ← TensorBoard training logs
```

---

## Requirements

- **Python 3.11.9** — download from:
  `https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe`
- **Windows 10/11 64-bit**
- **Webcam**
- **NVIDIA RTX 3050** (or any CUDA-capable GPU) — optional, CPU fallback works
- **4GB RAM minimum**, 8GB recommended
- **Internet connection** — for Giphy sticker fetching (first run)

---

## Installation

### 1. Install Python 3.11.9
Download and run: `https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe`
- ✅ Check **"Add Python 3.11 to PATH"**
- Click **Install Now**

### 2. Clone the repo
```bash
git clone <your-repo-url>
cd project
```

### 3. Download face detector weights
```bash
py -3.11 python download_models.py
```

### 4. Set up AnimeGAN2
```bash
git clone https://github.com/bryandlee/animegan2-pytorch
# Place face_paint_512_v2.pt in model/ folder
```

### 5. Install Python packages
```bash
py -3.11 -m pip install -r requirements.txt
```

### 6. Install PyTorch with GPU support (RTX 3050)
```bash
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 7. Download FER2013 dataset
Download from https://www.kaggle.com/datasets/msambare/fer2013 and place in:
```
dataset/train/emotion/angry/
dataset/train/emotion/happy/
dataset/train/emotion/sad/
...
```

### 8. Train the model
```bash
# CNN with residual blocks (~68% accuracy, faster)
py -3.11 train_model.py --arch cnn --epochs 60 --batch 64

# MobileNetV2 (~72% accuracy, slower)
py -3.11 train_model.py --arch mobilenet --epochs 50 --batch 32
```

---

## Running

### Main UI (recommended)
```bash
py -3.11 ui_app.py
```

### Fallback OpenCV app
```bash
py -3.11 main_app.py
```

### Standalone webcam only
```bash
py -3.11 webcam_app.py
py -3.11 webcam_app.py --no-anime      # no cartoon
py -3.11 webcam_app.py --no-gradcam    # no heatmap
```

---

## Controls

| Key | Action |
|-----|--------|
| `A` | Toggle AnimeGAN2 cartoon on/off |
| `G` | Toggle Grad-CAM heatmap on/off |
| `E` | Capture face and generate Noto emoji |
| `N` | Next emoji variation |
| `R` | Refresh Giphy sticker |
| `P` | Photo booth — 3-2-1 countdown, saves clean anime photo to `photos/` |
| `B` | Burst mode — 3 automatic shots, 2 seconds apart, saved to `photos/` |
| `V` | Start / stop video recording — saves clean anime video to `videos/` |
| `S` | Save snapshot as PNG |
| `Q` / `ESC` | Quit |

---

## Model Performance

| Model | Val Accuracy | Architecture | Notes |
|-------|-------------|-------------|-------|
| CNN (original) | ~62% | Plain CNN, 4 conv | First version |
| CNN Residual | ~68% | Residual blocks + focal loss | Current |
| MobileNetV2 | ~72% | Transfer learning, 2-phase fine-tune | Best accuracy |

Training improvements over original:
- **Focal loss** (γ=2) instead of cross-entropy — handles class imbalance
- **Residual blocks** — skip connections prevent vanishing gradients
- **Sqrt class weights** — balances minority classes without collapse
- **ReduceLROnPlateau** — stable learning rate decay

---

## Dataset

**FER2013** — grayscale 48×48 face images across 7 emotion classes.

| Emotion | Train samples |
|---------|--------------|
| Happy | 8,989 |
| Neutral | 6,198 |
| Sad | 6,077 |
| Fear | 5,121 |
| Angry | 4,953 |
| Surprise | 4,002 |
| Disgust | 547 |

Note: Human accuracy on FER2013 is ~65%.

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| UI Framework | Tkinter | Built-in (Python stdlib) |
| Deep Learning | TensorFlow / Keras | 2.13.0 |
| Face Detection | OpenCV DNN (ResNet SSD) | 4.8+ |
| Style Transfer | AnimeGAN2 (PyTorch) | 2.6+ cu124 |
| Explainability | Grad-CAM | Custom |
| Face Landmarks | MediaPipe | 0.10+ |
| Stickers | Giphy API | v1 |
| Emoji | Google Noto Emoji | Apache 2.0 |
| Language | Python | 3.11.9 |

---

## GPU Support

AnimeGAN2 auto-detects the best available device:

| Hardware | Device | Speed |
|----------|--------|-------|
| NVIDIA RTX 3050 | `cuda` | ~80ms/frame |
| Apple M1/M2/M3 | `mps` | ~120ms/frame |
| CPU only | `cpu` | ~2–3s/frame |

---

## Presentation Tips

1. **Pre-download stickers** — run the app before the presentation and trigger all 7 emotions to cache Giphy GIFs locally
2. **Pre-download Noto emojis** — press `E` + `N` for each emotion
3. **Use backup apps** if `ui_app.py` crashes — `backup/combined_app.py` or `backup/sticker_app.py`
4. **Record a screen capture** as last resort if live demo fails
5. **Good lighting** — face the window or lamp for best emotion detection

---

## Known Limitations

- TensorFlow runs CPU-only on native Windows ≥ 2.11 (use WSL2 for GPU TF)
- AnimeGAN2 updates every 1.5s on CPU, ~0.3s on GPU
- Emotion accuracy drops with poor lighting or extreme face angles
- Giphy API free tier has hourly rate limits

---

## Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [AnimeGAN2](https://github.com/bryandlee/animegan2-pytorch) by bryandlee
- [MediaPipe](https://developers.google.com/mediapipe) by Google
- [Google Noto Emoji](https://fonts.google.com/noto/specimen/Noto+Emoji) — Apache 2.0
- [Giphy API](https://developers.giphy.com)