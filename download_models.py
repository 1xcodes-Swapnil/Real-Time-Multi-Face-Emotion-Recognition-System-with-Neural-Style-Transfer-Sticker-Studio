"""
download_models.py
One-shot helper that downloads all optional model files:
  1. OpenCV DNN face detector  (deploy.prototxt + .caffemodel)
  2. Prompts about AnimeGANv2  (manual step — too large to auto-download)

Run once before starting the app:
    python download_models.py
"""

import os
import sys
import urllib.request

# ──────────────────────────────────────────────
# OpenCV DNN face-detector files
# ──────────────────────────────────────────────
DNN_FILES = {
    "models/deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "models/res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}

os.makedirs("models", exist_ok=True)


def _download(dest: str, url: str) -> bool:
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  [SKIP] {dest} already exists ({size:,} bytes)")
        return True
    print(f"  Downloading {dest} …")
    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                bar = int(pct / 5)
                print(
                    f"\r    [{'#' * bar}{' ' * (20 - bar)}] {pct:5.1f}%",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
        print(f"  [OK]   {dest} ({os.path.getsize(dest):,} bytes)")
        return True
    except Exception as e:
        print(f"\n  [FAIL] {dest}: {e}")
        return False


print("\n=== Downloading OpenCV DNN face-detector files ===\n")
all_ok = all(_download(dest, url) for dest, url in DNN_FILES.items())

if all_ok:
    print(
        "\n  DNN face detector ready.\n"
        "   The app will now use the more accurate DNN detector instead of Haar Cascade.\n"
    )
else:
    print(
        "\n  Some files could not be downloaded.\n"
        "   The app will fall back to Haar Cascade — still functional.\n"
        "   Check your internet connection and retry.\n"
    )

# ──────────────────────────────────────────────
# AnimeGANv2 reminder
# ──────────────────────────────────────────────
anime_weight = "model/face_paint_512_v2.pt"
if os.path.exists(anime_weight):
    print("  AnimeGANv2 weights found.")
else:
    print(
        "ℹ   AnimeGANv2 weights NOT found (optional).\n"
        "   To enable anime-style output:\n"
        "     1. git clone https://github.com/bryandlee/animegan2-pytorch\n"
        "     2. Download face_paint_512_v2.pt from the repo's Releases page\n"
        "     3. Place it at: model/face_paint_512_v2.pt\n"
        "   The app works fine without it (--no-anime flag is set automatically).\n"
    )

# ──────────────────────────────────────────────
# Emotion model reminder
# ──────────────────────────────────────────────
if os.path.exists("models/saved_model/emotion_model.h5"):
    print("  Trained emotion model found.")
else:
    print(
        "ℹ   No trained emotion model found.\n"
        "   Train it first:\n"
        "     python train_model.py --arch cnn --epochs 50 --batch 64\n"
        "   The app will run in DEMO mode (random predictions) until a model is trained.\n"
    )

print()
