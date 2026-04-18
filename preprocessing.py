"""
preprocessing.py
Improved face preprocessing with tighter crop and CLAHE normalisation.

Key improvements over original:
  - Tighter crop: expand bbox slightly but clip to frame — removes neck/background
  - CLAHE equalisation: handles dark/bright lighting conditions much better
  - Gaussian blur before resize: reduces aliasing on small 48x48 target
"""
import cv2
import numpy as np

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = 48


def preprocess_face(face_img: np.ndarray, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Preprocess a raw face crop for the CNN.
    Returns float32 array of shape (1, 48, 48, 1).

    Improvements:
      - Converts to grayscale
      - Applies CLAHE for lighting normalisation
      - Light Gaussian blur before downscale (anti-aliasing)
      - Normalises to [0, 1]
    """
    # Grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()

    # CLAHE — adaptive histogram equalisation handles dark/bright faces
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # Gentle blur before resize to avoid aliasing
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Resize to target
    resized = cv2.resize(gray, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)

    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, target_size, target_size, 1)


def preprocess_face_rgb(face_img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Preprocess for RGB transfer-learning models (MobileNetV2)."""
    rgb     = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size))
    return (resized.astype(np.float32) / 255.0).reshape(1, target_size, target_size, 3)


def tight_crop(frame: np.ndarray, x: int, y: int, w: int, h: int,
               pad_frac: float = 0.15) -> np.ndarray:
    """
    Expand face bbox slightly (pad_frac=15%) to include full face,
    then crop tightly. Removes neck and background better than raw bbox.
    Use this to get face_crop before calling preprocess_face().
    """
    fh, fw = frame.shape[:2]
    pad_x  = int(w * pad_frac)
    pad_y  = int(h * pad_frac)
    x1 = max(0,  x - pad_x)
    y1 = max(0,  y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    return frame[y1:y2, x1:x2]


def augment_image(img: np.ndarray) -> np.ndarray:
    """Basic augmentation: random horizontal flip + small rotation."""
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    angle = np.random.uniform(-10, 10)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))