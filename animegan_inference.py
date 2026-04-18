"""
animegan_inference.py
AnimeGAN2 style transfer using animegan2-pytorch.

Requirements (already in your project):
    - animegan2-pytorch/  folder (git cloned)
    - model/face_paint_512_v2.pt  (weights)
    - pip install torch torchvision

Returns original frame unchanged if model not available.
"""
import os
import sys
import cv2
import numpy as np

_HERE           = os.path.dirname(os.path.abspath(__file__))
ANIMEGAN_WEIGHT = os.path.join(_HERE, "model", "face_paint_512_v2.pt")
REPO_DIR        = os.path.join(_HERE, "animegan2-pytorch")

_model = None


def _load():
    global _model
    if _model is not None:
        return _model

    try:
        import torch
    except ImportError:
        print("[AnimeGAN] PyTorch not installed.")
        return None

    if not os.path.exists(ANIMEGAN_WEIGHT):
        print(f"[AnimeGAN] Weights not found: {ANIMEGAN_WEIGHT}")
        return None

    if not os.path.isdir(REPO_DIR):
        print(f"[AnimeGAN] Repo not found: {REPO_DIR}")
        return None

    try:
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
        from model import Generator

        # Auto-select best available device
        # Note: CUDA inside Qt threads causes memory violations on some systems
        # Use environment variable ANIMEGAN_DEVICE to override
        import os as _os
        forced = _os.environ.get("ANIMEGAN_DEVICE", "").lower()
        if forced in ("cuda","cpu","mps"):
            device = forced
        elif torch.cuda.is_available():
            device = "cuda"        # NVIDIA GPU (RTX 3050, etc.)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"         # Apple Silicon (M1/M2/M3)
        else:
            device = "cpu"         # Fallback — any machine
        gen    = Generator()
        gen.load_state_dict(torch.load(ANIMEGAN_WEIGHT, map_location=device))
        gen.eval().to(device)
        _model = (gen, device)
        print(f"[AnimeGAN] Ready on {device}")
        return _model
    except Exception as e:
        print(f"[AnimeGAN] Load error: {e}")
        return None


def apply_anime_style(frame_bgr: np.ndarray, size: int = 512) -> np.ndarray:
    """Apply AnimeGAN2 to BGR frame. Returns original if unavailable."""
    result = _load()
    if result is None:
        return frame_bgr

    model, device = result
    try:
        import torch
        from torchvision import transforms

        h, w   = frame_bgr.shape[:2]
        scale  = size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        small  = cv2.resize(frame_bgr, (nw, nh))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])
        tensor = t(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)

        out = out.squeeze(0).cpu().permute(1,2,0).numpy()
        out = ((out * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)

    except Exception as e:
        print(f"[AnimeGAN] Error: {e}")
        return frame_bgr


def is_available() -> bool:
    return os.path.exists(ANIMEGAN_WEIGHT) and os.path.isdir(REPO_DIR)