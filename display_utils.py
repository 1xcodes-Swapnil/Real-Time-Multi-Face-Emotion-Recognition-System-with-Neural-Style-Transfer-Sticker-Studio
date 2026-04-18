"""
app/display_utils.py
Rendering helpers: bounding boxes, emotion labels, Grad-CAM overlays,
and anime face thumbnails drawn onto OpenCV frames.
"""
import cv2
import numpy as np

# Colors per emotion (BGR)
EMOTION_COLORS = {
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   140, 0),
    "Fear":     (148, 0,   211),
    "Happy":    (0,   215, 255),
    "Neutral":  (200, 200, 200),
    "Sad":      (255, 144, 30),
    "Surprise": (0,   165, 255),
}
DEFAULT_COLOR = (255, 255, 255)


def draw_face_box(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    emotion: str,
    confidence: float,
) -> np.ndarray:
    """Draw a colored bounding box and emotion label on the frame."""
    color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)
    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Label background
    label = f"{emotion} ({confidence:.0%})"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y - lh - 10), (x + lw + 6, y), color, -1)
    # Label text
    cv2.putText(
        frame, label, (x + 3, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
    )
    return frame


def embed_thumbnail(
    frame: np.ndarray,
    thumb: np.ndarray,
    x: int, y: int, w: int, h: int,
    thumb_size: int = 80,
    label: str = "",
    border_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Embed a small thumbnail in the top-right of the face bounding box."""
    thumb_resized = cv2.resize(thumb, (thumb_size, thumb_size))
    tx, ty = x + w - thumb_size, y
    tx = max(0, tx)
    ty = max(0, ty)
    # Clip to frame bounds
    if ty + thumb_size > frame.shape[0] or tx + thumb_size > frame.shape[1]:
        return frame
    # Border
    cv2.rectangle(frame, (tx - 2, ty - 2), (tx + thumb_size + 2, ty + thumb_size + 2),
                  border_color, 2)
    frame[ty:ty + thumb_size, tx:tx + thumb_size] = thumb_resized
    if label:
        cv2.putText(frame, label, (tx, ty + thumb_size + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


def make_info_panel(
    emotions_data: list[dict],
    panel_w: int = 220,
    panel_h: int = 480,
) -> np.ndarray:
    """
    Sidebar panel listing each detected face with emotion + confidence bar.
    emotions_data: list of {"id": int, "emotion": str, "confidence": float}
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(panel, "Detected Faces", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.line(panel, (10, 32), (panel_w - 10, 32), (100, 100, 100), 1)

    for i, data in enumerate(emotions_data):
        y_offset = 55 + i * 70
        if y_offset + 60 > panel_h:
            break
        color = EMOTION_COLORS.get(data["emotion"], DEFAULT_COLOR)
        cv2.putText(panel, f"Face {data['id']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(panel, data["emotion"], (10, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        # Confidence bar
        bar_w = int((panel_w - 20) * data["confidence"])
        cv2.rectangle(panel, (10, y_offset + 30), (panel_w - 10, y_offset + 45),
                      (60, 60, 60), -1)
        cv2.rectangle(panel, (10, y_offset + 30), (10 + bar_w, y_offset + 45),
                      color, -1)
        cv2.putText(panel, f"{data['confidence']:.0%}", (panel_w - 50, y_offset + 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return panel
