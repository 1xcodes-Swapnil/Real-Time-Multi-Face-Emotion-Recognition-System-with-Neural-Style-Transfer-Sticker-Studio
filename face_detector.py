"""
utils/face_detector.py
Multi-face detection using OpenCV's DNN face detector (preferred)
with a Haar Cascade fallback.
"""
import os
import cv2
import numpy as np


class FaceDetector:
    """
    Detects multiple faces in a frame.
    Uses OpenCV's DNN-based detector when model files are available,
    otherwise falls back to Haar Cascade.
    """

    DNN_PROTO = "models/deploy.prototxt"
    DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._load_detector()

    def _load_detector(self):
        if os.path.exists(self.DNN_PROTO) and os.path.exists(self.DNN_MODEL):
            self.net = cv2.dnn.readNetFromCaffe(self.DNN_PROTO, self.DNN_MODEL)
            self.mode = "dnn"
            print("[FaceDetector] Using DNN detector.")
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.mode = "haar"
            print("[FaceDetector] DNN model not found — using Haar Cascade fallback.")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Returns a list of (x, y, w, h) bounding boxes for detected faces.
        """
        if self.mode == "dnn":
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def _detect_haar(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return []
        return [(x, y, w, h) for (x, y, w, h) in faces]
