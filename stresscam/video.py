import cv2
import numpy as np
import mediapipe as mp

from .config import Config


class VideoStream:
    def __init__(self, cfg: Config):
        self.cap = cv2.VideoCapture(cfg.device_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        if cfg.frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Falha ao ler da webcam.")
        return cv2.flip(frame, 1)  # espelhar melhora UX

    def release(self):
        self.cap.release()


class FaceProcessor:
    def __init__(self, cfg: Config):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.smooth_bbox = None
        self.alpha = cfg.bbox_smooth_alpha

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def landmarks(self, results):
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def bbox_from_landmarks(self, landmarks, frame_shape):
        h, w, _ = frame_shape
        xs = [p.x * w for p in landmarks.landmark]
        ys = [p.y * h for p in landmarks.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bbox = np.array([x1, y1, x2, y2], dtype=float)
        if self.smooth_bbox is None:
            self.smooth_bbox = bbox
        else:
            self.smooth_bbox = self.smooth_bbox + self.alpha * (bbox - self.smooth_bbox)
        return self.smooth_bbox


def normalize_lighting(frame, cfg: Config):
    # equalização adaptativa no canal de luminância para reduzir variação de luz
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tiles)
    y_eq = clahe.apply(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
