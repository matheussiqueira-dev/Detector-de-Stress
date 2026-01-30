import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import time

from .config import Config

_CLAHE_CACHE: dict[tuple[float, tuple[int, int]], cv2.CLAHE] = {}


class VideoStream:
    def __init__(self, cfg: Config):
        cv2.setUseOptimized(True)
        self.cfg = cfg
        source = cfg.device_name if cfg.device_name else cfg.device_index
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.buffer_size is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
        if cfg.frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        if cfg.frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não consegui abrir a câmera: {source}")
        self._failures = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= 3:
                raise RuntimeError("Falha ao ler da webcam.")
            time.sleep(0.02)  # breve backoff antes de tentar novamente
            return self.read()
        self._failures = 0
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def release(self):
        self.cap.release()


TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def _download_model(dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path):
        urllib.request.urlretrieve(TASK_MODEL_URL, dst_path)
    return dst_path


class FaceProcessor:
    def __init__(self, cfg: Config):
        self.alpha = cfg.bbox_smooth_alpha
        self.smooth_bbox = None
        self.using_solutions = False

        if hasattr(mp, "solutions"):
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    refine_landmarks=True, max_num_faces=1,
                    min_detection_confidence=0.6, min_tracking_confidence=0.6
                )
                self.using_solutions = True
            except Exception:
                self.using_solutions = False

        if not self.using_solutions:
            # fallback: MediaPipe Tasks FaceLandmarker
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            package_dir = os.path.dirname(mp.__file__)
            target_abs = os.path.join(package_dir, "face_landmarker.task")
            _download_model(target_abs)
            with open(target_abs, "rb") as f:
                model_bytes = f.read()
            base_opts = mp_python.BaseOptions(model_asset_buffer=model_bytes)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            self._ts_ms = 0

    def detect(self, frame):
        if self.using_solutions:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.face_mesh.process(rgb)
        # tasks API
        self._ts_ms += 33  # ~30 fps timestamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)
        return result

    def landmarks(self, results):
        if self.using_solutions:
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0]
        # tasks result
        if not results.face_landmarks:
            return None
        class _Wrapper:
            def __init__(self, lmks):
                self.landmark = lmks
        return _Wrapper(results.face_landmarks[0])

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
    clahe = _CLAHE_CACHE.setdefault(
        (cfg.clahe_clip, cfg.clahe_tiles),
        cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tiles),
    )
    y_eq = clahe.apply(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def preprocess_frame(frame, cfg: Config):
    """
    Centraliza pré-processamentos configuráveis para reuso.
    """
    if cfg.normalize_light:
        frame = normalize_lighting(frame, cfg)
    return frame
