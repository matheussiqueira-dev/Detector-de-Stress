import time
import numpy as np
import cv2

from .config import Config
from .video import VideoStream, FaceProcessor, normalize_lighting
from .features import eye_aspect_ratio, facial_tension, pupil_area, EAR_LEFT, EAR_RIGHT
from .temporal import TemporalBuffer, BaselineNormalizer
from .model import StressRegressor
from .viz import draw_hud


def pack_features(feats):
    return np.hstack([
        feats["blink_rate"],
        feats["ear_mean"], feats["ear_std"],
        feats["tension_mean"], feats["tension_std"],
        feats["pupil_mean"], feats["pupil_std"],
        feats["entropy_tension"]
    ])


def run(cfg: Config | None = None):
    cfg = cfg or Config()
    stream = VideoStream(cfg)
    face = FaceProcessor(cfg)
    buf = TemporalBuffer(cfg)
    base = BaselineNormalizer()
    model = StressRegressor(cfg)

    t0 = time.time()
    prev_score = 0.5

    try:
        while True:
            frame = stream.read()
            frame = normalize_lighting(frame, cfg)

            results = face.detect(frame)
            lms = face.landmarks(results)

            bbox = None
            if lms:
                bbox = face.bbox_from_landmarks(lms, frame.shape)
                ear_L = eye_aspect_ratio(lms, EAR_LEFT)
                ear_R = eye_aspect_ratio(lms, EAR_RIGHT)
                ear = (ear_L + ear_R) / 2.0
                tension_vec = facial_tension(lms)
                pupil = pupil_area(lms)

                buf.append(ear, tension_vec, pupil)
                feats = buf.features()

                if feats:
                    fvec = pack_features(feats)
                    # baseline nos primeiros segundos
                    if (time.time() - t0) < cfg.baseline_sec:
                        base.collect(fvec)
                    elif (not model.trained) and base.ready():
                        X0, y0 = base.dump()
                        model.fit_baseline(X0, y0)

                    score_raw = model.predict(fvec)
                    if buf.score_ema is None:
                        buf.score_ema = score_raw
                    else:
                        buf.score_ema = buf.score_ema + cfg.ema_alpha * (score_raw - buf.score_ema)
                    trend = buf.score_ema - prev_score
                    prev_score = buf.score_ema
                    frame = draw_hud(frame, buf.score_ema, trend, bbox)
            else:
                cv2.putText(frame, "Face nao detectada", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("StressCam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
