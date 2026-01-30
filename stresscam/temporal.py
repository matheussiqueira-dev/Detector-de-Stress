import numpy as np
from collections import deque

from .config import Config
from .features import blink_rate


class TemporalBuffer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.maxlen = cfg.window_len()
        self.ear = deque(maxlen=self.maxlen)
        self.tension = deque(maxlen=self.maxlen)
        self.pupil = deque(maxlen=self.maxlen)
        self.score_ema = None

    def append(self, ear, tension_vec, pupil):
        self.ear.append(ear)
        self.tension.append(tension_vec)
        self.pupil.append(pupil)

    def ready(self) -> bool:
        return len(self.ear) >= int(self.maxlen * self.cfg.min_fill_ratio)

    def _entropy(self, values, bins=8):
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = np.clip(hist, 1e-9, 1.0)
        return float(-(hist * np.log2(hist)).sum())

    def features(self):
        if not self.ready():
            return None
        ear_arr = np.array(self.ear)
        tension_arr = np.vstack(self.tension)
        pupil_arr = np.array(self.pupil)

        feats = {
            "blink_rate": blink_rate(ear_arr, self.cfg.fps, self.cfg.blink_ear_thresh),
            "ear_mean": ear_arr.mean(),
            "ear_std": ear_arr.std(),
            "tension_mean": tension_arr.mean(axis=0),
            "tension_std": tension_arr.std(axis=0),
            "pupil_mean": pupil_arr.mean(),
            "pupil_std": pupil_arr.std(),
            "entropy_tension": self._entropy(tension_arr),
        }
        return feats


class BaselineNormalizer:
    """
    Acumula janela inicial para servir de baseline individual.
    """
    def __init__(self):
        self.X = []
        self.y = []
        self.ready_flag = False

    def collect(self, feat_vec, target=0.5):
        self.X.append(feat_vec)
        self.y.append(target)

    def ready(self, min_samples=20):
        return len(self.X) >= min_samples

    def dump(self):
        self.ready_flag = True
        return np.vstack(self.X), np.array(self.y)
