import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

from .config import Config


def build_model(cfg: Config):
    if cfg.model_type == "rf":
        return RandomForestRegressor(
            n_estimators=60, max_depth=6, n_jobs=-1, random_state=42
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", SGDRegressor(alpha=1e-4, max_iter=300, penalty="l2"))
    ])


class StressRegressor:
    def __init__(self, cfg: Config):
        self.model = build_model(cfg)
        self.trained = False

    def fit_baseline(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, feat_vec):
        if not self.trained:
            return 0.5
        pred = self.model.predict([feat_vec])[0]
        return float(np.clip(pred, 0.0, 1.0))
