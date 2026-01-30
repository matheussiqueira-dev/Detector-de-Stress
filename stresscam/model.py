import numpy as np

from .config import Config

# tentamos usar scikit-learn; se falhar (ex: política bloqueando DLL), caímos em fallback numpy
SKLEARN_AVAILABLE = True
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover - ambiente sem scipy permitido
    SKLEARN_AVAILABLE = False


def build_model(cfg: Config):
    if not SKLEARN_AVAILABLE:
        return None
    if cfg.model_type == "rf":
        return RandomForestRegressor(
            n_estimators=60, max_depth=6, n_jobs=-1, random_state=42
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", SGDRegressor(alpha=1e-4, max_iter=300, penalty="l2"))
    ])


class _FallbackRegressor:
    """
    Modelo simplificado baseado em z-score médio das features.
    Evita dependência de DLLs (SciPy) quando indisponíveis.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6

    def predict(self, X):
        z = (X - self.mean) / self.std
        # média de |z| comprimida para [0,1] via tanh
        mag = np.tanh(np.abs(z).mean(axis=1))
        return mag


class StressRegressor:
    def __init__(self, cfg: Config):
        self.use_sklearn = SKLEARN_AVAILABLE
        if self.use_sklearn:
            self.model = build_model(cfg)
        else:
            self.model = _FallbackRegressor()
        self.trained = False

    def fit_baseline(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, feat_vec):
        if not self.trained:
            return 0.5
        pred = self.model.predict(np.atleast_2d(feat_vec))[0]
        return float(np.clip(pred, 0.0, 1.0))
