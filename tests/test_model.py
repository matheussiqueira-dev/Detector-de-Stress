"""
Testes unitários para stresscam.model.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from stresscam.config import Config
from stresscam.model import StressRegressor, _FallbackRegressor


def _make_cfg(model_type: str = "sgd") -> Config:
    return Config(model_type=model_type)


def _make_training_data(n: int = 50, n_features: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features)).astype(np.float64)
    y = np.full(n, 0.5, dtype=np.float64)
    return X, y


class TestStressRegressorSGD:
    def test_not_trained_returns_half(self):
        reg = StressRegressor(_make_cfg("sgd"))
        fvec = np.zeros(12, dtype=np.float64)
        assert reg.predict(fvec) == pytest.approx(0.5)

    def test_trained_flag_false_initially(self):
        reg = StressRegressor(_make_cfg("sgd"))
        assert not reg.trained

    def test_fit_baseline_sets_trained(self):
        reg = StressRegressor(_make_cfg("sgd"))
        X, y = _make_training_data()
        reg.fit_baseline(X, y)
        assert reg.trained

    def test_predict_after_training_in_range(self):
        reg = StressRegressor(_make_cfg("sgd"))
        X, y = _make_training_data()
        reg.fit_baseline(X, y)
        fvec = np.zeros(12, dtype=np.float64)
        pred = reg.predict(fvec)
        assert 0.0 <= pred <= 1.0

    def test_predict_clipped_to_unit_interval(self):
        reg = StressRegressor(_make_cfg("sgd"))
        X, y = _make_training_data()
        reg.fit_baseline(X, y)
        # Feature extrema
        fvec = np.full(12, 1e6, dtype=np.float64)
        pred = reg.predict(fvec)
        assert 0.0 <= pred <= 1.0


class TestStressRegressorRF:
    def test_rf_model_trains_and_predicts(self):
        reg = StressRegressor(_make_cfg("rf"))
        X, y = _make_training_data()
        reg.fit_baseline(X, y)
        pred = reg.predict(np.zeros(12, dtype=np.float64))
        assert 0.0 <= pred <= 1.0


class TestModelPersistence:
    def test_save_returns_none_if_not_trained(self):
        reg = StressRegressor(_make_cfg())
        result = reg.save(directory="tmp_models")
        assert result is None

    def test_save_and_load_roundtrip(self):
        reg = StressRegressor(_make_cfg("sgd"))
        X, y = _make_training_data()
        reg.fit_baseline(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = reg.save(directory=tmpdir)
            assert path is not None and path.exists()

            reg2 = StressRegressor(_make_cfg("sgd"))
            loaded = reg2.load(directory=tmpdir)
            assert loaded
            assert reg2.trained

            fvec = np.zeros(12, dtype=np.float64)
            p1 = reg.predict(fvec)
            p2 = reg2.predict(fvec)
            assert abs(p1 - p2) < 1e-6

    def test_load_returns_false_if_no_file(self):
        reg = StressRegressor(_make_cfg())
        with tempfile.TemporaryDirectory() as tmpdir:
            result = reg.load(directory=tmpdir)
            assert not result


class TestFallbackRegressor:
    def test_predict_before_fit_returns_half(self):
        fb = _FallbackRegressor()
        X = np.zeros((1, 8), dtype=np.float64)
        result = fb.predict(X)
        assert result[0] == pytest.approx(0.5)

    def test_fit_and_predict(self):
        fb = _FallbackRegressor()
        X = np.random.default_rng(42).random((20, 8)).astype(np.float64)
        y = np.full(20, 0.5)
        fb.fit(X, y)
        result = fb.predict(X[:1])
        assert len(result) == 1
        assert isinstance(float(result[0]), float)
