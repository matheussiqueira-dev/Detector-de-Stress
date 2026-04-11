"""
Testes unitários para stresscam.temporal.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
import numpy as np
import pytest

from stresscam.config import Config
from stresscam.temporal import BaselineNormalizer, TemporalBuffer


def _make_cfg(**kwargs) -> Config:
    """Config mínima para testes."""
    defaults = dict(fps=30, win_size_sec=2, min_fill_ratio=0.5, baseline_sec=5)
    defaults.update(kwargs)
    return Config(**defaults)


def _random_tension(seed: int = 0) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    return rng.random(3).astype(np.float64)


class TestTemporalBuffer:
    def test_initial_not_ready(self):
        cfg = _make_cfg()
        buf = TemporalBuffer(cfg)
        assert not buf.ready()

    def test_ready_after_sufficient_samples(self):
        cfg = _make_cfg(fps=10, win_size_sec=2, min_fill_ratio=0.5)
        buf = TemporalBuffer(cfg)
        # maxlen = 20, min_fill = 10
        for i in range(10):
            buf.append(0.30, _random_tension(i), 0.002)
        assert buf.ready()

    def test_features_none_when_not_ready(self):
        cfg = _make_cfg()
        buf = TemporalBuffer(cfg)
        assert buf.features() is None

    def test_features_dict_keys(self):
        cfg = _make_cfg(fps=10, win_size_sec=2, min_fill_ratio=0.5)
        buf = TemporalBuffer(cfg)
        for i in range(10):
            buf.append(0.30, _random_tension(i), 0.002)
        feats = buf.features()
        assert feats is not None
        expected_keys = {
            "blink_rate", "ear_mean", "ear_std",
            "tension_mean", "tension_std",
            "pupil_mean", "pupil_std",
            "entropy_tension",
        }
        assert expected_keys == set(feats.keys())

    def test_ear_mean_correct(self):
        cfg = _make_cfg(fps=10, win_size_sec=2, min_fill_ratio=0.5)
        buf = TemporalBuffer(cfg)
        for _ in range(10):
            buf.append(0.30, _random_tension(), 0.002)
        feats = buf.features()
        assert feats is not None
        assert abs(feats["ear_mean"] - 0.30) < 1e-6

    def test_score_ema_none_initially(self):
        cfg = _make_cfg()
        buf = TemporalBuffer(cfg)
        assert buf.score_ema is None

    def test_update_window_preserves_data(self):
        cfg = _make_cfg(fps=10, win_size_sec=2)
        buf = TemporalBuffer(cfg)
        for i in range(5):
            buf.append(0.30, _random_tension(i), 0.002)
        # Reduz janela
        cfg2 = _make_cfg(fps=10, win_size_sec=1)
        buf.update_window(cfg2)
        assert len(buf.ear) <= 10  # maxlen novo

    def test_entropy_non_negative(self):
        cfg = _make_cfg(fps=10, win_size_sec=2, min_fill_ratio=0.5)
        buf = TemporalBuffer(cfg)
        for i in range(10):
            buf.append(0.30, _random_tension(i), 0.002)
        feats = buf.features()
        assert feats is not None
        assert feats["entropy_tension"] >= 0.0


class TestBaselineNormalizer:
    def test_not_ready_initially(self):
        norm = BaselineNormalizer()
        assert not norm.ready(min_samples=20)

    def test_ready_after_enough_samples(self):
        norm = BaselineNormalizer()
        fvec = np.zeros(12, dtype=np.float64)
        for _ in range(20):
            norm.collect(fvec)
        assert norm.ready(min_samples=20)

    def test_dump_shapes(self):
        norm = BaselineNormalizer()
        fvec = np.ones(12, dtype=np.float64)
        for _ in range(25):
            norm.collect(fvec)
        X, y = norm.dump()
        assert X.shape == (25, 12)
        assert y.shape == (25,)

    def test_dump_default_target(self):
        norm = BaselineNormalizer()
        fvec = np.zeros(12)
        for _ in range(5):
            norm.collect(fvec)
        _, y = norm.dump()
        assert np.allclose(y, 0.5)

    def test_custom_target(self):
        norm = BaselineNormalizer()
        fvec = np.zeros(12)
        norm.collect(fvec, target=0.3)
        _, y = norm.dump()
        assert y[0] == pytest.approx(0.3)

    def test_n_samples_property(self):
        norm = BaselineNormalizer()
        fvec = np.zeros(12)
        for _ in range(7):
            norm.collect(fvec)
        assert norm.n_samples == 7
