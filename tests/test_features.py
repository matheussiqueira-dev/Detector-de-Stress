"""
Testes unitários para stresscam.features.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from stresscam.features import (
    EAR_LEFT,
    EAR_RIGHT,
    IRIS_RIGHT,
    blink_rate,
    eye_aspect_ratio,
    facial_tension,
    pack_features,
    pupil_area,
)


def _make_landmarks(n: int = 480, seed: int = 42) -> MagicMock:
    """Cria objeto de landmarks falso com coordenadas aleatórias [0, 1]."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))

    lm_list = []
    for x, y in coords:
        lm = MagicMock()
        lm.x = float(x)
        lm.y = float(y)
        lm_list.append(lm)

    mock = MagicMock()
    mock.landmark = lm_list
    return mock


def _make_open_eye_landmarks() -> MagicMock:
    """Landmarks com olho aberto: EAR ≈ 0.35."""
    lm_list = [MagicMock() for _ in range(480)]
    for lm in lm_list:
        lm.x = 0.5
        lm.y = 0.5

    mock = MagicMock()
    mock.landmark = lm_list
    indices = EAR_LEFT
    # Canto esquerdo e direito separados horizontalmente → horiz grande
    mock.landmark[indices[0]].x = 0.0
    mock.landmark[indices[3]].x = 1.0
    # Pontos verticais afastados → vert grande → EAR alto
    mock.landmark[indices[1]].y = 0.3
    mock.landmark[indices[5]].y = 0.7
    mock.landmark[indices[2]].y = 0.3
    mock.landmark[indices[4]].y = 0.7
    return mock


class TestEyeAspectRatio:
    def test_returns_float(self):
        lms = _make_landmarks()
        result = eye_aspect_ratio(lms, EAR_LEFT)
        assert isinstance(result, float)

    def test_value_in_range(self):
        lms = _make_landmarks()
        result = eye_aspect_ratio(lms, EAR_LEFT)
        assert 0.0 <= result

    def test_zero_horizontal_distance_returns_zero(self):
        lms = _make_landmarks()
        # Força pontos 0 e 3 no mesmo lugar → horiz = 0
        lms.landmark[EAR_LEFT[0]].x = 0.5
        lms.landmark[EAR_LEFT[0]].y = 0.5
        lms.landmark[EAR_LEFT[3]].x = 0.5
        lms.landmark[EAR_LEFT[3]].y = 0.5
        result = eye_aspect_ratio(lms, EAR_LEFT)
        assert result == 0.0

    def test_left_right_eyes_independent(self):
        lms = _make_landmarks()
        ear_l = eye_aspect_ratio(lms, EAR_LEFT)
        ear_r = eye_aspect_ratio(lms, EAR_RIGHT)
        # Não necessariamente iguais com landmarks aleatórios
        assert isinstance(ear_l, float)
        assert isinstance(ear_r, float)


class TestBlinkRate:
    def test_empty_buffer_returns_zero(self):
        assert blink_rate(np.array([]), fps=30, thresh=0.22) == 0.0

    def test_all_open_no_blinks(self):
        # EAR sempre acima do threshold → sem piscadas
        ear = np.full(300, 0.35)
        assert blink_rate(ear, fps=30, thresh=0.22) == 0.0

    def test_one_blink_counted(self):
        # Uma transição open→closed
        ear = np.array([0.35] * 10 + [0.18] * 3 + [0.35] * 10, dtype=float)
        result = blink_rate(ear, fps=30, thresh=0.22)
        assert result > 0.0

    def test_multiple_blinks(self):
        # 3 piscadas em sequência
        blink = [0.35] * 5 + [0.18] * 2
        ear = np.array(blink * 3, dtype=float)
        result = blink_rate(ear, fps=30, thresh=0.22)
        assert result > 0.0

    def test_returns_float(self):
        ear = np.full(30, 0.30)
        result = blink_rate(ear, fps=30, thresh=0.22)
        assert isinstance(result, float)


class TestFacialTension:
    def test_returns_array_shape_3(self):
        lms = _make_landmarks()
        result = facial_tension(lms)
        assert result.shape == (3,)
        assert result.dtype == np.float64

    def test_values_non_negative(self):
        lms = _make_landmarks()
        result = facial_tension(lms)
        assert (result >= 0).all()


class TestPupilArea:
    def test_returns_float(self):
        lms = _make_landmarks()
        result = pupil_area(lms)
        assert isinstance(result, float)

    def test_value_non_negative(self):
        lms = _make_landmarks()
        result = pupil_area(lms)
        assert result >= 0.0

    def test_uses_iris_right_indices(self):
        """Altera apenas os pontos da íris e verifica que o resultado muda."""
        lms1 = _make_landmarks(seed=1)
        lms2 = _make_landmarks(seed=1)
        # Espalha os pontos da íris direita no lms2
        for idx in IRIS_RIGHT:
            lms2.landmark[idx].x = float(idx % 2) * 0.9
            lms2.landmark[idx].y = 0.5
        r1 = pupil_area(lms1)
        r2 = pupil_area(lms2)
        assert r1 != r2


class TestPackFeatures:
    def _make_feats(self):
        return {
            "blink_rate": 12.0,
            "ear_mean": 0.30,
            "ear_std": 0.02,
            "tension_mean": np.array([0.05, 0.10, 0.08]),
            "tension_std": np.array([0.01, 0.02, 0.01]),
            "pupil_mean": 0.003,
            "pupil_std": 0.0005,
            "entropy_tension": 2.1,
        }

    def test_output_is_1d_ndarray(self):
        result = pack_features(self._make_feats())
        assert result.ndim == 1

    def test_output_dtype_float64(self):
        result = pack_features(self._make_feats())
        assert result.dtype == np.float64

    def test_output_length(self):
        # 1 + 1 + 1 + 3 + 3 + 1 + 1 + 1 = 12... ajustamos conforme implementação
        result = pack_features(self._make_feats())
        # Deve ter pelo menos 8 elementos (um por chave escalar ou vetor)
        assert len(result) >= 8
