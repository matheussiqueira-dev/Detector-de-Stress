"""
Modelos de regressão para estimativa de stress fisiológico.

Suporta dois backends:
  - ``sgd``: Pipeline scikit-learn (StandardScaler + SGDRegressor), treinamento online-compatível.
  - ``rf``: RandomForestRegressor, mais estável para datasets pequenos.
  - Fallback NumPy: utilizado quando scikit-learn não está disponível (ex.: políticas de DLL).

Modelos treinados podem ser persistidos em disco via ``save()`` / ``load()``.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .logger import get_logger

_log = get_logger(__name__)

# ── Importação condicional de scikit-learn ────────────────────────────────
SKLEARN_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:
    SKLEARN_AVAILABLE = False
    _log.warning("scikit-learn indisponível — usando fallback NumPy para regressão.")

# ── Importação condicional de joblib ──────────────────────────────────────
JOBLIB_AVAILABLE = True
try:
    import joblib  # type: ignore[import-untyped]
except ImportError:
    JOBLIB_AVAILABLE = False
    _log.warning("joblib indisponível — persistência de modelo desabilitada.")


# ── Tipos internos ────────────────────────────────────────────────────────
_SKLearnModel = Union["Pipeline", "RandomForestRegressor"]


def _build_sklearn_model(cfg: Config) -> _SKLearnModel:
    """Constrói o modelo scikit-learn de acordo com o tipo configurado."""
    if cfg.model_type == "rf":
        return RandomForestRegressor(
            n_estimators=60, max_depth=6, n_jobs=-1, random_state=42
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", SGDRegressor(alpha=1e-4, max_iter=300, penalty="l2", random_state=42)),
    ])


class _FallbackRegressor:
    """
    Regressor simplificado baseado em z-score quando scikit-learn é indisponível.

    Comprime o módulo do z-score médio via tanh para produzir scores em [0, 1].
    """

    def __init__(self) -> None:
        self._mean: Optional[NDArray[np.float64]] = None
        self._std: Optional[NDArray[np.float64]] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-6

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._mean is None or self._std is None:
            return np.full(len(X), 0.5)
        z = (X - self._mean) / self._std
        return np.tanh(np.abs(z).mean(axis=1))


class StressRegressor:
    """
    Encapsulamento do modelo de regressão de stress.

    Seleciona automaticamente entre backend scikit-learn e fallback NumPy.
    Suporta persistência em disco via ``save()`` e ``load()``.

    Attributes:
        trained: True após ``fit_baseline()`` ser chamado com sucesso.
    """

    _MODEL_FILENAME = "stress_regressor.joblib"
    _META_FILENAME = "stress_regressor_meta.json"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.trained: bool = False
        self._model: Union[_SKLearnModel, _FallbackRegressor]

        if SKLEARN_AVAILABLE:
            self._model = _build_sklearn_model(cfg)
        else:
            self._model = _FallbackRegressor()

    def fit_baseline(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        """
        Treina o modelo com os dados de baseline do usuário.

        Args:
            X: Matriz de features de shape (n_samples, n_features).
            y: Vetor de targets de shape (n_samples,), tipicamente todos 0.5.
        """
        self._model.fit(X, y)
        self.trained = True
        _log.info("Modelo treinado com %d amostras.", len(X))

    def predict(self, feat_vec: NDArray[np.float64]) -> float:
        """
        Prediz o score de stress para um vetor de features.

        Args:
            feat_vec: Vetor 1-D de features retornado por ``pack_features()``.

        Returns:
            Score de stress em [0.0, 1.0]. Retorna 0.5 antes do treino.
        """
        if not self.trained:
            return 0.5
        pred = self._model.predict(np.atleast_2d(feat_vec))[0]
        return float(np.clip(pred, 0.0, 1.0))

    def save(self, directory: Union[str, Path] = "models") -> Optional[Path]:
        """
        Salva o modelo treinado em disco usando joblib.

        Args:
            directory: Diretório onde os arquivos do modelo serão salvos.

        Returns:
            Caminho do arquivo salvo, ou None se joblib for indisponível ou
            o modelo não tiver sido treinado.
        """
        if not self.trained:
            _log.warning("Tentativa de salvar modelo não treinado — ignorado.")
            return None

        if not JOBLIB_AVAILABLE:
            _log.warning("joblib indisponível — modelo não salvo.")
            return None

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        model_path = dir_path / self._MODEL_FILENAME
        joblib.dump(self._model, model_path)
        _log.info("Modelo salvo em %s", model_path)
        return model_path

    def load(self, directory: Union[str, Path] = "models") -> bool:
        """
        Carrega um modelo previamente salvo em disco.

        Args:
            directory: Diretório onde o arquivo do modelo está armazenado.

        Returns:
            True se o modelo foi carregado com sucesso, False caso contrário.
        """
        if not JOBLIB_AVAILABLE:
            _log.warning("joblib indisponível — modelo não carregado.")
            return False

        model_path = Path(directory) / self._MODEL_FILENAME
        if not model_path.exists():
            _log.info("Nenhum modelo salvo encontrado em %s.", model_path)
            return False

        try:
            self._model = joblib.load(model_path)
            self.trained = True
            _log.info("Modelo carregado de %s.", model_path)
            return True
        except Exception as exc:
            _log.error("Falha ao carregar modelo de %s: %s", model_path, exc)
            return False
