"""
Buffers temporais deslizantes e normalização de baseline para o StressCam.

Mantém janelas de EAR, tensão facial e área pupilar para cálculo de
estatísticas agregadas usadas pelo modelo de stress.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .features import blink_rate


class TemporalBuffer:
    """
    Mantém buffers deslizantes dos sinais fisiológicos e computa features agregadas.

    Os buffers têm tamanho máximo definido por ``cfg.window_len()`` (fps × win_size_sec).
    O preenchimento mínimo para computar features é ``cfg.min_fill_ratio``.

    Attributes:
        score_ema: Média exponencial do score atual (None até o primeiro cálculo).
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.maxlen: int = cfg.window_len()
        self.ear: Deque[float] = deque(maxlen=self.maxlen)
        self.tension: Deque[NDArray[np.float64]] = deque(maxlen=self.maxlen)
        self.pupil: Deque[float] = deque(maxlen=self.maxlen)
        self.score_ema: Optional[float] = None

    def update_window(self, cfg: Config) -> None:
        """
        Atualiza o tamanho da janela sem perder os dados existentes.

        Deve ser chamado ao alterar ``cfg.win_size_sec`` em tempo real
        (ex.: ao ativar/desativar modo demonstração).
        """
        self.cfg = cfg
        new_maxlen = cfg.window_len()
        if new_maxlen == self.maxlen:
            return
        self.maxlen = new_maxlen
        self.ear = deque(self.ear, maxlen=self.maxlen)
        self.tension = deque(self.tension, maxlen=self.maxlen)
        self.pupil = deque(self.pupil, maxlen=self.maxlen)

    def append(
        self,
        ear: float,
        tension_vec: NDArray[np.float64],
        pupil: float,
    ) -> None:
        """
        Adiciona uma amostra aos buffers.

        Args:
            ear: Valor médio de EAR do frame atual.
            tension_vec: Vetor de tensão facial de shape (3,).
            pupil: Área estimada da íris direita.
        """
        self.ear.append(float(ear))
        self.tension.append(tension_vec)
        self.pupil.append(float(pupil))

    def ready(self) -> bool:
        """Retorna True quando o buffer tem dados suficientes para computar features."""
        return len(self.ear) >= int(self.maxlen * self.cfg.min_fill_ratio)

    def _entropy(self, values: NDArray[np.float64], bins: int = 8) -> float:
        """
        Calcula a entropia de Shannon de um array de valores.

        Args:
            values: Array N-D; será achatado antes do histograma.
            bins: Número de bins do histograma.

        Returns:
            Entropia em bits (log base 2). Retorna 0.0 se o array for vazio.
        """
        hist, _ = np.histogram(values, bins=bins)
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist.astype(np.float64) / total
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def features(self) -> Optional[dict]:
        """
        Computa e retorna o dicionário de features agregadas da janela atual.

        Returns:
            Dicionário com as seguintes chaves, ou None se o buffer não estiver pronto:
              - ``blink_rate`` (float): piscadas por minuto
              - ``ear_mean`` (float): média do EAR na janela
              - ``ear_std`` (float): desvio padrão do EAR
              - ``tension_mean`` (NDArray[float64], shape (3,)): média da tensão
              - ``tension_std`` (NDArray[float64], shape (3,)): std da tensão
              - ``pupil_mean`` (float): média da área pupilar
              - ``pupil_std`` (float): std da área pupilar
              - ``entropy_tension`` (float): entropia da tensão facial
        """
        if not self.ready():
            return None

        ear_arr: NDArray[np.float64] = np.array(self.ear, dtype=np.float64)
        tension_arr: NDArray[np.float64] = np.vstack(self.tension).astype(np.float64)
        pupil_arr: NDArray[np.float64] = np.array(self.pupil, dtype=np.float64)

        return {
            "blink_rate": blink_rate(ear_arr, self.cfg.fps, self.cfg.blink_ear_thresh),
            "ear_mean": float(ear_arr.mean()),
            "ear_std": float(ear_arr.std()),
            "tension_mean": tension_arr.mean(axis=0),
            "tension_std": tension_arr.std(axis=0),
            "pupil_mean": float(pupil_arr.mean()),
            "pupil_std": float(pupil_arr.std()),
            "entropy_tension": self._entropy(tension_arr),
        }


class BaselineNormalizer:
    """
    Acumula amostras durante o período inicial de calibração (baseline).

    O baseline representa o estado fisiológico neutro do usuário,
    permitindo normalizar scores individuais em vez de usar referências genéricas.
    Após o período de baseline, ``dump()`` retorna os vetores de features e targets
    para treinar o modelo com score alvo = 0.5 (estado neutro).
    """

    _DEFAULT_TARGET: float = 0.5

    def __init__(self) -> None:
        self._X: list[NDArray[np.float64]] = []
        self._y: list[float] = []
        self.ready_flag: bool = False

    def collect(self, feat_vec: NDArray[np.float64], target: float = _DEFAULT_TARGET) -> None:
        """
        Acumula uma amostra do baseline.

        Args:
            feat_vec: Vetor de features do frame atual.
            target: Score alvo para essa amostra (padrão: 0.5 = neutro).
        """
        self._X.append(feat_vec)
        self._y.append(float(target))

    def ready(self, min_samples: int = 20) -> bool:
        """
        Retorna True quando houver amostras suficientes para treinar o modelo.

        Args:
            min_samples: Número mínimo de amostras acumuladas.
        """
        return len(self._X) >= min_samples

    @property
    def n_samples(self) -> int:
        """Número de amostras coletadas até o momento."""
        return len(self._X)

    def dump(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Retorna os vetores de features e targets acumulados para treino.

        Deve ser chamado apenas após ``ready()`` retornar True.

        Returns:
            Tupla (X, y) onde X tem shape (n_samples, n_features) e
            y tem shape (n_samples,).
        """
        self.ready_flag = True
        return np.vstack(self._X), np.array(self._y, dtype=np.float64)
