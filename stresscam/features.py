"""
Extração de features fisiológicas a partir dos landmarks faciais do MediaPipe.

Implementa as seguintes métricas:
  - Eye Aspect Ratio (EAR): indicador de abertura ocular / piscadas
  - Blink Rate: taxa de piscadas por minuto a partir de buffer temporal
  - Facial Tension: vetor de distâncias entre pontos-chave (sobrancelha, lábios, mandíbula)
  - Pupil Area: área da íris direita como proxy de dilatação pupilar
  - pack_features: serializa o dicionário de features em vetor 1D para o modelo

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ── Índices dos landmarks MediaPipe ──────────────────────────────────────
# Face Mesh (soluções API) com refinamento de íris ativado (468+ pontos)

#: Índices do olho esquerdo para EAR (horizontal + 4 pontos verticais)
EAR_LEFT: list[int] = [33, 160, 158, 133, 153, 144]

#: Índices do olho direito para EAR
EAR_RIGHT: list[int] = [362, 385, 387, 263, 373, 380]

#: Índices da íris direita para estimativa de área pupilar
IRIS_RIGHT: list[int] = [468, 469, 470, 471, 472]


def eye_aspect_ratio(landmarks: Any, idx: list[int]) -> float:
    """
    Calcula o Eye Aspect Ratio (EAR) para um conjunto de landmarks oculares.

    O EAR é a razão entre a abertura vertical média e a abertura horizontal do olho.
    Valores baixos indicam olho fechado (piscada).

    Args:
        landmarks: Objeto de landmarks do MediaPipe com atributo `.landmark`.
        idx: Lista de 6 índices: [canto_esq, sup1, sup2, canto_dir, inf1, inf2].

    Returns:
        Valor EAR em [0, 1]. Retorna 0.0 se a distância horizontal for zero.
    """
    pts: NDArray[np.float64] = np.array(
        [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in idx],
        dtype=np.float64,
    )
    vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3]) * 2.0
    return float(vertical / horizontal) if horizontal > 1e-8 else 0.0


def blink_rate(buffer_ear: NDArray[np.float64], fps: int, thresh: float) -> float:
    """
    Estima a taxa de piscadas por minuto a partir de um buffer de valores EAR.

    Conta transições de aberto (EAR >= thresh) para fechado (EAR < thresh).

    Args:
        buffer_ear: Array 1-D com valores EAR ao longo do tempo.
        fps: Taxa de captura usada para converter frames em minutos.
        thresh: Limiar de EAR abaixo do qual o olho é considerado fechado.

    Returns:
        Taxa de piscadas por minuto. Retorna 0.0 se o buffer estiver vazio.
    """
    if len(buffer_ear) == 0:
        return 0.0
    closed: NDArray[np.int8] = (buffer_ear < thresh).astype(np.int8)
    blinks: int = int(np.count_nonzero(np.diff(closed) == 1))
    minutes: float = len(buffer_ear) / (fps * 60.0)
    return blinks / minutes if minutes > 1e-9 else 0.0


def facial_tension(landmarks: Any) -> NDArray[np.float64]:
    """
    Extrai um vetor de tensão facial a partir de distâncias entre pontos-chave.

    Mede:
      - ``brow_gap``: distância entre sobrancelhas (ponto 70 e 105)
      - ``lip_stretch``: largura horizontal dos lábios (ponto 61 e 291)
      - ``jaw_drop``: abertura vertical da mandíbula (ponto 17 e 152)

    Args:
        landmarks: Objeto de landmarks do MediaPipe.

    Returns:
        Array de shape (3,) com as três distâncias normalizadas em coordenadas
        de imagem (0–1).
    """

    def _dist(i: int, j: int) -> float:
        a, b = landmarks.landmark[i], landmarks.landmark[j]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    brow_gap = _dist(70, 105)
    lip_stretch = _dist(61, 291)
    jaw_drop = _dist(17, 152)
    return np.array([brow_gap, lip_stretch, jaw_drop], dtype=np.float64)


def pupil_area(landmarks: Any) -> float:
    """
    Estima a área da íris direita como proxy de dilatação pupilar.

    Usa os 5 pontos da íris direita (índices 468–472) para calcular
    a bounding box e retornar largura × altura.

    Args:
        landmarks: Objeto de landmarks do MediaPipe (requer refinamento de íris).

    Returns:
        Área estimada da íris em unidades de coordenadas normalizadas (0–1²).
    """
    pts: NDArray[np.float64] = np.array(
        [[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in IRIS_RIGHT],
        dtype=np.float64,
    )
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    return w * h


def pack_features(feats: dict) -> NDArray[np.float64]:
    """
    Empacota o dicionário de features em vetor 1-D para entrada no modelo.

    O vetor resultante tem 8 elementos:
      [blink_rate, ear_mean, ear_std, tension_mean(3), tension_std(3),
       pupil_mean, pupil_std, entropy_tension]

    Args:
        feats: Dicionário retornado por ``TemporalBuffer.features()``.

    Returns:
        Array 1-D de dtype float64, shape (13,).
    """
    return np.hstack([
        feats["blink_rate"],
        feats["ear_mean"],
        feats["ear_std"],
        feats["tension_mean"],   # shape (3,)
        feats["tension_std"],    # shape (3,)
        feats["pupil_mean"],
        feats["pupil_std"],
        feats["entropy_tension"],
    ]).astype(np.float64)
