"""
Diagnósticos de performance em tempo de execução para o StressCam.

Registra FPS, tempo de inferência e picos de latência via logging estruturado,
eliminando uso de print() direto.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import time
from typing import Optional

from .logger import get_logger

_log = get_logger(__name__)


class DiagLogger:
    """
    Registra FPS e tempo médio de inferência para identificar gargalos de performance.

    Utiliza o sistema de logging centralizado (sem print direto).
    Rastreia picos de latência (frames acima de 2x a média) para alertas de jitter.
    """

    def __init__(self, interval_sec: int = 5, latency_spike_factor: float = 2.0) -> None:
        """
        Args:
            interval_sec: Intervalo de logging em segundos.
            latency_spike_factor: Multiplicador sobre a média para considerar pico de latência.
        """
        self.interval = interval_sec
        self.latency_spike_factor = latency_spike_factor
        self._last_report = time.monotonic()
        self._frame_count: int = 0
        self._infer_sum: float = 0.0
        self._infer_peak: float = 0.0
        self._spike_count: int = 0

    @property
    def avg_infer_ms(self) -> float:
        """Tempo médio de inferência em milissegundos desde o último reset."""
        if self._frame_count == 0:
            return 0.0
        return (self._infer_sum / self._frame_count) * 1000.0

    def tick(self, infer_s: float) -> None:
        """
        Registra um frame processado e dispara log a cada `interval_sec` segundos.

        Args:
            infer_s: Tempo de inferência do frame atual em **segundos**.
        """
        self._frame_count += 1
        self._infer_sum += infer_s
        infer_ms = infer_s * 1000.0

        if infer_ms > self._infer_peak:
            self._infer_peak = infer_ms

        # Detecta pico em relação à média acumulada
        avg = self.avg_infer_ms
        if avg > 0 and infer_ms > avg * self.latency_spike_factor:
            self._spike_count += 1

        now = time.monotonic()
        elapsed = now - self._last_report
        if elapsed >= self.interval:
            fps = self._frame_count / elapsed
            _log.info(
                "diag | fps=%.1f avg_infer_ms=%.2f peak_ms=%.2f spikes=%d",
                fps,
                self.avg_infer_ms,
                self._infer_peak,
                self._spike_count,
            )
            self._reset(now)

    def _reset(self, now: Optional[float] = None) -> None:
        self._last_report = now if now is not None else time.monotonic()
        self._frame_count = 0
        self._infer_sum = 0.0
        self._infer_peak = 0.0
        self._spike_count = 0
