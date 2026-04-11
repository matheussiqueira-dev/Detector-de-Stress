"""
Gravação de sessão de stress em JSON e CSV.

Acumula leituras de score durante a sessão e exporta um relatório completo
ao encerrar, incluindo metadados (duração, score máximo/médio, percentual
de tempo acima dos limiares de alerta).

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .logger import get_logger

_log = get_logger(__name__)


@dataclass
class StressReading:
    """Uma leitura pontual do score de stress."""
    ts: float        # timestamp UNIX (segundos)
    score: float     # score normalizado [0, 1]
    trend: float     # delta em relação à leitura anterior
    mode: str        # "baseline" ou "análise"


@dataclass
class SessionSummary:
    """Metadados calculados sobre toda a sessão."""
    session_id: str
    started_at: str          # ISO 8601 UTC
    ended_at: str
    duration_s: float
    n_readings: int
    score_mean: float
    score_max: float
    score_min: float
    score_std: float
    time_above_medium_pct: float   # % do tempo com score >= 0.5
    time_above_high_pct: float     # % do tempo com score >= 0.75


class SessionRecorder:
    """
    Grava o histórico de stress de uma sessão e exporta ao encerrar.

    Uso::

        recorder = SessionRecorder(output_dir="sessions")
        recorder.record(ts=time.time(), score=0.42, trend=0.01, mode="análise")
        # ... loop de inferência ...
        paths = recorder.save()  # retorna [json_path, csv_path]

    Args:
        output_dir: Diretório de saída (criado automaticamente se não existir).
        threshold_medium: Limiar para contagem de tempo em stress moderado.
        threshold_high: Limiar para contagem de tempo em stress crítico.
    """

    def __init__(
        self,
        output_dir: str | Path = "sessions",
        threshold_medium: float = 0.50,
        threshold_high: float = 0.75,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.threshold_medium = threshold_medium
        self.threshold_high = threshold_high
        self._readings: list[StressReading] = []
        self._start_ts: float = time.time()
        self._session_id: str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def record(
        self,
        ts: float,
        score: float,
        trend: float,
        mode: str = "análise",
    ) -> None:
        """
        Adiciona uma leitura à sessão.

        Args:
            ts: Timestamp UNIX da leitura.
            score: Score de stress normalizado [0, 1].
            trend: Delta de score em relação à leitura anterior.
            mode: Modo do pipeline ("baseline" ou "análise").
        """
        self._readings.append(StressReading(
            ts=float(ts),
            score=float(score),
            trend=float(trend),
            mode=mode,
        ))

    def _compute_summary(self) -> SessionSummary:
        scores = [r.score for r in self._readings]
        n = len(scores)
        if n == 0:
            scores = [0.0]

        arr_scores = scores
        n_medium = sum(1 for s in arr_scores if s >= self.threshold_medium)
        n_high = sum(1 for s in arr_scores if s >= self.threshold_high)

        end_ts = time.time()
        return SessionSummary(
            session_id=self._session_id,
            started_at=datetime.fromtimestamp(self._start_ts, tz=timezone.utc).isoformat(),
            ended_at=datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            duration_s=round(end_ts - self._start_ts, 2),
            n_readings=n,
            score_mean=round(sum(arr_scores) / len(arr_scores), 4),
            score_max=round(max(arr_scores), 4),
            score_min=round(min(arr_scores), 4),
            score_std=round(float(_std(arr_scores)), 4),
            time_above_medium_pct=round(n_medium / n * 100, 2),
            time_above_high_pct=round(n_high / n * 100, 2),
        )

    def save(self) -> list[Path]:
        """
        Salva a sessão em JSON e CSV.

        Retorna:
            Lista com os caminhos dos arquivos gerados [json_path, csv_path].
        """
        if not self._readings:
            _log.info("Sessão vazia — nenhum arquivo gerado.")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = self._compute_summary()
        base_name = f"session_{self._session_id}"
        saved: list[Path] = []

        # ── JSON ──────────────────────────────────────────────────────────
        json_path = self.output_dir / f"{base_name}.json"
        payload = {
            "summary": asdict(summary),
            "readings": [asdict(r) for r in self._readings],
        }
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            saved.append(json_path)
            _log.info("JSON da sessão salvo em %s (%d leituras)", json_path, len(self._readings))
        except OSError as exc:
            _log.error("Falha ao salvar JSON: %s", exc)

        # ── CSV ───────────────────────────────────────────────────────────
        csv_path = self.output_dir / f"{base_name}.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["ts", "score", "trend", "mode"])
                writer.writeheader()
                for reading in self._readings:
                    writer.writerow(asdict(reading))
            saved.append(csv_path)
            _log.info("CSV da sessão salvo em %s", csv_path)
        except OSError as exc:
            _log.error("Falha ao salvar CSV: %s", exc)

        _log.info(
            "Resumo da sessão: duração=%.1fs, média=%.3f, máx=%.3f, "
            "tempo_acima_médio=%.1f%%, tempo_crítico=%.1f%%",
            summary.duration_s,
            summary.score_mean,
            summary.score_max,
            summary.time_above_medium_pct,
            summary.time_above_high_pct,
        )

        return saved

    @property
    def n_readings(self) -> int:
        """Número de leituras acumuladas na sessão."""
        return len(self._readings)


def _std(values: list[float]) -> float:
    """Desvio padrão simples sem dependência de NumPy para cálculo puro."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5
