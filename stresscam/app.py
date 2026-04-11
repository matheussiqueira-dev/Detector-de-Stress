"""
Orquestrador principal do pipeline StressCam.

Responsável por iniciar todos os subsistemas (câmera, detecção facial,
extração de features, modelo, servidor, gravação) e executar o loop
de inferência em tempo real, com encerramento limpo via sinais SIGINT/SIGTERM
e tecla 'q' na janela OpenCV.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import os
import signal
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from .config import Config
from .diag import DiagLogger
from .features import EAR_LEFT, EAR_RIGHT, eye_aspect_ratio, facial_tension, pack_features, pupil_area
from .logger import get_logger, setup_logging
from .model import StressRegressor
from .server import ScoreServer
from .temporal import BaselineNormalizer, TemporalBuffer
from .video import FaceProcessor, VideoStream, preprocess_frame
from .viz import ScoreHistory, ajustar_grafico_fundo, color_for_score, draw_status

_log = get_logger(__name__)

# Flag global para sinalizar encerramento limpo entre threads
_shutdown_event = threading.Event()


def _register_signal_handlers() -> None:
    """Registra handlers para SIGINT e SIGTERM para encerramento limpo."""

    def _handler(signum: int, _frame) -> None:  # type: ignore[type-arg]
        sig_name = signal.Signals(signum).name
        _log.info("Sinal %s recebido — encerrando...", sig_name)
        _shutdown_event.set()

    # SIGTERM pode não estar disponível no Windows; tratamos o ImportError
    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (OSError, AttributeError):
        pass


# ── Helpers de texto ──────────────────────────────────────────────────────

_TEXT_FIXES: dict[str, str] = {
    "n??vel": "nível",
    "streess": "stress",
    "stresss": "stress",
    "tend??ncia": "tendência",
    "es??vel": "estável",
    "estavel": "estável",
    "tendencia": "tendência",
}


def _fix_encoding(lines: List[str]) -> List[str]:
    """Corrige artefatos de encoding em mensagens de texto."""
    result = []
    for line in lines:
        for broken, correct in _TEXT_FIXES.items():
            line = line.replace(broken, correct)
        result.append(line)
    return result


def _status_messages(
    mode: str,
    score: float,
    trend: float,
    baseline_remaining: Optional[float],
) -> List[str]:
    msgs: List[str] = []
    if mode == "baseline" and baseline_remaining and baseline_remaining > 0:
        msgs.append(f"Aguardando baseline de {baseline_remaining:0.0f}s (permaneça neutro)")
        msgs.append("Detectando face e alinhando...")
        msgs.append("Calculando taxa de piscadas...")
        msgs.append("Analisando microexpressões faciais...")
    else:
        msgs.append(f"Estimando nível de stress (score: {score:.2f})")
        msgs.append("Calculando taxa de piscadas...")
        msgs.append("Analisando microexpressões faciais...")
        if trend > 0.02:
            msgs.append("Tendência: carga fisiológica crescente")
        elif trend < -0.02:
            msgs.append("Stress diminuindo — respiração controlada detectada?")
        else:
            msgs.append("Tendência estável")
    return msgs


def _warnings(frame: np.ndarray, bbox: Optional[np.ndarray], cfg: Config) -> List[str]:
    h, w = frame.shape[:2]
    warns: List[str] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < cfg.low_light_thresh:
        warns.append("Iluminação fraca detectada")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        if abs(cx - 0.5) > cfg.center_tolerance or abs(cy - 0.5) > cfg.center_tolerance:
            warns.append("Face fora do centro — reposicione")
    return warns


def _preloop_instruction() -> None:
    canvas = 255 * np.ones((300, 640, 3), dtype=np.uint8)
    lines = [
        "Por favor, mantenha uma expressão neutra",
        "nos próximos 15 segundos para calibragem do baseline.",
        "",
        "Pressione 'q' para sair, 's' para salvar frame.",
        "Pressione 'd' para ativar/desativar modo demonstração.",
    ]
    y = 70
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
        y += 42
    cv2.imshow("StressCam", canvas)
    cv2.waitKey(1200)


# ── Modo demonstração ─────────────────────────────────────────────────────

def _activate_demo_mode(cfg: Config, history: ScoreHistory, backup: dict) -> None:
    if not backup.get("saved"):
        backup.update({
            "win_size_sec": cfg.win_size_sec,
            "ema_alpha": cfg.ema_alpha,
            "min_fill_ratio": cfg.min_fill_ratio,
            "bbox_smooth_alpha": cfg.bbox_smooth_alpha,
            "demo_gain": cfg.demo_gain,
            "saved": True,
        })
    cfg.win_size_sec = 4
    cfg.ema_alpha = 0.45
    cfg.min_fill_ratio = 0.2
    cfg.bbox_smooth_alpha = 0.08
    cfg.demo_gain = 2.2
    ajustar_grafico_fundo(history, cfg.graph_bg)
    cfg.high_sensitivity = True


def _deactivate_demo_mode(cfg: Config, backup: dict) -> None:
    if backup.get("saved"):
        cfg.win_size_sec = backup["win_size_sec"]
        cfg.ema_alpha = backup["ema_alpha"]
        cfg.min_fill_ratio = backup["min_fill_ratio"]
        cfg.bbox_smooth_alpha = backup["bbox_smooth_alpha"]
        cfg.demo_gain = backup["demo_gain"]
    cfg.high_sensitivity = False


# ── Eventos visuais intermitentes ─────────────────────────────────────────

def _build_event_messages(
    feats: Optional[dict],
    trend: float,
    now: float,
    event_queue: list[tuple[str, float]],
) -> List[str]:
    def enqueue(msg: str) -> None:
        event_queue.append((msg, now + 2.5))

    if trend < -0.03:
        enqueue("Respiração controlada detectada — queda no stress")
    elif trend > 0.03:
        enqueue("Stress aumentando — fique atento")

    if feats:
        if feats.get("blink_rate", 0) > 18:
            enqueue("Piscadas detectadas — aumento rápido no stress")
        tension_mag = float(np.linalg.norm(feats.get("tension_mean", 0)))
        if tension_mag > 0.05:
            enqueue("Contração facial — stress subindo")
        elif tension_mag < 0.02 and trend < -0.02:
            enqueue("Relaxamento facial — queda acentuada no stress")

    active: List[str] = []
    alive: list[tuple[str, float]] = []
    for msg, expiry in event_queue:
        if expiry > now:
            alive.append((msg, expiry))
            active.append(msg)
    event_queue[:] = alive
    return active[:3]


# ── Loop principal ────────────────────────────────────────────────────────

def run(cfg: Optional[Config] = None) -> None:
    """
    Inicia o pipeline completo de detecção de stress.

    Args:
        cfg: Configuração do pipeline. Se None, usa Config.from_env().
    """
    global _shutdown_event
    _shutdown_event = threading.Event()

    if cfg is None:
        cfg = Config.from_env()

    setup_logging(log_file=cfg.log_file, level=cfg.log_level)
    _register_signal_handlers()

    _log.info("StressCam iniciando — modelo=%s baseline=%ds", cfg.model_type, cfg.baseline_sec)

    history = ScoreHistory(seconds=30, bg_color=cfg.graph_bg)
    cfg_backup: dict = {}
    event_queue: list[tuple[str, float]] = []
    demo_mode = False

    if cfg.high_sensitivity:
        _activate_demo_mode(cfg, history, cfg_backup)
        demo_mode = True
    ajustar_grafico_fundo(history, cfg.graph_bg)

    stream = VideoStream(cfg)
    face = FaceProcessor(cfg)
    buf = TemporalBuffer(cfg)
    base = BaselineNormalizer()
    model = StressRegressor(cfg)
    diag = DiagLogger(cfg.log_interval_sec) if cfg.log_diag else None
    server = ScoreServer(cfg.http_port, cfg.ws_port, cfg.broadcast_hz) if cfg.enable_server else None

    # Importação opcional do recorder para não quebrar se não existir
    recorder = None
    if cfg.enable_recording:
        try:
            from .recorder import SessionRecorder  # type: ignore[import]
            recorder = SessionRecorder(output_dir=cfg.recording_dir)
        except ImportError:
            _log.warning("stresscam.recorder não encontrado — gravação de sessão desabilitada")

    if server:
        server.start()
        _log.info("Servidor REST em http://0.0.0.0:%d | WS em ws://0.0.0.0:%d", cfg.http_port, cfg.ws_port)

    _preloop_instruction()
    t0 = time.time()
    prev_score = 0.5
    saved_count = 0

    try:
        while not _shutdown_event.is_set():
            frame = stream.read()
            frame = preprocess_frame(frame, cfg)
            loop_start = time.time()
            runtime = loop_start - t0

            results = face.detect(frame)
            lms = face.landmarks(results)

            bbox: Optional[np.ndarray] = None
            score = prev_score
            trend = 0.0
            mode = "baseline" if runtime < cfg.baseline_sec else "análise"
            baseline_remaining: Optional[float] = (
                max(0.0, cfg.baseline_sec - runtime) if mode == "baseline" else None
            )
            messages: List[str] = []
            feats: Optional[dict] = None

            if lms:
                bbox = face.bbox_from_landmarks(lms, frame.shape)
                ear_L = eye_aspect_ratio(lms, EAR_LEFT)
                ear_R = eye_aspect_ratio(lms, EAR_RIGHT)
                ear = (ear_L + ear_R) / 2.0
                tension_vec = facial_tension(lms)
                pupil = pupil_area(lms)

                buf.append(ear, tension_vec, pupil)
                feats = buf.features()

                if feats:
                    fvec = pack_features(feats)
                    if mode == "baseline":
                        base.collect(fvec)
                    elif (not model.trained) and base.ready():
                        X0, y0 = base.dump()
                        model.fit_baseline(X0, y0)
                        _log.info("Baseline treinado com %d amostras", len(X0))
                        if cfg.auto_save_model:
                            model.save(cfg.model_dir)

                    score_raw = model.predict(fvec)
                    if cfg.high_sensitivity:
                        score_raw = 0.5 + (score_raw - 0.5) * cfg.demo_gain
                    score_raw = float(np.clip(score_raw, 0.0, 1.0))

                    if buf.score_ema is None:
                        buf.score_ema = score_raw
                    else:
                        buf.score_ema = buf.score_ema + cfg.ema_alpha * (score_raw - buf.score_ema)
                    score = buf.score_ema
                    trend = score - prev_score
                    prev_score = score
                    history.add(runtime, score)

                    if server:
                        server.update(score, trend)

                    if recorder:
                        recorder.record(ts=loop_start, score=score, trend=trend, mode=mode)

                    # Verificação de alertas de threshold
                    if score >= cfg.alert_threshold_high:
                        _log.warning("ALERTA CRÍTICO: stress=%.2f (threshold=%.2f)", score, cfg.alert_threshold_high)
                    elif score >= cfg.alert_threshold_medium:
                        _log.info("Alerta moderado: stress=%.2f", score)

                messages = _status_messages(mode, score, trend, baseline_remaining)
            else:
                messages = ["Detectando face e alinhando...", "Mantenha-se visível e iluminado"]

            warns = _warnings(frame, bbox, cfg)
            messages = _fix_encoding(messages)
            warns = _fix_encoding(warns)
            events = _build_event_messages(feats if lms else None, trend, loop_start, event_queue)
            events = _fix_encoding(events)

            frame = draw_status(
                frame, score, trend, bbox, mode,
                runtime_s=runtime,
                messages=messages,
                history=history,
                baseline_remaining=baseline_remaining,
                warnings=warns,
                demo_mode=demo_mode,
                event_msgs=events,
            )

            if diag:
                diag.tick(time.time() - loop_start)

            cv2.imshow("StressCam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                _log.info("Encerrado pelo usuário (tecla q).")
                break
            elif key == ord("s"):
                saved_count += 1
                path = f"frame_{saved_count:03d}.png"
                cv2.imwrite(path, frame)
                _log.info("Frame salvo em %s", path)
            elif key == ord("d"):
                demo_mode = not demo_mode
                if demo_mode:
                    _activate_demo_mode(cfg, history, cfg_backup)
                    buf.update_window(cfg)
                    _log.info("Modo demonstração ativado (alta sensibilidade).")
                else:
                    _deactivate_demo_mode(cfg, cfg_backup)
                    buf.update_window(cfg)
                    _log.info("Modo demonstração desativado.")

    except Exception as exc:
        _log.exception("Erro inesperado no loop principal: %s", exc)
        raise
    finally:
        _log.info("Encerrando subsistemas...")
        if server:
            server.stop()
        stream.release()
        cv2.destroyAllWindows()
        if recorder:
            saved_paths = recorder.save()
            for path in saved_paths:
                _log.info("Sessão gravada em %s", path)
        _log.info("StressCam encerrado.")


if __name__ == "__main__":
    run()
