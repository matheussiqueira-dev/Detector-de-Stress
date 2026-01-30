import logging
import os
import time
from typing import List
import numpy as np
import cv2

from .config import Config
from .video import VideoStream, FaceProcessor, preprocess_frame
from .features import eye_aspect_ratio, facial_tension, pupil_area, pack_features, EAR_LEFT, EAR_RIGHT
from .temporal import TemporalBuffer, BaselineNormalizer
from .model import StressRegressor
from .viz import draw_status, ScoreHistory, color_for_score, ajustar_grafico_fundo
from .server import ScoreServer
from .diag import DiagLogger


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("stresscam.log", encoding="utf-8"),
        ],
    )


def corrigir_textos(lines: List[str]) -> List[str]:
    """Corrige possíveis artefatos de encoding e ortografia nas mensagens."""
    fixes = {
        "n??vel": "nível",
        "streess": "stress",
        "stresss": "stress",
        "tend??ncia": "tendência",
        "es??vel": "estável",
        "estavel": "estável",
        "tendencia": "tendência",
        "subindo": "subindo",
        "descendo": "descendo",
    }
    out = []
    for ln in lines:
        for k, v in fixes.items():
            ln = ln.replace(k, v)
        out.append(ln)
    return out


def _status_messages(mode: str, score: float, trend: float, baseline_remaining: float | None) -> List[str]:
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


def _warnings(frame, bbox, cfg: Config) -> list[str]:
    h, w, _ = frame.shape
    warnings: list[str] = []
    # luz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = gray.mean()
    if mean_lum < cfg.low_light_thresh:
        warnings.append("Iluminação fraca detectada")
    # centralização
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        if abs(cx - 0.5) > cfg.center_tolerance or abs(cy - 0.5) > cfg.center_tolerance:
            warnings.append("Face fora do centro — reposicione")
    return warnings


def _preloop_instruction():
    canvas = 255 * np.ones((300, 640, 3), dtype=np.uint8)
    msgs = [
        "Por favor, mantenha uma expressão neutra",
        "nos próximos 10 segundos para calibragem do baseline.",
        "",
        "Pressione 'q' para sair, 's' para salvar frame.",
    ]
    y = 80
    for m in msgs:
        cv2.putText(canvas, m, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 40
    cv2.imshow("StressCam", canvas)
    cv2.waitKey(1200)


def aumentar_sensibilidade(cfg: Config, backup: dict):
    """Torna o sistema mais reativo para demonstração (versão extrema)."""
    if not backup.get("saved"):
        backup.update({
            "win_size_sec": cfg.win_size_sec,
            "ema_alpha": cfg.ema_alpha,
            "min_fill_ratio": cfg.min_fill_ratio,
            "bbox_smooth_alpha": cfg.bbox_smooth_alpha,
            "demo_gain": cfg.demo_gain,
            "saved": True,
        })
    cfg.win_size_sec = 4  # janela curta
    cfg.ema_alpha = 0.45  # suavização mínima para reatividade
    cfg.min_fill_ratio = 0.2
    cfg.bbox_smooth_alpha = 0.08
    cfg.demo_gain = 2.2


def ativar_modo_demonstracao(cfg: Config, history: ScoreHistory, backup: dict):
    """Encapsula ajustes de sensibilidade e HUD para gravação."""
    aumentar_sensibilidade(cfg, backup)
    ajustar_grafico_fundo(history, cfg.graph_bg)
    cfg.high_sensitivity = True


def restaurar_sensibilidade(cfg: Config, backup: dict):
    """Restaura parâmetros originais se modo demo for desativado."""
    if backup.get("saved"):
        cfg.win_size_sec = backup["win_size_sec"]
        cfg.ema_alpha = backup["ema_alpha"]
        cfg.min_fill_ratio = backup["min_fill_ratio"]
        cfg.bbox_smooth_alpha = backup["bbox_smooth_alpha"]
        cfg.demo_gain = backup["demo_gain"]
    cfg.high_sensitivity = False


def exibir_feedback_visual(feats, trend, now, event_queue: list[tuple[str, float]]) -> list[str]:
    """Gera mensagens intermitentes com base em eventos recentes."""
    def push(msg):
        event_queue.append((msg, now + 2.5))

    if trend < -0.03:
        push("Respiração controlada detectada — queda no stress")
    elif trend > 0.03:
        push("Stress aumentando — fique atento")
    if feats:
        if feats.get("blink_rate", 0) > 18:
            push("Piscadas detectadas — aumento rápido no stress")
        tension_mag = float(np.linalg.norm(feats.get("tension_mean", 0)))
        if tension_mag > 0.05:
            push("Contração facial — stress subindo")
        elif tension_mag < 0.02 and trend < -0.02:
            push("Relaxamento facial — queda acentuada no stress")

    # filtrar e devolver apenas mensagens ativas
    active = []
    alive = []
    for msg, expiry in event_queue:
        if expiry > now:
            alive.append((msg, expiry))
            active.append(msg)
    event_queue[:] = alive
    return active[:3]


def run(cfg: Config | None = None):
    cfg = cfg or Config()
    _init_logging()
    # permite override via env: STRESSCAM_DEVICE (string) ou STRESSCAM_DEVICE_INDEX
    env_device = os.getenv("STRESSCAM_DEVICE")
    if env_device:
        cfg.device_name = env_device if env_device.startswith("video=") else f"video={env_device}"
    env_index = os.getenv("STRESSCAM_DEVICE_INDEX")
    if env_index:
        try:
            cfg.device_index = int(env_index)
            cfg.device_name = None
        except ValueError:
            pass

    history = ScoreHistory(seconds=30, bg_color=cfg.graph_bg)
    cfg_backup: dict = {}
    event_queue: list[tuple[str, float]] = []
    demo_mode = False
    if cfg.high_sensitivity:
        ativar_modo_demonstracao(cfg, history, cfg_backup)
        demo_mode = True
    ajustar_grafico_fundo(history, cfg.graph_bg)

    logging.info("Abrindo câmera...")
    stream = VideoStream(cfg)
    face = FaceProcessor(cfg)
    buf = TemporalBuffer(cfg)
    base = BaselineNormalizer()
    model = StressRegressor(cfg)
    diag = DiagLogger(cfg.log_interval_sec) if cfg.log_diag else None
    server = ScoreServer(cfg.http_port, cfg.ws_port, cfg.broadcast_hz) if cfg.enable_server else None

    if server:
        server.start()

    _preloop_instruction()
    t0 = time.time()
    prev_score = 0.5
    saved_count = 0

    try:
        while True:
            frame = stream.read()
            frame = preprocess_frame(frame, cfg)
            loop_start = time.time()
            runtime = loop_start - t0

            results = face.detect(frame)
            lms = face.landmarks(results)

            bbox = None
            score = prev_score
            trend = 0.0
            mode = "baseline" if runtime < cfg.baseline_sec else "análise"
            baseline_remaining = max(0.0, cfg.baseline_sec - runtime) if mode == "baseline" else None
            messages: List[str] = []
            feats = None

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
                        logging.info("Baseline treinado com %d amostras", len(X0))

                    score_raw = model.predict(fvec)
                    if cfg.high_sensitivity:
                        # ganho exagerado em torno do baseline 0.5 para visibilidade
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

                messages = _status_messages(mode, score, trend, baseline_remaining)
            else:
                messages = ["Detectando face e alinhando...", "Mantenha-se visível e iluminado"]

            warns = _warnings(frame, bbox, cfg)
            messages = corrigir_textos(messages)
            warns = corrigir_textos(warns)
            events = exibir_feedback_visual(feats if lms else None, trend, loop_start, event_queue)
            events = corrigir_textos(events)
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

            # log leve a cada segundo
            if diag:
                diag.tick(time.time() - loop_start)

            cv2.imshow("StressCam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logging.info("Encerrado pelo usuário.")
                break
            if key == ord("s"):
                saved_count += 1
                path = f"frame_{saved_count:03d}.png"
                cv2.imwrite(path, frame)
                logging.info("Frame salvo em %s", path)
            if key == ord("d"):
                demo_mode = not demo_mode
                if demo_mode:
                    ativar_modo_demonstracao(cfg, history, cfg_backup)
                    buf.update_window(cfg)
                    logging.info("Modo demonstração ativado (alta sensibilidade).")
                else:
                    restaurar_sensibilidade(cfg, cfg_backup)
                    buf.update_window(cfg)
                    logging.info("Modo demonstração desativado (sensibilidade normal).")
    finally:
        if server:
            server.stop()
        stream.release()
        cv2.destroyAllWindows()


# Extensões futuras (esboço):
# - Detecção de emoções com FER+ ou modelos leves (ex: FER2013) acoplados à pipeline.
# - Personalização por usuário: login simples e baseline persistido em disco por ID.
# - Relatório automático pós-sessão com gráficos (score, trend) e anotações de eventos.
# - Dashboard interativo em Streamlit/Gradio conectado ao servidor WS/REST existente.
# - Feedback sonoro quando score > 0.7 (beep/tts) para alertas passivos.
# - Exportação contínua do score para APIs externas (HTTP POST/MQTT/OSC).


if __name__ == "__main__":
    run()
