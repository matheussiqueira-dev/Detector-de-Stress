import cv2
import numpy as np
from collections import deque
from typing import Deque, Tuple, Optional

Color = Tuple[int, int, int]


def color_for_score(score: float) -> Color:
    """Mapa simples verde->amarelo->vermelho para interpretações rápidas."""
    if score < 0.35:
        return (0, 200, 0)
    if score < 0.7:
        return (0, 200, 255)
    return (0, 0, 255)


def ajustar_grafico_fundo(history: "ScoreHistory", color: Color):
    """Permite trocar o fundo do gráfico em tempo real."""
    history.set_background(color)


def draw_text_block(img, lines, pos, color=(255, 255, 255), bg=(20, 20, 20), alpha=0.75):
    """Desenha bloco de texto multi-linha com fundo translúcido."""
    x, y = pos
    pad = 8
    lh = 22
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t in lines) + pad * 2
    h = lh * len(lines) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + pad, y + pad + lh * (i + 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_bbox(img, bbox, color=(255, 200, 0)):
    if bbox is None:
        return
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


class ScoreHistory:
    """Buffer de pontos (t, score) para gráfico em tempo real."""

    def __init__(self, seconds: float = 30.0, bg_color: Color = (15, 15, 15)):
        self.seconds = seconds
        self.data: Deque[Tuple[float, float]] = deque()
        self.bg_color = bg_color

    def add(self, t: float, score: float):
        self.data.append((t, score))
        while self.data and (t - self.data[0][0]) > self.seconds:
            self.data.popleft()

    def set_background(self, color: Color):
        self.bg_color = color

    def plot_on(self, img, rect):
        """Desenha gráfico simples dentro de rect=(x1,y1,x2,y2)."""
        x1, y1, x2, y2 = rect
        # fundo sólido de alto contraste
        cv2.rectangle(img, (x1, y1), (x2, y2), self.bg_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (140, 140, 140), 1)
        if len(self.data) < 2:
            cv2.putText(img, "Aguardando dados", (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            return
        times = np.array([p[0] for p in self.data])
        vals = np.array([p[1] for p in self.data])
        t_min, t_max = times.min(), times.max()
        s_min, s_max = 0.0, 1.0
        span_t = max(1e-3, t_max - t_min)
        span_s = max(1e-3, s_max - s_min)
        pts = []
        for t, v in zip(times, vals):
            px = int(x1 + (t - t_min) / span_t * (x2 - x1))
            py = int(y2 - (v - s_min) / span_s * (y2 - y1))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color_for_score(vals[i]), 3)
        # eixos e legendas
        cv2.putText(img, "0s", (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        cv2.putText(img, f"{int(self.seconds)}s", (x2 - 38, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        cv2.putText(img, "Stress", (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)


def draw_status(frame, score, trend, bbox, mode, runtime_s, messages, history: ScoreHistory,
                baseline_remaining: Optional[float], warnings: list[str],
                demo_mode: bool = False, event_msgs: Optional[list[str]] = None):
    h, w, _ = frame.shape
    draw_bbox(frame, bbox, color=color_for_score(score))

    status_lines = [
        f"Modo: {mode}",
        f"Stress: {score:.2f}",
        f"Tendência: {'subindo' if trend > 0.01 else 'descendo' if trend < -0.01 else 'estável'}",
        f"Tempo: {format_runtime(runtime_s)}",
    ]
    if baseline_remaining is not None and baseline_remaining > 0:
        status_lines.append(f"Baseline: {baseline_remaining:0.0f}s")
    draw_text_block(frame, status_lines, (10, 10))

    # destaque do score em fonte grande
    overlay = frame.copy()
    big_color = color_for_score(score)
    cv2.rectangle(overlay, (w - 280, 10), (w - 20, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, f"{score:0.2f}", (w - 270, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.1, big_color, 4)

    # mensagens interpretáveis
    if messages:
        draw_text_block(frame, messages, (10, 110), color=(230, 230, 230), bg=(0, 60, 90), alpha=0.65)

    # avisos
    if warnings:
        draw_text_block(frame, warnings, (10, h - 10 - 22 * len(warnings) - 16), color=(0, 200, 255), bg=(40, 20, 20), alpha=0.8)

    # eventos intermitentes
    if event_msgs:
        draw_text_block(frame, event_msgs, (w - 280, 10), color=(255, 255, 255), bg=(60, 40, 10), alpha=0.85)

    # selo de demonstração
    if demo_mode:
        draw_text_block(
            frame,
            ["⚠️ MODO DEMONSTRAÇÃO – ALTA SENSIBILIDADE ATIVADO", "Reatividade exagerada para vídeo"],
            (w - 380, h - 200),
            color=(0, 255, 255),
            bg=(60, 0, 0),
            alpha=0.85,
        )

    # gráfico rolling
    history.plot_on(frame, (w - 270, h - 150, w - 10, h - 10))
    return frame


def format_runtime(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
