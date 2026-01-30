from dataclasses import dataclass


@dataclass
class Config:
    # captura
    fps: int = 30
    device_index: int = 0
    frame_width: int | None = None
    frame_height: int | None = None

    # pré-processamento
    clahe_clip: float = 2.0
    clahe_tiles: tuple[int, int] = (8, 8)
    bbox_smooth_alpha: float = 0.2

    # janela temporal
    win_size_sec: int = 10
    min_fill_ratio: float = 0.5

    # sinais
    blink_ear_thresh: float = 0.22

    # suavização de saída
    ema_alpha: float = 0.2

    # baseline
    baseline_sec: int = 15

    # servidor de score (REST/WS)
    enable_server: bool = True
    http_port: int = 8000
    ws_port: int = 8765
    broadcast_hz: float = 5.0

    # diagnósticos
    log_diag: bool = True
    log_interval_sec: int = 5

    # modelo
    model_type: str = "sgd"  # "sgd" ou "rf"

    def window_len(self) -> int:
        return self.win_size_sec * self.fps
