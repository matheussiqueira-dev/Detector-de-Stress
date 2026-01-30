from dataclasses import dataclass


@dataclass
class Config:
    # captura
    fps: int = 30
    device_index: int = 0
    device_name: str | None = None  # ex: "video=BRIO 305" (DirectShow)
    mirror: bool = True  # espelhar câmera melhora UX, pode ser desligado
    buffer_size: int | None = 1  # 1 reduz latência; None mantém padrão do backend
    frame_width: int | None = None
    frame_height: int | None = None
    normalize_light: bool = True  # equalização adaptativa por CLAHE

    # pré-processamento
    clahe_clip: float = 2.0
    clahe_tiles: tuple[int, int] = (8, 8)
    bbox_smooth_alpha: float = 0.2
    low_light_thresh: int = 60  # média de luminância 0-255 para alertar pouca luz
    center_tolerance: float = 0.18  # fração da largura/altura para alertar rosto fora do centro
    high_sensitivity: bool = True  # modo demonstração mais responsivo (exagerado)
    demo_gain: float = 2.0  # ganho aplicado à variação do score no modo demo
    graph_bg: tuple[int, int, int] = (10, 10, 20)  # fundo do gráfico (BGR)

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
