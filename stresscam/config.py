"""
Configuração centralizada do StressCam com validação e suporte a variáveis de ambiente.

Carrega parâmetros de um arquivo .env (se presente) via python-dotenv,
valida tipos e intervalos antes do uso, e expõe métodos auxiliares para
reuso na pipeline.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


def _load_env(env_file: str | Path = ".env") -> None:
    """Carrega variáveis de ambiente do arquivo .env, se disponível."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(dotenv_path=str(env_file), override=False)
    except ImportError:
        pass  # python-dotenv opcional; sem .env, os.getenv usa o ambiente do processo


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_str(key: str, default: Optional[str]) -> Optional[str]:
    return os.getenv(key, default)


@dataclass
class Config:
    """
    Parâmetros de configuração do pipeline StressCam.

    Todos os valores têm defaults razoáveis e podem ser sobrescritos via
    variáveis de ambiente (prefixo ``STRESSCAM_``) ou instanciação direta.
    Chame ``Config.from_env()`` para carregar automaticamente o arquivo .env.
    """

    # ── Captura de vídeo ──────────────────────────────────────────────────
    fps: int = 30
    device_index: int = 0
    device_name: Optional[str] = None  # ex: "video=BRIO 305" (DirectShow)
    mirror: bool = True
    buffer_size: Optional[int] = 1  # 1 reduz latência; None mantém padrão do backend
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    normalize_light: bool = True  # equalização adaptativa por CLAHE

    # ── Pré-processamento ─────────────────────────────────────────────────
    clahe_clip: float = 2.0
    clahe_tiles: tuple[int, int] = (8, 8)
    bbox_smooth_alpha: float = 0.2
    low_light_thresh: int = 60       # luminância média 0-255 para alertar pouca luz
    center_tolerance: float = 0.18   # fração da largura/altura p/ alertar fora do centro
    high_sensitivity: bool = True    # modo demo mais responsivo
    demo_gain: float = 2.0           # ganho no score durante modo demo
    graph_bg: tuple[int, int, int] = (10, 10, 20)  # fundo do gráfico (BGR)

    # ── Janela temporal ───────────────────────────────────────────────────
    win_size_sec: int = 10
    min_fill_ratio: float = 0.5

    # ── Sinais fisiológicos ───────────────────────────────────────────────
    blink_ear_thresh: float = 0.22

    # ── Suavização de saída ───────────────────────────────────────────────
    ema_alpha: float = 0.2

    # ── Baseline ──────────────────────────────────────────────────────────
    baseline_sec: int = 15

    # ── Servidor REST/WebSocket ───────────────────────────────────────────
    enable_server: bool = True
    http_port: int = 8000
    ws_port: int = 8765
    broadcast_hz: float = 5.0

    # ── Persistência de modelo ────────────────────────────────────────────
    model_dir: str = "models"        # diretório para salvar/carregar modelos treinados
    auto_save_model: bool = True     # salva modelo após treino do baseline

    # ── Gravação de sessão ────────────────────────────────────────────────
    enable_recording: bool = True    # grava histórico de score em JSON/CSV ao encerrar
    recording_dir: str = "sessions"  # diretório de saída das gravações

    # ── Diagnósticos ──────────────────────────────────────────────────────
    log_diag: bool = True
    log_interval_sec: int = 5
    log_level: str = "INFO"
    log_file: str = "stresscam.log"

    # ── Modelo ────────────────────────────────────────────────────────────
    model_type: str = "sgd"          # "sgd" ou "rf"

    # ── Alertas de stress ─────────────────────────────────────────────────
    alert_threshold_high: float = 0.75   # score acima desse valor → alerta crítico
    alert_threshold_medium: float = 0.50 # score acima desse valor → alerta moderado
    alert_cooldown_sec: float = 30.0     # tempo mínimo entre alertas repetidos

    def window_len(self) -> int:
        """Número de frames na janela temporal."""
        return self.win_size_sec * self.fps

    def validate(self) -> None:
        """
        Valida os parâmetros de configuração e lança ValueError detalhado em caso de erro.

        Deve ser chamado antes de iniciar o pipeline.
        """
        errors: list[str] = []

        if self.fps <= 0:
            errors.append(f"fps deve ser > 0, recebeu {self.fps}")
        if self.device_index < 0:
            errors.append(f"device_index deve ser >= 0, recebeu {self.device_index}")
        if not (0.0 < self.clahe_clip <= 10.0):
            errors.append(f"clahe_clip deve estar em (0, 10], recebeu {self.clahe_clip}")
        if not (0.0 < self.bbox_smooth_alpha <= 1.0):
            errors.append(f"bbox_smooth_alpha deve estar em (0, 1], recebeu {self.bbox_smooth_alpha}")
        if not (0 <= self.low_light_thresh <= 255):
            errors.append(f"low_light_thresh deve estar em [0, 255], recebeu {self.low_light_thresh}")
        if not (0.0 < self.min_fill_ratio <= 1.0):
            errors.append(f"min_fill_ratio deve estar em (0, 1], recebeu {self.min_fill_ratio}")
        if not (0.0 < self.blink_ear_thresh < 1.0):
            errors.append(f"blink_ear_thresh deve estar em (0, 1), recebeu {self.blink_ear_thresh}")
        if not (0.0 < self.ema_alpha <= 1.0):
            errors.append(f"ema_alpha deve estar em (0, 1], recebeu {self.ema_alpha}")
        if self.baseline_sec <= 0:
            errors.append(f"baseline_sec deve ser > 0, recebeu {self.baseline_sec}")
        if self.http_port <= 0 or self.http_port > 65535:
            errors.append(f"http_port inválido: {self.http_port}")
        if self.ws_port <= 0 or self.ws_port > 65535:
            errors.append(f"ws_port inválido: {self.ws_port}")
        if self.http_port == self.ws_port:
            errors.append("http_port e ws_port não podem ser iguais")
        if self.broadcast_hz <= 0:
            errors.append(f"broadcast_hz deve ser > 0, recebeu {self.broadcast_hz}")
        if self.model_type not in ("sgd", "rf"):
            errors.append(f"model_type deve ser 'sgd' ou 'rf', recebeu '{self.model_type}'")
        if not (0.0 < self.alert_threshold_high <= 1.0):
            errors.append(f"alert_threshold_high deve estar em (0, 1], recebeu {self.alert_threshold_high}")
        if not (0.0 < self.alert_threshold_medium < self.alert_threshold_high):
            errors.append(
                f"alert_threshold_medium ({self.alert_threshold_medium}) deve ser < "
                f"alert_threshold_high ({self.alert_threshold_high})"
            )

        if errors:
            bullet_list = "\n  • ".join(errors)
            raise ValueError(f"Configuração inválida:\n  • {bullet_list}")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Config":
        """
        Cria uma instância de Config carregando valores do arquivo .env e do ambiente.

        Variáveis de ambiente disponíveis (prefixo STRESSCAM_):
            STRESSCAM_DEVICE_INDEX, STRESSCAM_DEVICE, STRESSCAM_FPS,
            STRESSCAM_HTTP_PORT, STRESSCAM_WS_PORT, STRESSCAM_MODEL_TYPE,
            STRESSCAM_LOG_LEVEL, STRESSCAM_BASELINE_SEC, STRESSCAM_HIGH_SENSITIVITY,
            STRESSCAM_ENABLE_RECORDING, STRESSCAM_ENABLE_SERVER

        Returns:
            Config validada e pronta para uso.
        """
        _load_env(env_file)

        cfg = cls(
            device_index=_env_int("STRESSCAM_DEVICE_INDEX", cls.device_index),
            device_name=_env_str("STRESSCAM_DEVICE", cls.device_name),
            fps=_env_int("STRESSCAM_FPS", cls.fps),
            http_port=_env_int("STRESSCAM_HTTP_PORT", cls.http_port),
            ws_port=_env_int("STRESSCAM_WS_PORT", cls.ws_port),
            model_type=_env_str("STRESSCAM_MODEL_TYPE", cls.model_type) or cls.model_type,
            log_level=_env_str("STRESSCAM_LOG_LEVEL", cls.log_level) or cls.log_level,
            baseline_sec=_env_int("STRESSCAM_BASELINE_SEC", cls.baseline_sec),
            high_sensitivity=_env_bool("STRESSCAM_HIGH_SENSITIVITY", cls.high_sensitivity),
            enable_recording=_env_bool("STRESSCAM_ENABLE_RECORDING", cls.enable_recording),
            enable_server=_env_bool("STRESSCAM_ENABLE_SERVER", cls.enable_server),
            auto_save_model=_env_bool("STRESSCAM_AUTO_SAVE_MODEL", cls.auto_save_model),
        )
        cfg.validate()
        return cfg

    def __post_init__(self) -> None:
        # Garante que device_name receba o prefixo DirectShow correto se necessário
        if self.device_name and not self.device_name.startswith("video="):
            self.device_name = f"video={self.device_name}"
