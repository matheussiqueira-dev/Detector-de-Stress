"""
Configuração centralizada de logging para o StressCam.

Fornece um logger nomeado por módulo, saída simultânea para console e arquivo,
e utilitários de formatação compatíveis com sistemas de observabilidade.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


_LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_initialized = False


def setup_logging(
    log_file: str | Path = "stresscam.log",
    level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    stream: bool = True,
) -> None:
    """
    Configura o sistema de logging global com rotação de arquivo e saída no console.

    Args:
        log_file: Caminho do arquivo de log (rotacionado automaticamente ao atingir max_bytes).
        level: Nível mínimo de logging (debug/info/warning/error/critical).
        max_bytes: Tamanho máximo do arquivo antes de rotacionar (padrão: 5 MB).
        backup_count: Número de arquivos de backup mantidos.
        stream: Se True, também loga no stderr.
    """
    global _initialized
    if _initialized:
        return

    root_logger = logging.getLogger()
    resolved_level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    root_logger.setLevel(resolved_level)

    # Evita handlers duplicados em reentradas (ex: testes)
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Handler de arquivo com rotação
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_FORMATTER)
        file_handler.setLevel(resolved_level)
        root_logger.addHandler(file_handler)
    except OSError as exc:
        print(f"[stresscam] Aviso: não foi possível criar log em arquivo: {exc}", file=sys.stderr)

    # Handler de console
    if stream:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(_FORMATTER)
        console_handler.setLevel(resolved_level)
        root_logger.addHandler(console_handler)

    _initialized = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Retorna um logger nomeado para o módulo especificado.

    Recomendado: ``logger = get_logger(__name__)`` em cada módulo.

    Args:
        name: Nome do logger (geralmente ``__name__`` do módulo chamador).
        level: Sobrescreve o nível herdado quando especificado.

    Returns:
        Logger configurado e pronto para uso.
    """
    if not _initialized:
        setup_logging(
            level=os.getenv("STRESSCAM_LOG_LEVEL", "INFO"),
        )

    logger = logging.getLogger(name)
    if level:
        logger.setLevel(_LOG_LEVELS.get(level.lower(), logging.NOTSET))
    return logger


def reset_logging() -> None:
    """Remove todos os handlers e reseta o estado (útil em testes)."""
    global _initialized
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    _initialized = False
