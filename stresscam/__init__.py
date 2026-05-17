"""
StressCam package: captura e estimativa de estresse via sinais visuais.
"""

from .config import Config

__all__ = ["Config", "run"]


def run(*args, **kwargs):
    """Executa o pipeline principal sem importar OpenCV no import do pacote."""
    from .app import run as _run

    return _run(*args, **kwargs)
