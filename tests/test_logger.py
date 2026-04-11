"""
Testes unitários para stresscam.logger.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
import logging
import tempfile
from pathlib import Path

import pytest

from stresscam.logger import get_logger, reset_logging, setup_logging


@pytest.fixture(autouse=True)
def clean_logging():
    """Reseta o estado global de logging antes de cada teste."""
    reset_logging()
    yield
    reset_logging()


class TestSetupLogging:
    def test_setup_creates_handlers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=log_file, stream=False)
            root = logging.getLogger()
            assert len(root.handlers) >= 1

    def test_setup_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=log_file, stream=False)
            setup_logging(log_file=log_file, stream=False)
            root = logging.getLogger()
            assert len(root.handlers) == 1  # não duplica

    def test_log_level_debug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=log_file, level="debug", stream=False)
            assert logging.getLogger().level == logging.DEBUG


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_correct(self):
        logger = get_logger("stresscam.test")
        assert logger.name == "stresscam.test"

    def test_different_modules_different_loggers(self):
        l1 = get_logger("module.a")
        l2 = get_logger("module.b")
        assert l1 is not l2

    def test_same_module_returns_same_logger(self):
        l1 = get_logger("module.x")
        l2 = get_logger("module.x")
        assert l1 is l2
