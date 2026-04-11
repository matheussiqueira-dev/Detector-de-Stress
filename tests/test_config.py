"""
Testes unitários para stresscam.config.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
import os

import pytest

from stresscam.config import Config


class TestConfigDefaults:
    """Verifica que os defaults estão dentro dos intervalos válidos."""

    def test_default_fps_positive(self):
        cfg = Config()
        assert cfg.fps > 0

    def test_default_ports_valid(self):
        cfg = Config()
        assert 1024 <= cfg.http_port <= 65535
        assert 1024 <= cfg.ws_port <= 65535
        assert cfg.http_port != cfg.ws_port

    def test_default_ema_alpha_range(self):
        cfg = Config()
        assert 0.0 < cfg.ema_alpha <= 1.0

    def test_default_model_type_valid(self):
        cfg = Config()
        assert cfg.model_type in ("sgd", "rf")

    def test_window_len_positive(self):
        cfg = Config()
        assert cfg.window_len() > 0

    def test_window_len_formula(self):
        cfg = Config(fps=30, win_size_sec=10)
        assert cfg.window_len() == 300


class TestConfigValidation:
    """Verifica que validate() rejeita valores fora dos limites."""

    def test_valid_defaults_pass_validation(self):
        Config().validate()  # Não deve levantar

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps"):
            Config(fps=0).validate()

    def test_invalid_http_port_raises(self):
        with pytest.raises(ValueError, match="http_port"):
            Config(http_port=99999).validate()

    def test_duplicate_ports_raises(self):
        with pytest.raises(ValueError, match="ws_port"):
            Config(http_port=8000, ws_port=8000).validate()

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            Config(model_type="xgboost").validate()

    def test_invalid_ema_alpha_raises(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            Config(ema_alpha=0.0).validate()

    def test_invalid_blink_thresh_raises(self):
        with pytest.raises(ValueError, match="blink_ear_thresh"):
            Config(blink_ear_thresh=1.5).validate()

    def test_alert_thresholds_ordering(self):
        # medium deve ser < high
        with pytest.raises(ValueError, match="alert_threshold_medium"):
            Config(alert_threshold_medium=0.9, alert_threshold_high=0.75).validate()

    def test_invalid_broadcast_hz_raises(self):
        with pytest.raises(ValueError, match="broadcast_hz"):
            Config(broadcast_hz=-1.0).validate()


class TestConfigFromEnv:
    """Verifica carregamento de variáveis de ambiente."""

    def test_env_override_http_port(self, monkeypatch):
        monkeypatch.setenv("STRESSCAM_HTTP_PORT", "9000")
        cfg = Config.from_env()
        assert cfg.http_port == 9000

    def test_env_override_model_type(self, monkeypatch):
        monkeypatch.setenv("STRESSCAM_MODEL_TYPE", "rf")
        cfg = Config.from_env()
        assert cfg.model_type == "rf"

    def test_env_invalid_port_uses_default(self, monkeypatch):
        monkeypatch.setenv("STRESSCAM_HTTP_PORT", "not_a_number")
        cfg = Config.from_env()
        assert cfg.http_port == Config.http_port  # default


class TestConfigDeviceName:
    """Verifica normalização do device_name."""

    def test_device_name_prefixed_correctly(self):
        cfg = Config(device_name="BRIO 305")
        assert cfg.device_name == "video=BRIO 305"

    def test_device_name_already_prefixed(self):
        cfg = Config(device_name="video=BRIO 305")
        assert cfg.device_name == "video=BRIO 305"

    def test_device_name_none_stays_none(self):
        cfg = Config(device_name=None)
        assert cfg.device_name is None
