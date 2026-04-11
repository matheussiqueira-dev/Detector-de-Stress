"""
Testes de integração para stresscam.server.

Testa ScoreStore, RESTHandler e ScoreServer via requests HTTP.

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
import json
import socket
import threading
import time
from http.client import HTTPConnection

import pytest

from stresscam.server import ScoreServer, ScoreStore


def _free_port() -> int:
    """Encontra uma porta TCP livre no localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestScoreStore:
    def test_initial_score_is_half(self):
        store = ScoreStore()
        assert store.score == pytest.approx(0.5)

    def test_set_updates_values(self):
        store = ScoreStore()
        store.set(0.72, 0.03)
        assert store.score == pytest.approx(0.72)
        assert store.trend == pytest.approx(0.03)

    def test_snapshot_returns_dict(self):
        store = ScoreStore()
        snap = store.snapshot()
        assert isinstance(snap, dict)
        assert "score" in snap and "trend" in snap and "ts" in snap

    def test_health_returns_dict_with_keys(self):
        store = ScoreStore()
        h = store.health()
        assert h["status"] == "ok"
        assert "uptime_s" in h
        assert "frame_count" in h

    def test_history_empty_initially(self):
        store = ScoreStore()
        assert store.history() == []

    def test_history_grows_with_sets(self):
        store = ScoreStore()
        for i in range(5):
            store.set(float(i) / 10, 0.0)
        assert len(store.history()) == 5

    def test_history_limit_respected(self):
        store = ScoreStore()
        for i in range(50):
            store.set(0.5, 0.0)
        limited = store.history(limit=10)
        assert len(limited) == 10

    def test_thread_safety(self):
        """Múltiplas threads escrevendo e lendo sem deadlock."""
        store = ScoreStore()
        errors: list[Exception] = []

        def writer():
            for i in range(100):
                try:
                    store.set(0.5, 0.0)
                except Exception as exc:
                    errors.append(exc)

        def reader():
            for _ in range(100):
                try:
                    store.snapshot()
                    store.history()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Erros em threads: {errors}"


class TestRESTEndpoints:
    """Testes de integração HTTP contra o servidor real."""

    @pytest.fixture(autouse=True)
    def start_server(self):
        http_port = _free_port()
        ws_port = _free_port()
        self.server = ScoreServer(http_port=http_port, ws_port=ws_port, hz=5.0)
        self.server.start()
        time.sleep(0.2)  # aguarda servidor iniciar
        self.http_port = http_port
        yield
        self.server.stop()

    def _get(self, path: str) -> tuple[int, dict]:
        conn = HTTPConnection("127.0.0.1", self.http_port, timeout=3)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        return resp.status, body

    def test_score_endpoint_returns_200(self):
        status, body = self._get("/score")
        assert status == 200
        assert "score" in body and "trend" in body and "ts" in body

    def test_health_endpoint_returns_ok(self):
        status, body = self._get("/health")
        assert status == 200
        assert body["status"] == "ok"

    def test_history_endpoint_returns_list(self):
        status, body = self._get("/history")
        assert status == 200
        assert "history" in body
        assert isinstance(body["history"], list)

    def test_unknown_path_returns_404(self):
        status, body = self._get("/nonexistent")
        assert status == 404
        assert "error" in body

    def test_score_updates_after_set(self):
        self.server.update(0.88, 0.05)
        _, body = self._get("/score")
        assert body["score"] == pytest.approx(0.88, abs=1e-3)

    def test_history_limit_query_param(self):
        for i in range(20):
            self.server.update(float(i) / 20, 0.0)
        _, body = self._get("/history?limit=5")
        assert len(body["history"]) <= 5
