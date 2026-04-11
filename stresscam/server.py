"""
Servidor opcional para expor o score de stress via REST e WebSocket.

Endpoints REST disponíveis:
  GET /score    → {"score": float, "trend": float, "ts": epoch_seconds}
  GET /health   → {"status": "ok", "uptime_s": float, "frame_count": int}
  GET /history  → {"history": [{"ts": float, "score": float, "trend": float}, ...]}

WebSocket:
  ws://host:8765/  → mensagens JSON periódicas com score, trend e ts

Author: Matheus Siqueira <https://www.matheussiqueira.dev/>
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Deque
from urllib.parse import urlparse

import websockets

from .logger import get_logger

_log = get_logger(__name__)

# Número máximo de entradas mantidas no histórico em memória (≈ 10 min @ 5 Hz)
_HISTORY_MAXLEN = 3000


class ScoreStore:
    """
    Armazenamento thread-safe do score atual e histórico de pontuações.

    Utiliza RLock para garantir que leituras e escritas compostas sejam atômicas,
    evitando race conditions entre o loop de inferência e as requisições HTTP/WS.
    """

    def __init__(self, history_maxlen: int = _HISTORY_MAXLEN) -> None:
        self._lock = threading.RLock()
        self.score: float = 0.5
        self.trend: float = 0.0
        self.ts: float = time.time()
        self._start_ts: float = time.time()
        self._frame_count: int = 0
        self._history: Deque[dict] = deque(maxlen=history_maxlen)

    def set(self, score: float, trend: float) -> None:
        """Atualiza o score atual e acrescenta entrada ao histórico."""
        with self._lock:
            self.score = float(score)
            self.trend = float(trend)
            self.ts = time.time()
            self._frame_count += 1
            self._history.append({"ts": self.ts, "score": self.score, "trend": self.trend})

    def snapshot(self) -> dict:
        """Retorna o estado atual como dicionário serializável."""
        with self._lock:
            return {"score": self.score, "trend": self.trend, "ts": self.ts}

    def health(self) -> dict:
        """Retorna metadados de saúde do servidor."""
        with self._lock:
            return {
                "status": "ok",
                "uptime_s": round(time.time() - self._start_ts, 1),
                "frame_count": self._frame_count,
            }

    def history(self, limit: int = 300) -> list[dict]:
        """
        Retorna as últimas `limit` entradas do histórico.

        Args:
            limit: Número máximo de entradas a retornar (padrão: 300 ≈ 1 min @ 5 Hz).
        """
        with self._lock:
            entries = list(self._history)
        return entries[-limit:]


class RESTHandler(BaseHTTPRequestHandler):
    """Handler HTTP que expõe os endpoints /score, /health e /history."""

    store: ScoreStore  # injetado pela fábrica de handlers

    # Parâmetros aceitos em /history
    _HISTORY_MAX_LIMIT = 3000

    def _send_json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Responde a preflight requests de CORS."""
        self._send_json({})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/score":
            self._send_json(self.store.snapshot())

        elif path == "/health":
            self._send_json(self.store.health())

        elif path == "/history":
            # Suporte a ?limit=N
            try:
                from urllib.parse import parse_qs
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", [300])[0])
                limit = max(1, min(limit, self._HISTORY_MAX_LIMIT))
            except (ValueError, KeyError):
                limit = 300
            self._send_json({"history": self.store.history(limit)})

        else:
            self._send_json({"error": "not found"}, code=404)

    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        """Suprime logs ruidosos do http.server; delega ao logger estruturado."""
        _log.debug("HTTP %s", format % args)


class ScoreServer:
    """
    Servidor combinado REST + WebSocket para exposição do score em tempo real.

    O servidor HTTP roda em uma thread daemon separada.
    O servidor WebSocket roda em um event loop asyncio em outra thread daemon.
    Ambos são encerrados via stop().
    """

    def __init__(self, http_port: int = 8000, ws_port: int = 8765, hz: float = 5.0) -> None:
        self.store = ScoreStore()
        self.http_port = http_port
        self.ws_port = ws_port
        self.period = 1.0 / hz if hz > 0 else 0.2
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_future: asyncio.Future | None = None
        self._http_server: HTTPServer | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Inicia os servidores HTTP e WebSocket em threads daemon."""
        self._start_http()
        self._start_ws()

    def stop(self) -> None:
        """Para os servidores graciosamente e aguarda as threads encerrarem."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()
            _log.debug("Servidor HTTP encerrado.")

        if self._loop and self._loop.is_running():
            def _stop() -> None:
                if self._stop_future and not self._stop_future.done():
                    self._stop_future.set_result(None)
                self._loop.stop()  # type: ignore[union-attr]

            self._loop.call_soon_threadsafe(_stop)

        for thread in self._threads:
            thread.join(timeout=2.0)
        _log.debug("Servidor WebSocket encerrado.")

    def update(self, score: float, trend: float) -> None:
        """Atualiza o score armazenado (chamado pelo loop de inferência)."""
        self.store.set(score, trend)

    # ── HTTP ──────────────────────────────────────────────────────────────

    def _start_http(self) -> None:
        # Cria subclasse ad-hoc para injetar a store sem variável global
        handler_cls = type("_BoundHandler", (RESTHandler,), {"store": self.store})
        self._http_server = HTTPServer(("0.0.0.0", self.http_port), handler_cls)
        thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="stresscam-http",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
        _log.debug("HTTP server escutando em :%d", self.http_port)

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _ws_handler(self, websocket) -> None:
        """Envia o snapshot periodicamente para cada cliente WebSocket conectado."""
        remote = getattr(websocket, "remote_address", "?")
        _log.debug("WS cliente conectado: %s", remote)
        try:
            while True:
                await asyncio.sleep(self.period)
                snap = self.store.snapshot()
                await websocket.send(json.dumps(snap))
        except websockets.ConnectionClosed:
            _log.debug("WS cliente desconectado: %s", remote)

    async def _ws_main(self) -> None:
        async with websockets.serve(self._ws_handler, "0.0.0.0", self.ws_port):
            self._stop_future = asyncio.get_running_loop().create_future()
            await self._stop_future

    def _start_ws(self) -> None:
        def runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._ws_main())

        thread = threading.Thread(target=runner, name="stresscam-ws", daemon=True)
        thread.start()
        self._threads.append(thread)
        _log.debug("WebSocket server escutando em :%d", self.ws_port)
