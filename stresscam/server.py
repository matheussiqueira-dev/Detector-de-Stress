"""
Servidor opcional para expor o score via REST e WebSocket.
REST:  GET /score -> {"score": float, "trend": float, "ts": epoch}
WS:    mensagens JSON periódicas com o mesmo payload.
"""
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets


class ScoreStore:
    def __init__(self):
        self.score = 0.5
        self.trend = 0.0
        self.ts = time.time()
        self._lock = threading.Lock()

    def set(self, score: float, trend: float):
        with self._lock:
            self.score = float(score)
            self.trend = float(trend)
            self.ts = time.time()

    def snapshot(self):
        with self._lock:
            return {"score": self.score, "trend": self.trend, "ts": self.ts}


class RESTHandler(BaseHTTPRequestHandler):
    store: ScoreStore = None

    def _set_headers(self, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        if self.path != "/score":
            self._set_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return
        snap = self.store.snapshot()
        self._set_headers()
        self.wfile.write(json.dumps(snap).encode("utf-8"))

    def log_message(self, format, *args):
        # suprime logs ruidosos do http.server
        return


class ScoreServer:
    def __init__(self, http_port=8000, ws_port=8765, hz=5.0):
        self.store = ScoreStore()
        self.http_port = http_port
        self.ws_port = ws_port
        self.period = 1.0 / hz if hz > 0 else 0.2
        self._loop = None
        self._ws_server = None
        self._threads = []

    def start(self):
        self._start_http()
        self._start_ws()

    def stop(self):
        # HTTP server stops when process ends; WS loop should be stopped
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def update(self, score, trend):
        self.store.set(score, trend)

    # --------- HTTP ---------
    def _start_http(self):
        handler = type("Handler", (RESTHandler,), {})
        handler.store = self.store
        server = HTTPServer(("0.0.0.0", self.http_port), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self._threads.append(t)

    # --------- WebSocket ---------
    async def _ws_handler(self, websocket):
        try:
            while True:
                await asyncio.sleep(self.period)
                snap = self.store.snapshot()
                await websocket.send(json.dumps(snap))
        except websockets.ConnectionClosed:
            return

    def _start_ws(self):
        def runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ws_server = websockets.serve(self._ws_handler, "0.0.0.0", self.ws_port)
            self._loop.run_until_complete(self._ws_server)
            self._loop.run_forever()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        self._threads.append(t)
