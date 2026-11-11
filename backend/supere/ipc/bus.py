"""JSON messages over ``ws://127.0.0.1``. Chosen over ZeroMQ to avoid another runtime."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


class IpcServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.clients: set = set()
        self.server = None
        self.on_shutdown: Callable[[], None] | None = None
        self.on_command: Callable[[dict], Awaitable[None]] | None = None

    async def start(self) -> None:
        self.server = await serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            ping_interval=20,
            ping_timeout=120,
        )
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handler(self, websocket) -> None:
        self.clients.add(websocket)
        try:
            async for raw in websocket:
                message = json.loads(raw)
                if message.get("type") == "shutdown":
                    if self.on_shutdown is not None:
                        self.on_shutdown()
                elif message.get("type") == "command" and self.on_command is not None:
                    await self.on_command(message)
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        dead = []
        for websocket in list(self.clients):
            try:
                await websocket.send(encoded)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.clients.discard(websocket)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None


class LiveHub:
    """API-side cache of what the ingestor just broadcast."""

    def __init__(self) -> None:
        self.forming: list[dict] = []
        self.closed: deque[dict] = deque(maxlen=40)
        self.depth: dict | None = None
        self.ticker: dict | None = None
        self.mark: dict | None = None
        self.trades: deque[dict] = deque(maxlen=80)
        self.liquidations: deque[dict] = deque(maxlen=100)
        self.status: dict = {}
        self.connected = False
        self._ws = None

    def apply(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "status":
            self.status = message
        elif kind == "depth":
            self.depth = message
        elif kind == "miniTicker":
            self.ticker = message
        elif kind == "markPrice":
            self.mark = message
        elif kind == "aggTrade":
            self.trades.appendleft(message)
        elif kind == "forceOrder":
            self.liquidations.appendleft(message)
        elif kind == "kline":
            bar = message.get("bar")
            if not bar:
                return
            # Drop any prior forming bar for the same market/symbol/interval
            # (not only the same open_time). Missed closed events used to leave
            # zombies that made the chart look frozen.
            msi = (bar.get("market"), bar.get("symbol"), bar.get("interval"))
            self.forming = [
                item
                for item in self.forming
                if (item.get("market"), item.get("symbol"), item.get("interval")) != msi
            ]
            if message.get("closed"):
                self.closed.append(bar)
            else:
                self.forming.append(bar)
        elif kind == "forming":
            self.forming = list(message.get("bars") or [])

    async def run(self, url: str, stop) -> None:
        while not stop.is_set():
            try:
                async with connect(url, proxy=None, ping_interval=20, ping_timeout=120, compression=None) as ws:
                    self._ws = ws
                    self.connected = True
                    async for raw in ws:
                        self.apply(json.loads(raw))
            except Exception:
                self.connected = False
                self._ws = None
                if stop.is_set():
                    break
                try:
                    await __import__("asyncio").sleep(1)
                except Exception:
                    break
            else:
                self.connected = False
                self._ws = None

    async def send(self, payload: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(payload))
