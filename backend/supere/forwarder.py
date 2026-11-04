"""Tiny TCP splice used by the outage acceptance check.

Point superE's proxy at this process. It forwards bytes to the real proxy
(default ``127.0.0.1:7890``). Pausing it closes every socket, which is the
scripted equivalent of cutting the proxy without touching the system proxy.
"""

from __future__ import annotations

import asyncio


class PauseProxy:
    def __init__(self, upstream_host: str, upstream_port: int, listen_host: str = "127.0.0.1", listen_port: int = 7891) -> None:
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.paused = False
        self._server: asyncio.Server | None = None
        self._pipes: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.listen_host, self.listen_port)
        sockets = self._server.sockets or []
        if sockets:
            self.listen_port = sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.paused:
            writer.close()
            return
        try:
            remote_reader, remote_writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)
        except Exception:
            writer.close()
            return
        task = asyncio.create_task(self._splice(reader, writer, remote_reader, remote_writer))
        self._pipes.add(task)
        task.add_done_callback(self._pipes.discard)

    async def _splice(self, reader, writer, remote_reader, remote_writer) -> None:
        async def pump(src, dst):
            try:
                while not self.paused:
                    data = await src.read(65536)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except Exception:
                return

        try:
            await asyncio.gather(pump(reader, remote_writer), pump(remote_reader, writer))
        finally:
            for stream in (writer, remote_writer):
                try:
                    stream.close()
                except Exception:
                    pass

    async def pause(self) -> None:
        self.paused = True
        pipes = list(self._pipes)
        for task in pipes:
            task.cancel()
        if pipes:
            await asyncio.gather(*pipes, return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def resume(self) -> None:
        self.paused = False
        await self.start()

    async def close(self) -> None:
        await self.pause()


async def serve_control(proxy: PauseProxy, control_port: int = 7892) -> None:
    """Local text protocol: PAUSE / RESUME / QUIT. Loopback only."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        if not peer or peer[0] not in {"127.0.0.1", "::1"}:
            writer.close()
            return
        raw = await reader.readline()
        cmd = raw.decode("utf-8", "replace").strip().upper()
        if cmd == "PAUSE":
            await proxy.pause()
            writer.write(b"PAUSED\n")
        elif cmd == "RESUME":
            await proxy.resume()
            writer.write(b"RESUMED\n")
        elif cmd == "QUIT":
            writer.write(b"BYE\n")
            await writer.drain()
            writer.close()
            raise SystemExit(0)
        else:
            writer.write(b"UNKNOWN\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", control_port)
    async with server:
        await server.serve_forever()
