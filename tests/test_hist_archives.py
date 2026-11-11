from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from supere.hist.archives import (
    ArchiveItem,
    ArchiveMissing,
    DownloadError,
    VisionClient,
    archive_url,
    sidecar_path,
    write_sidecar,
)
from supere.hist.manifest import ArchiveRecord, Manifest


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    ranges: list[str] = []

    def do_GET(self):  # noqa: N802 - http.server API
        if "missing" in self.path:
            self.send_error(404)
            return
        range_header = self.headers.get("Range")
        type(self).ranges.append(range_header or "")
        body = type(self).payload
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            chunk = body[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):  # noqa: D401 - silence test server
        return


@pytest.fixture()
def server():
    payload = b"abcdefghijklmnopqrstuvwxyz" * 1000
    _RangeHandler.payload = payload
    _RangeHandler.ranges = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, payload
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def _item(port: int, name: str = "ETHUSDT-aggTrades-2024-01-01.zip") -> ArchiveItem:
    url = f"http://127.0.0.1:{port}/{name}"
    return ArchiveItem("aggTrades", "daily", "2024-01-01", name, url)


def test_url_shapes():
    assert (
        archive_url("aggTrades", "monthly", "2024-01")
        == "https://data.binance.vision/data/futures/um/monthly/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2024-01.zip"
    )
    assert (
        archive_url("bookDepth", "daily", "2023-01-01")
        == "https://data.binance.vision/data/futures/um/daily/bookDepth/ETHUSDT/ETHUSDT-bookDepth-2023-01-01.zip"
    )
    with pytest.raises(ValueError):
        archive_url("bookDepth", "monthly", "2023-01")


def test_download_and_resume(server, tmp_path: Path):
    httpd, payload = server
    item = _item(httpd.server_address[1])
    expected = hashlib.sha256(payload).hexdigest()
    part = tmp_path / (item.filename + ".part")
    part.write_bytes(payload[:100])
    client = VisionClient(proxy=None, retries=1, opener=None, sleep=lambda _s: None)
    result = client.download(item, tmp_path, expected_sha=expected)
    assert result.path.read_bytes() == payload
    assert result.sha256 == expected
    assert any(value.startswith("bytes=100-") for value in _RangeHandler.ranges)
    assert sidecar_path(result.path).read_text().startswith(expected)

    # Second call must skip the network entirely.
    before = list(_RangeHandler.ranges)
    again = client.download(item, tmp_path, expected_sha=expected)
    assert again.sha256 == expected
    assert list(_RangeHandler.ranges) == before


def test_checksum_mismatch_removes_part(server, tmp_path: Path):
    httpd, payload = server
    item = _item(httpd.server_address[1])
    client = VisionClient(proxy=None, retries=0, sleep=lambda _s: None)
    with pytest.raises(DownloadError):
        client.download(item, tmp_path, expected_sha="0" * 64)
    assert not (tmp_path / (item.filename + ".part")).exists()
    assert not (tmp_path / item.filename).exists()


def test_missing_archive(server, tmp_path: Path):
    httpd, _ = server
    item = _item(httpd.server_address[1], "missing.zip")
    client = VisionClient(proxy=None, retries=0, sleep=lambda _s: None)
    with pytest.raises(ArchiveMissing):
        client.download(item, tmp_path, expected_sha="0" * 64)


def test_sidecar_allows_offline_reuse(tmp_path: Path):
    payload = b"payload"
    item = ArchiveItem(
        "aggTrades", "daily", "2024-01-01", "x.zip", "http://127.0.0.1:1/x.zip"
    )
    dest = tmp_path / item.filename
    dest.write_bytes(payload)
    write_sidecar(dest, hashlib.sha256(payload).hexdigest())

    class _NoNetwork:
        def open(self, *args, **kwargs):
            raise AssertionError("network must not be used")

    client = VisionClient(proxy=None, opener=_NoNetwork(), sleep=lambda _s: None)
    result = client.download(item, tmp_path, expected_sha=None)
    assert result.path == dest
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_finished_part_is_promoted_without_network(tmp_path: Path):
    payload = b"finished-part"
    item = ArchiveItem(
        "aggTrades", "daily", "2024-01-01", "y.zip", "http://127.0.0.1:1/y.zip"
    )
    (tmp_path / (item.filename + ".part")).write_bytes(payload)

    class _NoNetwork:
        def open(self, *args, **kwargs):
            raise AssertionError("network must not be used")

    client = VisionClient(proxy=None, opener=_NoNetwork(), sleep=lambda _s: None)
    result = client.download(item, tmp_path, expected_sha=hashlib.sha256(payload).hexdigest())
    assert result.path.read_bytes() == payload
    assert not (tmp_path / (item.filename + ".part")).exists()


def test_manifest_latest_status(tmp_path: Path):
    manifest = Manifest(tmp_path / "m.jsonl")
    manifest.append(ArchiveRecord("aggTrades", "daily", "2024-01-01", "a.zip", "u", status="failed"))
    manifest.append(
        ArchiveRecord(
            "aggTrades",
            "daily",
            "2024-01-01",
            "a.zip",
            "u",
            status="converted",
            rows=10,
            date_counts={"2024-01-01": 10},
        )
    )
    assert set(manifest.converted()) == {"a.zip"}
    assert manifest.summaries()["a.zip"]["rows"] == 10

    manifest.append(ArchiveRecord("aggTrades", "daily", "2024-01-02", "b.zip", "u", status="missing"))
    assert manifest.summaries()["b.zip"]["status"] == "missing"

    # A later failed retry must not hide an earlier successful conversion.
    manifest.append(
        ArchiveRecord(
            "aggTrades",
            "daily",
            "2024-01-02",
            "b.zip",
            "u",
            status="converted",
            rows=5,
            date_counts={"2024-01-02": 5},
        )
    )
    manifest.append(ArchiveRecord("aggTrades", "daily", "2024-01-02", "b.zip", "u", status="failed"))
    assert set(manifest.converted()) == {"a.zip", "b.zip"}
