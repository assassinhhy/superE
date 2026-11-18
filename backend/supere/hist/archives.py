"""Download data.binance.vision archives with resume and SHA-256 checks.

The proxy is resolved by :func:`supere.netproxy.resolve_proxy`, so
``HTTPS_PROXY`` / ``HTTP_PROXY`` wins over the value passed by the caller and
an explicit empty string means "connect directly".  A partial file is kept as
``<name>.part`` and resumed with an HTTP Range request.  A finished file is
renamed to ``<name>`` and a ``<name>.sha256`` sidecar is written, so a restart
can verify it without touching the network.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from supere.binance.archive import parse_checksum
from supere.netproxy import resolve_proxy

VISION_DEFAULT = "https://data.binance.vision/data"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
_CHUNK = 1 << 20


class ArchiveMissing(RuntimeError):
    """The archive (or its CHECKSUM) does not exist on the bucket."""


class DownloadError(RuntimeError):
    """A download failed after all retries."""


@dataclass(frozen=True)
class ArchiveItem:
    kind: str
    period: str  # "daily" or "monthly"
    stamp: str
    filename: str
    url: str

    @property
    def checksum_url(self) -> str:
        return self.url + ".CHECKSUM"


@dataclass(frozen=True)
class DownloadResult:
    item: ArchiveItem
    path: Path
    sha256: str
    bytes: int


def archive_filename(kind: str, stamp: str, symbol: str = "ETHUSDT") -> str:
    return f"{symbol.upper()}-{kind}-{stamp}.zip"


def archive_url(
    kind: str,
    period: str,
    stamp: str,
    symbol: str = "ETHUSDT",
    base: str = VISION_DEFAULT,
    venue: str = "um",
) -> str:
    """Build the archive URL for ``kind`` (aggTrades/bookDepth/metrics)."""
    if period not in ("daily", "monthly"):
        raise ValueError(f"period 必须是 daily 或 monthly，收到 {period!r}")
    if kind not in ("aggTrades", "bookDepth", "metrics"):
        raise ValueError(f"未知存档类型 {kind!r}")
    if kind in ("bookDepth", "metrics") and period != "daily":
        raise ValueError(f"{kind} 只有日存档")
    name = archive_filename(kind, stamp, symbol.upper())
    return (
        f"{base.rstrip('/')}/futures/{venue}/{period}/{kind}/"
        f"{symbol.upper()}/{name}"
    )


def archive_prefix(kind: str, period: str, symbol: str = "ETHUSDT", venue: str = "um") -> str:
    return f"data/futures/{venue}/{period}/{kind}/{symbol.upper()}/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + ".sha256")


def read_sidecar(path: Path) -> str | None:
    try:
        text = sidecar_path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text.split()[0].lower() if text else None


def write_sidecar(path: Path, digest: str) -> None:
    sidecar_path(path).write_text(f"{digest}  {Path(path).name}\n", encoding="utf-8")


def default_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy:
        handlers.append(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


_PROXY_UNSET = object()


class VisionClient:
    def __init__(
        self,
        proxy: str | None | object = _PROXY_UNSET,
        *,
        timeout: float = 60.0,
        retries: int = 5,
        backoff: float = 2.0,
        opener=None,
        sleep=time.sleep,
    ) -> None:
        # Omitting ``proxy`` keeps the historical default while letting the
        # environment override it.  ``None`` still means "direct" because
        # callers (and the tests) use it for loopback servers.
        if proxy is _PROXY_UNSET:
            resolved = resolve_proxy()
        elif proxy is None:
            resolved = None
        else:
            resolved = resolve_proxy(proxy)  # type: ignore[arg-type]
        self.proxy = resolved
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._opener = opener or default_opener(resolved)
        self._sleep = sleep

    # -- low level ---------------------------------------------------------
    def _open(self, url: str, headers: dict[str, str] | None = None):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "superE-hist/1.0 (+research; no trading)",
                **(headers or {}),
            },
        )
        return self._opener.open(request, timeout=self.timeout)

    def fetch_text(self, url: str) -> str:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with self._open(url) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 404, 410):
                    raise ArchiveMissing(f"{url}: HTTP {exc.code}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                last = exc
            if attempt < self.retries:
                self._sleep(self.backoff ** attempt)
        raise DownloadError(f"读取失败 {url}: {last}")

    def fetch_checksum(self, item: ArchiveItem) -> str:
        return parse_checksum(self.fetch_text(item.checksum_url))

    # -- listing -----------------------------------------------------------
    def list_archives(self, prefix: str) -> dict[str, int]:
        """Return ``{filename: size}`` for every ``.zip`` under ``prefix``."""
        result: dict[str, int] = {}
        marker = ""
        while True:
            query = f"{S3_LIST}?prefix={urllib.parse.quote(prefix)}"
            if marker:
                query += f"&marker={urllib.parse.quote(marker)}"
            text = self.fetch_text(query)
            root = ET.fromstring(text)
            contents = root.findall("s:Contents", _NS)
            for node in contents:
                key = node.find("s:Key", _NS)
                size = node.find("s:Size", _NS)
                if key is None or size is None:
                    continue
                name = key.text.rsplit("/", 1)[-1]
                if name.endswith(".zip"):
                    result[name] = int(size.text)
            truncated = root.find("s:IsTruncated", _NS)
            if truncated is None or truncated.text != "true":
                break
            if not contents:
                break
            next_marker = contents[-1].find("s:Key", _NS)
            if next_marker is None or next_marker.text == marker:
                break
            marker = next_marker.text
        return result

    # -- download ----------------------------------------------------------
    def download(self, item: ArchiveItem, dest_dir: Path, expected_sha: str | None = None) -> DownloadResult:
        """Download ``item`` into ``dest_dir``.  Returns a verified result."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / item.filename
        part = dest_dir / (item.filename + ".part")

        if expected_sha is None:
            # A verified sidecar is enough to reuse a local file without any
            # network access (important when the proxy/bucket is down).
            side = read_sidecar(dest)
            if dest.exists() and side:
                return DownloadResult(item, dest, side, dest.stat().st_size)
            expected = self.fetch_checksum(item)
        else:
            expected = expected_sha
        expected = expected.lower()

        if dest.exists():
            side = read_sidecar(dest)
            if side == expected:
                return DownloadResult(item, dest, expected, dest.stat().st_size)
            digest = sha256_file(dest)
            if digest == expected:
                write_sidecar(dest, digest)
                return DownloadResult(item, dest, digest, dest.stat().st_size)
            dest.unlink(missing_ok=True)
            sidecar_path(dest).unlink(missing_ok=True)

        # A process may have died after fsync but before the rename.  Hash the
        # finished part once; promote it when it already matches.
        if part.exists() and part.stat().st_size > 0:
            digest = sha256_file(part)
            if digest == expected:
                os.replace(part, dest)
                write_sidecar(dest, digest)
                return DownloadResult(item, dest, digest, dest.stat().st_size)
            # Otherwise keep it: the next request resumes with a Range.

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._download_once(item, part, expected)
                digest = sha256_file(part)
                if digest != expected:
                    part.unlink(missing_ok=True)
                    raise ValueError(
                        f"SHA-256 不一致 {item.filename}: 期望 {expected} 实际 {digest}"
                    )
                os.replace(part, dest)
                write_sidecar(dest, digest)
                return DownloadResult(item, dest, digest, dest.stat().st_size)
            except ArchiveMissing:
                part.unlink(missing_ok=True)
                raise
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                if attempt < self.retries:
                    self._sleep(self.backoff ** attempt)
        part.unlink(missing_ok=True)
        raise DownloadError(f"下载失败 {item.url}: {last}")

    def _download_once(self, item: ArchiveItem, part: Path, expected: str) -> None:
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            response = self._open(item.url, headers)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):
                raise ArchiveMissing(f"{item.url}: HTTP {exc.code}") from exc
            if exc.code == 416 and part.exists():
                # The local part is already at or beyond EOF; restart from 0.
                part.unlink(missing_ok=True)
            raise
        with response:
            status = getattr(response, "status", 200)
            if offset and status == 200:
                # Server ignored Range; restart from zero.
                offset = 0
            elif offset and status != 206:
                raise DownloadError(f"Range 请求返回 HTTP {status}")
            if offset:
                content_range = response.headers.get("Content-Range", "")
                prefix = content_range.split("-", 1)[0].split()[-1] if content_range else ""
                if not prefix.isdigit() or int(prefix) != offset:
                    part.unlink(missing_ok=True)
                    raise DownloadError(f"Content-Range 起点不符: {content_range!r} != {offset}")
            mode = "ab" if offset else "wb"
            with part.open(mode) as handle:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())


def expected_checksum_from_text(text: str) -> str:
    """Kept as a named helper so tests do not depend on the binance module."""
    return parse_checksum(text)
