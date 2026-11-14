"""One-at-a-time public market fetch. Reads the main database, writes only the board cache.

Every URL goes through ``guard_url``. HTTP 429, body code -1003, and HTTP 418
back off and do not present the previous payload as a new observation.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import httpx

from supere.binance.parse import metrics_from_oi_hist, metrics_from_ratio
from supere.binance.safety import guard_url
from supere.board.ban import BanStore, parse_penalty
from supere.board.cache import BoardCache
from supere.board.reader import BOARD_INTERVALS, METRIC_MS, MainReader
from supere.board.settings import AnalysisConfig
from supere.board.snapshot import assemble
log = logging.getLogger("supere.board")

_METRIC_PATHS = (
    ("/futures/data/openInterestHist", None),
    ("/futures/data/globalLongShortAccountRatio", "global_lsr"),
    ("/futures/data/topLongShortPositionRatio", "top_pos_lsr"),
    ("/futures/data/takerlongshortRatio", "taker_lsr"),
)


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class BoardFetcher:
    def __init__(
        self,
        analysis: AnalysisConfig,
        data_root: Path,
        *,
        proxy: str | None = None,
        transport: httpx.BaseTransport | None = None,
        clock=None,
        client: httpx.Client | None = None,
    ) -> None:
        self.analysis = analysis
        self.root = Path(data_root)
        self.reader = MainReader(self.root / "superE.sqlite")
        self.cache = BoardCache(self.root / "board" / "board_cache.sqlite")
        self.ban = BanStore(self.root / "run" / "binance_ban.json")
        self.clock = clock or (lambda: int(__import__("time").time() * 1000))
        self.request_log: list[str] = []
        self._lock = threading.Lock()
        self.last_oi = 0
        self.last_okx = 0
        self.last_metrics = 0
        self.last_funding = 0
        self._metric_step = 0
        self._kline_index = 0
        self._own_client = client is None
        if client is None:
            self.client = httpx.Client(
                proxy=proxy,
                transport=transport,
                timeout=httpx.Timeout(20.0, connect=10.0),
                follow_redirects=False,
                headers={"User-Agent": "superE/0.2.0 (research; public market data)"},
            )
        else:
            self.client = client

    def close(self) -> None:
        self.reader.close()
        self.cache.close()
        if self._own_client:
            self.client.close()

    def now(self) -> int:
        return int(self.clock())

    def view(self, now_ms: int | None = None) -> dict:
        moment = self.now() if now_ms is None else int(now_ms)
        with self._lock:
            return assemble(self.reader, self.cache, self.ban, self.analysis, moment)

    def poll_once(self) -> str | None:
        with self._lock:
            now = self.now()
            job = self._next_job(now, self.ban.active(now))
            if job is None:
                return None
            self._run(job, now)
            return job

    def _binance(self, url: str) -> bool:
        host = url.split("/")[2] if "://" in url else ""
        return "binance.com" in host

    def _url(self, path: str) -> str:
        return self.analysis.board.endpoints.binance.rstrip("/") + path

    def _next_job(self, now: int, banned: bool) -> str | None:
        gap = self.analysis.board.poll.oi_seconds * 1000
        if not banned:
            if self._series_stale("metrics", now) and (self._metric_step or now - self.last_metrics >= gap):
                return "metrics"
            if self._series_stale("funding", now) and now - self.last_funding >= gap:
                return "funding"
            if self._series_stale("klines", now):
                return "klines"
            if now - self.last_oi >= gap:
                return "oi"
        if now - self.last_okx >= self.analysis.board.poll.okx_seconds * 1000:
            return "okx"
        return None

    def _series_stale(self, kind: str, now: int) -> bool:
        snap = assemble(self.reader, self.cache, self.ban, self.analysis, now)
        if kind == "metrics":
            return snap["metrics"]["status"] != "ok"
        if kind == "funding":
            return snap["funding"]["status"] != "ok"
        return any(snap["klines"][interval]["status"] != "ok" for interval in BOARD_INTERVALS)

    def _run(self, job: str, now: int) -> None:
        if job == "metrics":
            self._fetch_metrics(now)
        elif job == "funding":
            self._fetch_funding(now)
        elif job == "klines":
            self._fetch_klines(now)
        elif job == "oi":
            self._fetch_oi(now)
        elif job == "okx":
            self._fetch_okx(now)

    def get_json(self, url: str, params: dict | None = None):
        guard_url(url)
        if params:
            guard_url(str(httpx.URL(url, params=params)))
        if self._binance(url) and self.ban.active(self.now()):
            return None
        self.request_log.append(url)
        try:
            response = self.client.get(url, params=params)
        except httpx.HTTPError as exc:
            log.warning("拉取失败 %s：%s", url, exc)
            return None
        try:
            body = response.json()
        except Exception:
            body = {"msg": response.text}
        backoff = self.analysis.board.backoff
        penalty = parse_penalty(
            status=response.status_code,
            headers=response.headers,
            body=body,
            now_ms=self.now(),
            strikes=self.ban.strikes,
            source_url=url,
            first_seconds=backoff.first_seconds,
            max_seconds=backoff.max_seconds,
            ban_extra_seconds=backoff.ban_extra_seconds,
            ban_parse_fail_seconds=backoff.ban_parse_fail_seconds,
        )
        if penalty is not None:
            self.ban.record(penalty)
            self.cache.add_event(self.now(), "rate_limit", penalty.as_dict())
            log.warning("币安限流 %s，来源 %s", penalty.reason, url)
            return None
        if response.status_code >= 400:
            log.warning("行情请求失败 %s %s", response.status_code, url)
            return None
        if self._binance(url):
            self.ban.note_success(self.now())
        return body

    def _fetch_metrics(self, now: int) -> None:
        path, field = _METRIC_PATHS[self._metric_step]
        url = self._url(path)
        body = self.get_json(url, {"symbol": self.analysis.board.symbol, "period": "5m", "limit": 30})
        self._metric_step = (self._metric_step + 1) % len(_METRIC_PATHS)
        if self._metric_step == 0 or body is None:
            self.last_metrics = now
        if not isinstance(body, list):
            return
        if field is None:
            parsed = metrics_from_oi_hist(body, "5m")
        else:
            parsed = metrics_from_ratio(body, field, "5m")
        symbol = self.analysis.board.symbol
        stored = {item["ts"]: dict(item["payload"]) for item in self.cache.points("metric", symbol, limit=500)}
        for row in sorted(parsed, key=lambda item: item.ts):
            payload = stored.get(row.ts) or {
                "ts": row.ts,
                "known_at": row.ts + METRIC_MS,
                "oi": None,
                "oi_value": None,
                "top_pos_lsr": None,
                "top_acc_lsr": None,
                "global_lsr": None,
                "taker_lsr": None,
            }
            payload["ts"] = row.ts
            payload["known_at"] = row.ts + METRIC_MS
            for key in ("oi", "oi_value", "top_pos_lsr", "top_acc_lsr", "global_lsr", "taker_lsr"):
                value = _num(getattr(row, key))
                if value is not None:
                    payload[key] = value
            stored[row.ts] = payload
            self.cache.upsert_point("metric", symbol, row.ts, payload, now, "币安 /futures/data")

    def _fetch_funding(self, now: int) -> None:
        url = self._url("/fapi/v1/premiumIndex")
        body = self.get_json(url, {"symbol": self.analysis.board.symbol})
        self.last_funding = now
        if not isinstance(body, dict) or body.get("markPrice") is None:
            return
        data_ts = int(body["time"]) if body.get("time") else now
        payload = {
            "rate": _num(body.get("lastFundingRate")),
            "next_funding_time": int(body["nextFundingTime"]) if body.get("nextFundingTime") else None,
            "mark": _num(body.get("markPrice")),
            "index": _num(body.get("indexPrice")),
        }
        self.cache.upsert_point("funding", self.analysis.board.symbol, data_ts, payload, now, "币安 /fapi/v1/premiumIndex")

    def _fetch_oi(self, now: int) -> None:
        url = self._url("/fapi/v1/openInterest")
        body = self.get_json(url, {"symbol": self.analysis.board.symbol})
        self.last_oi = now
        if not isinstance(body, dict) or body.get("openInterest") is None:
            return
        data_ts = int(body["time"]) if body.get("time") else now
        self.cache.upsert_point(
            "oi",
            self.analysis.board.symbol,
            data_ts,
            {"open_interest": _num(body.get("openInterest"))},
            now,
            "币安 /fapi/v1/openInterest",
        )

    def _fetch_okx(self, now: int) -> None:
        url = self.analysis.board.endpoints.okx_ticker
        body = self.get_json(url)
        self.last_okx = now
        if not isinstance(body, dict):
            return
        rows = body.get("data") or []
        if not rows:
            return
        price = _num(rows[0].get("last"))
        data_ts = int(rows[0]["ts"]) if rows[0].get("ts") else now
        if price is None:
            return
        self.cache.upsert_point("okx", self.analysis.board.symbol, data_ts, {"price": price}, now, "OKX")

    def _fetch_klines(self, now: int) -> None:
        snap = assemble(self.reader, self.cache, self.ban, self.analysis, now)
        interval = None
        for offset in range(len(BOARD_INTERVALS)):
            candidate = BOARD_INTERVALS[(self._kline_index + offset) % len(BOARD_INTERVALS)]
            if snap["klines"][candidate]["status"] != "ok":
                interval = candidate
                self._kline_index = (self._kline_index + offset + 1) % len(BOARD_INTERVALS)
                break
        if interval is None:
            return
        url = self._url("/fapi/v1/klines")
        body = self.get_json(
            url,
            {"symbol": self.analysis.board.symbol, "interval": interval, "limit": 99},
        )
        if not isinstance(body, list):
            return
        symbol = self.analysis.board.symbol
        seen = set()
        for item in sorted(body, key=lambda row: int(row[0])):
            open_time = int(item[0])
            if open_time in seen:
                continue
            seen.add(open_time)
            bar = {
                "open_time": open_time,
                "close_time": int(item[6]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "taker_buy": float(item[9]),
            }
            if bar["close_time"] > now:
                continue
            self.cache.upsert_point(f"kline:{interval}", symbol, open_time, bar, now, "币安 /fapi/v1/klines")
