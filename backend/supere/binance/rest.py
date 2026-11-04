"""Async REST client. Every URL is checked against the trading denylist."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from supere.binance.safety import guard_url
from supere.config import AppConfig, proxy_or_none
from supere.errors import RateLimited, RestHalted
from supere.ratelimit import WeightLimiter, kline_request_weight

KLINE_PAGE = 499

_WEIGHT_HEADERS = (
    "x-mbx-used-weight-1m",
    "x-mbx-used-weight-1M",
)


def weight_header(response: httpx.Response) -> int | None:
    for key in response.headers:
        if key.lower() in {name.lower() for name in _WEIGHT_HEADERS} or key.lower().startswith(
            "x-mbx-used-weight-1m"
        ):
            try:
                return int(response.headers[key])
            except ValueError:
                return None
    return None


class BinanceRest:
    def __init__(
        self,
        cfg: AppConfig,
        futures: WeightLimiter,
        spot: WeightLimiter,
        futures_data: WeightLimiter,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        on_event=None,
    ) -> None:
        self.cfg = cfg
        self.futures = futures
        self.spot = spot
        self.data_limiter = futures_data
        self.on_event = on_event
        proxy = proxy_or_none(cfg.proxy.url)
        self.client = httpx.AsyncClient(
            proxy=proxy,
            transport=transport,
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
            headers={"User-Agent": "superE/0.1.0 (research; public market data)"},
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    def _limiter(self, venue: str) -> WeightLimiter:
        if venue == "spot":
            return self.spot
        if venue == "futures_data":
            return self.data_limiter
        return self.futures

    async def _emit(self, level: str, code: str, msg: str, detail: dict | None = None) -> None:
        if self.on_event is not None:
            await self.on_event(level, "rest", code, msg, detail)

    async def request_json(
        self,
        url: str,
        *,
        params: dict | None = None,
        weight: int = 1,
        venue: str = "futures",
        allow_404: bool = False,
    ):
        guard_url(url)
        limiter = self._limiter(venue)
        if venue == "futures_data":
            await self.data_limiter.acquire(1)
        elif weight > 0:
            await limiter.acquire(weight)
        attempts = 0
        while True:
            if limiter.halted:
                raise RestHalted(f"{venue} REST 已停止")
            response = await self.client.get(url, params=params)
            header = weight_header(response)
            if header is not None and venue in {"futures", "spot"}:
                limiter.observe_header(header)
            if response.status_code == 418:
                limiter.halt()
                await self._emit(
                    "error",
                    "REST_418",
                    f"收到 HTTP 418，已停止 {venue} REST",
                    {"url": url},
                )
                raise RestHalted(url)
            if response.status_code == 429:
                delay = limiter.note_429()
                await self._emit(
                    "warning",
                    "REST_429",
                    f"收到 HTTP 429，退避 {delay:.0f} 秒",
                    {"url": url, "delay": delay},
                )
                attempts += 1
                if attempts >= 5:
                    raise RateLimited(url)
                await limiter._sleep(delay)
                continue
            if response.status_code == 404 and allow_404:
                return None
            response.raise_for_status()
            limiter.note_success()
            if response.content:
                return response.json()
            return None

    async def server_time(self) -> int:
        payload = await self.request_json(
            self.cfg.endpoints.fapi + "/fapi/v1/time", weight=1, venue="futures"
        )
        return int(payload["serverTime"])

    async def klines(
        self,
        market: str,
        symbol: str,
        interval: str,
        *,
        start: int | None = None,
        end: int | None = None,
        limit: int = KLINE_PAGE,
    ) -> list:
        params: dict = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end
        weight = kline_request_weight(limit)
        path, venue = _kline_route(self.cfg, market)
        try:
            data = await self.request_json(path, params=params, weight=weight, venue=venue)
        except (httpx.HTTPError, httpx.TransportError) as exc:
            if market != "spot":
                raise
            await self._emit(
                "warning",
                "SPOT_FALLBACK",
                f"现货 REST 失败，改用 data-api: {exc}",
                {},
            )
            data = await self.request_json(
                self.cfg.endpoints.spot_rest_fallback + "/api/v3/klines",
                params=params,
                weight=weight,
                venue="spot",
            )
        return list(data or [])

    async def funding_rate(
        self, symbol: str, start: int | None = None, end: int | None = None, limit: int = 1000
    ) -> list:
        params: dict = {"symbol": symbol.upper(), "limit": limit}
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end
        data = await self.request_json(
            self.cfg.endpoints.fapi + "/fapi/v1/fundingRate",
            params=params,
            weight=self.cfg.rate_limit.funding_weight,
            venue="futures",
        )
        return list(data or [])

    async def open_interest(self, symbol: str) -> dict:
        data = await self.request_json(
            self.cfg.endpoints.fapi + "/fapi/v1/openInterest",
            params={"symbol": symbol.upper()},
            weight=self.cfg.rate_limit.open_interest_weight,
            venue="futures",
        )
        return dict(data or {})

    async def futures_data(self, path: str, params: dict) -> list:
        data = await self.request_json(
            self.cfg.endpoints.fapi + path,
            params=params,
            weight=0,
            venue="futures_data",
            allow_404=True,
        )
        return list(data or [])

    async def agg_trades(self, symbol: str, from_id: int, limit: int = 1000) -> list:
        data = await self.request_json(
            self.cfg.endpoints.fapi + "/fapi/v1/aggTrades",
            params={"symbol": symbol.upper(), "fromId": from_id, "limit": limit},
            weight=self.cfg.rate_limit.agg_trade_weight,
            venue="futures",
        )
        return list(data or [])

    async def download(self, url: str, dest: Path) -> bool:
        """Download an archive file. Returns False on HTTP 404. No fapi weight."""
        guard_url(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        attempts = 0
        while True:
            response = await self.client.get(url)
            if response.status_code == 404:
                return False
            if response.status_code == 429:
                attempts += 1
                if attempts >= 5:
                    raise RateLimited(url)
                await self.futures._sleep(min(60.0, float(2 ** (attempts - 1))))
                continue
            response.raise_for_status()
            dest.write_bytes(response.content)
            return True

    async def download_verified(self, url: str, dest: Path) -> bool:
        ok = await self.download(url, dest)
        if not ok:
            return False
        checksum_dest = dest.with_suffix(dest.suffix + ".CHECKSUM")
        got = await self.download(url + ".CHECKSUM", checksum_dest)
        if not got:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"缺少校验文件: {url}.CHECKSUM")
        expected = checksum_dest.read_text(encoding="utf-8").strip().split()[0]
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest.lower() != expected.lower():
            dest.unlink(missing_ok=True)
            from supere.errors import ChecksumError

            raise ChecksumError(f"{dest.name} SHA-256 不一致")
        return True


def _kline_route(cfg: AppConfig, market: str) -> tuple[str, str]:
    base = cfg.endpoints.fapi
    if market == "spot":
        return cfg.endpoints.spot_rest + "/api/v3/klines", "spot"
    if market == "um":
        return base + "/fapi/v1/klines", "futures"
    if market == "um_mark":
        return base + "/fapi/v1/markPriceKlines", "futures"
    if market == "um_index":
        return base + "/fapi/v1/indexPriceKlines", "futures"
    if market == "um_premium":
        return base + "/fapi/v1/premiumIndexKlines", "futures"
    raise ValueError(f"未知 market {market}")
