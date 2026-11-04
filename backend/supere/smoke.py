"""Live network smoke test. Run this on the Windows machine, through the proxy.

The build environment gets HTTP 451 from fapi.binance.com, so this file is
not part of pytest.
"""

from __future__ import annotations

import asyncio
import json

import httpx
from websockets.asyncio.client import connect

from supere.acceptance import Check
from supere.binance.safety import guard_url
from supere.config import AppConfig, proxy_or_none


async def run_smoke(cfg: AppConfig, seconds: float = 8.0) -> list[Check]:
    checks: list[Check] = []
    proxy = proxy_or_none(cfg.proxy.url)
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(proxy=proxy, timeout=timeout, follow_redirects=True) as client:
        checks.append(await _get_json(client, cfg.endpoints.fapi + "/fapi/v1/time", "S1", "合约时间"))
        kline_url = cfg.endpoints.fapi + "/fapi/v1/klines"
        guard_url(kline_url)
        try:
            response = await client.get(
                kline_url,
                params={"symbol": cfg.symbols.um, "interval": "1m", "limit": 2},
            )
            weight = response.headers.get("x-mbx-used-weight-1m") or response.headers.get("X-MBX-USED-WEIGHT-1M")
            ok = response.status_code == 200 and isinstance(response.json(), list) and len(response.json()) >= 1
            checks.append(
                Check(
                    "S2",
                    "PASS" if ok else "FAIL",
                    f"合约 1m K 线 HTTP {response.status_code}，权重头={weight}",
                )
            )
        except Exception as exc:
            checks.append(Check("S2", "FAIL", f"合约 K 线失败: {exc}"))
        spot_url = cfg.endpoints.spot_rest_fallback + "/api/v3/klines"
        guard_url(spot_url)
        try:
            response = await client.get(
                spot_url,
                params={"symbol": cfg.symbols.spot, "interval": "1m", "limit": 2},
            )
            ok = response.status_code == 200 and len(response.json()) >= 1
            checks.append(Check("S3", "PASS" if ok else "FAIL", f"现货备用域名 HTTP {response.status_code}"))
        except Exception as exc:
            checks.append(Check("S3", "FAIL", f"现货备用域名失败: {exc}"))
        try:
            from supere.binance.safety import FORBIDDEN_PATH_MARKERS

            guard_url(cfg.endpoints.fapi + FORBIDDEN_PATH_MARKERS[0])
            checks.append(Check("S4", "FAIL", "下单路径没有被拒绝"))
        except Exception:
            checks.append(Check("S4", "PASS", "下单路径在发出前被拒绝，没有请求币安"))

    checks.append(
        await _ws_once(
            cfg.endpoints.fstream_market,
            [f"{cfg.symbols.um.lower()}@kline_1m"],
            proxy,
            seconds,
            "S5",
            "/market",
        )
    )
    checks.append(
        await _ws_once(
            cfg.endpoints.fstream_public,
            [f"{cfg.symbols.um.lower()}@depth20@100ms"],
            proxy,
            seconds,
            "S6",
            "/public depth20",
        )
    )
    return checks


async def _get_json(client: httpx.AsyncClient, url: str, code: str, label: str) -> Check:
    guard_url(url)
    try:
        response = await client.get(url)
        ok = response.status_code == 200 and "serverTime" in response.json()
        return Check(code, "PASS" if ok else "FAIL", f"{label} HTTP {response.status_code}")
    except Exception as exc:
        return Check(code, "FAIL", f"{label} 失败: {exc}")


async def _ws_once(
    url: str,
    streams: list[str],
    proxy: str | None,
    seconds: float,
    code: str,
    label: str,
) -> Check:
    try:
        async with connect(
            url,
            proxy=proxy,
            open_timeout=20,
            ping_interval=20,
            ping_timeout=20,
            compression=None,
            user_agent_header="superE/0.1.0",
        ) as ws:
            await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1}))
            deadline = asyncio.get_running_loop().time() + seconds
            saw = 0
            while asyncio.get_running_loop().time() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=seconds)
                payload = json.loads(raw)
                if payload.get("e") or (isinstance(payload.get("data"), dict) and payload["data"].get("e")):
                    saw += 1
                    break
            return Check(code, "PASS" if saw else "FAIL", f"{label} 收到业务消息 {saw} 条")
    except Exception as exc:
        return Check(code, "FAIL", f"{label} 失败: {exc}")


def main() -> int:
    from supere.config import load_config

    cfg, _path = load_config()
    checks = asyncio.run(run_smoke(cfg))
    failed = 0
    for item in checks:
        print(item.line())
        if item.status == "FAIL":
            failed += 1
    return 1 if failed else 0
