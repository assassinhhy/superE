"""Board data layer: backoff, freshness, read-only main database, no trading URLs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from supere.binance.safety import guard_url
from supere.board.ban import BanStore, parse_penalty
from supere.board.fetcher import BoardFetcher
from supere.board.fresh import crosscheck, judge
from supere.board.reader import MainReader
from supere.board.serve import create_app
from supere.board.settings import AnalysisConfig, load_analysis
from supere.errors import TradingDisabled
from supere.intervals import INTERVAL_MS
from supere.storage.db import Database
from supere.storage.models import MetricRow

NOW = 1_700_000_000_000
ROOT = Path(__file__).resolve().parents[1]


def _analysis() -> AnalysisConfig:
    cfg, _path = load_analysis(str(ROOT / "analysis.toml"))
    return cfg


def _seed(path: Path, *, metric_ts: int, kline_close: int, funding_at: int) -> None:
    db = Database(path)
    for interval in ("5m", "15m", "1h", "4h", "1d"):
        close_time = kline_close
        open_time = close_time - INTERVAL_MS[interval] + 1
        db.conn.execute(
            """
            INSERT INTO klines (
              market, symbol, interval, open_time, close_time, o, h, l, c, volume,
              quote_volume, trades, taker_buy_base, taker_buy_quote, source, ingested_at, ingested_local
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "um", "ETHUSDT", interval, open_time, close_time,
                "1000", "1001", "999", "1000", "10", "10000", 3, "4", "4000", "test", NOW, NOW,
            ),
        )
    db.upsert_metrics(
        [
            MetricRow(
                symbol="ETHUSDT",
                ts=metric_ts,
                oi="10",
                oi_value="10000",
                global_lsr="1.2",
                top_pos_lsr="1.1",
                taker_lsr="0.9",
                source="test",
            )
        ]
    )
    db.conn.execute(
        """
        INSERT INTO funding_live (
          symbol, rate, next_funding_time, mark_price, index_price, updated_at, updated_local
        ) VALUES (?,?,?,?,?,?,?)
        """,
        ("ETHUSDT", "0.0001", NOW + 3_600_000, "1000", "1000", funding_at, funding_at),
    )
    db.conn.commit()
    db.checkpoint()
    db.close()


def _fetcher(tmp_path: Path, handler, *, clock=None) -> BoardFetcher:
    return BoardFetcher(
        _analysis(),
        tmp_path,
        transport=httpx.MockTransport(handler),
        clock=clock or (lambda: NOW),
    )


def test_analysis_defaults_use_www_binance_and_local_override(tmp_path: Path):
    cfg, _path = load_analysis(str(ROOT / "analysis.toml"))
    assert cfg.board.endpoints.binance == "https://www.binance.com"
    assert cfg.board.port == 8770
    assert cfg.plans.stop_atr15 == 0.5
    assert cfg.plans.stop_atr1h_factor == 0.35
    assert cfg.notify.system is False
    base = tmp_path / "analysis.toml"
    base.write_text('[board]\nport = 8770\n', encoding="utf-8")
    (tmp_path / "analysis.local.toml").write_text("[board]\nport = 8791\n", encoding="utf-8")
    overridden, _found = load_analysis(str(base))
    assert overridden.board.port == 8791
    assert overridden.board.endpoints.binance == "https://www.binance.com"


def test_penalty_retry_after_and_1003_and_418():
    retry = parse_penalty(
        status=429,
        headers={"Retry-After": "90"},
        body={"code": -1003, "msg": "too many"},
        now_ms=NOW,
        strikes=0,
        source_url="https://www.binance.com/fapi/v1/openInterest",
    )
    assert retry is not None
    assert retry.wait_seconds == 90
    assert retry.until_ms == NOW + 90_000

    doubled = parse_penalty(
        status=429,
        headers={},
        body={"code": -1003, "msg": "too many requests"},
        now_ms=NOW,
        strikes=0,
        source_url="https://www.binance.com/futures/data/openInterestHist",
    )
    assert doubled is not None
    assert doubled.wait_seconds == 60
    assert doubled.strikes == 1
    again = parse_penalty(
        status=429,
        headers={},
        body={"code": -1003, "msg": "too many requests"},
        now_ms=NOW,
        strikes=doubled.strikes,
        source_url="https://www.binance.com/futures/data/openInterestHist",
    )
    assert again.wait_seconds == 120

    banned = parse_penalty(
        status=418,
        headers={},
        body={"code": -1003, "msg": "IP banned until 1700000000000."},
        now_ms=NOW,
        strikes=0,
        source_url="https://www.binance.com/fapi/v1/klines",
    )
    assert banned.kind == "ban"
    assert banned.until_ms == 1_700_000_000_000 + 30_000

    failed = parse_penalty(
        status=418,
        headers={},
        body={"code": -1003, "msg": "Way too many requests"},
        now_ms=NOW,
        strikes=0,
        source_url="https://www.binance.com/fapi/v1/klines",
    )
    assert failed.kind == "ban"
    assert failed.until_ms == NOW + 600_000


def test_ban_survives_restart(tmp_path: Path):
    path = tmp_path / "run" / "binance_ban.json"
    store = BanStore(path)
    penalty = parse_penalty(
        status=418,
        headers={},
        body={"msg": "banned until 1700001000000"},
        now_ms=NOW,
        strikes=2,
        source_url="https://www.binance.com/fapi/v1/openInterest",
    )
    store.record(penalty)
    restored = BanStore(path)
    assert restored.active(NOW)
    assert restored.until_ms == penalty.until_ms
    assert restored.reason == penalty.reason
    assert not restored.active(penalty.until_ms + 1)


def test_trading_url_is_refused_before_any_request(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("交易接口不应该发出请求")

    fetcher = _fetcher(tmp_path, handler)
    with pytest.raises(TradingDisabled):
        fetcher.get_json("https://www.binance.com/fapi/v1/order")
    with pytest.raises(TradingDisabled):
        guard_url("https://www.binance.com/fapi/v1/leverage")
    assert fetcher.request_log == []
    fetcher.close()


def test_board_package_never_writes_the_main_database(tmp_path: Path):
    text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "backend" / "supere" / "board").glob("*.py"))
    for banned in (
        "upsert_klines",
        "upsert_metrics",
        "upsert_funding",
        "INSERT INTO klines",
        "INSERT INTO um_metrics",
        "INSERT INTO funding",
    ):
        assert banned not in text
    db_path = tmp_path / "superE.sqlite"
    _seed(db_path, metric_ts=NOW - 300_000, kline_close=NOW - 1_000, funding_at=NOW)
    reader = MainReader(db_path)
    with pytest.raises(sqlite3.OperationalError):
        reader.conn.execute(
            "INSERT INTO app_events (ts, level, component, msg) VALUES (1, 'info', 'board', 'no')"
        )
    reader.close()


def test_backoff_sends_nothing_until_the_ban_lifts(tmp_path: Path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) > 1:
            raise AssertionError("退避期间又发出了请求")
        return httpx.Response(429, headers={"Retry-After": "90"}, json={"code": -1003, "msg": "too many"})

    _seed(tmp_path / "superE.sqlite", metric_ts=NOW - 300_000, kline_close=NOW - 1_000, funding_at=NOW)
    fetcher = _fetcher(tmp_path, handler)
    fetcher.last_okx = NOW
    assert fetcher.poll_once() == "oi"
    assert len(calls) == 1
    assert fetcher.ban.active(NOW)
    assert fetcher.poll_once() is None
    assert len(calls) == 1
    fetcher.close()


def test_next_job_assembles_the_snapshot_once_per_poll(tmp_path: Path, monkeypatch):
    import supere.board.fetcher as fetcher_module

    calls: list[int] = []
    real = fetcher_module.assemble

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fetcher_module, "assemble", counting)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("所有序列都新鲜时不应该发出请求")

    _seed(tmp_path / "superE.sqlite", metric_ts=NOW - 300_000, kline_close=NOW - 1_000, funding_at=NOW)
    fetcher = _fetcher(tmp_path, handler)
    fetcher.last_oi = NOW
    fetcher.last_okx = NOW
    assert fetcher.poll_once() is None
    # Before this change metrics, funding and klines each assembled their own
    # snapshot, so one poll read the whole cache three times.
    assert calls == [1]
    fetcher.close()


def test_fresh_metrics_do_not_call_futures_data(tmp_path: Path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        raise AssertionError(request.url)

    _seed(tmp_path / "superE.sqlite", metric_ts=NOW - 300_000, kline_close=NOW - 1_000, funding_at=NOW)
    fetcher = _fetcher(tmp_path, handler)
    fetcher.last_oi = NOW
    fetcher.last_okx = NOW
    assert fetcher.poll_once() is None
    assert calls == []
    view = fetcher.view()
    assert view["metrics"]["status"] == "ok"
    assert view["metrics"]["source"] == "主库 um_metrics"
    assert "/futures/data" not in "".join(calls)
    fetcher.close()


def test_stale_metrics_fall_back_and_keep_the_exchange_timestamp(tmp_path: Path):
    calls = []
    old = NOW - 30 * 60_000
    older = NOW - 40 * 60_000

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert request.url.path.endswith("/futures/data/openInterestHist")
        return httpx.Response(
            200,
            json=[
                {"symbol": "ETHUSDT", "sumOpenInterest": "3", "sumOpenInterestValue": "30", "timestamp": old},
                {"symbol": "ETHUSDT", "sumOpenInterest": "2", "sumOpenInterestValue": "20", "timestamp": older},
                {"symbol": "ETHUSDT", "sumOpenInterest": "3", "sumOpenInterestValue": "30", "timestamp": old},
            ],
        )

    _seed(tmp_path / "superE.sqlite", metric_ts=NOW - 40 * 60_000, kline_close=NOW - 1_000, funding_at=NOW)
    fetcher = _fetcher(tmp_path, handler)
    fetcher.last_oi = NOW
    fetcher.last_okx = NOW
    assert fetcher.poll_once() == "metrics"
    assert len(calls) == 1
    points = fetcher.cache.points("metric", "ETHUSDT")
    assert [item["ts"] for item in points] == [older - 300_000, old - 300_000]
    view = fetcher.view()
    assert view["metrics"]["status"] == "stale"
    assert view["metrics"]["data_ts"] == old
    assert view["metrics"]["data_ts"] != NOW
    assert "已过期" in view["metrics"]["label"]
    fetcher.close()


def test_stale_oi_is_not_relabeled_ok(tmp_path: Path):
    seen = judge(12.5, NOW - 181_000, NOW - 181_000, "币安 /fapi/v1/openInterest", NOW, 180_000)
    assert seen["status"] == "stale"
    assert seen["data_ts"] == NOW - 181_000
    assert seen["value"] == 12.5
    assert seen["label"].startswith("上次数据")

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("不应该为了刷新过期标记而请求")

    fetcher = _fetcher(tmp_path, handler)
    fetcher.cache.upsert_point(
        "oi",
        "ETHUSDT",
        NOW - 181_000,
        {"open_interest": 12.5},
        NOW - 181_000,
        "币安 /fapi/v1/openInterest",
    )
    view = fetcher.view()
    assert view["oi"]["status"] == "stale"
    assert view["oi"]["data_ts"] == NOW - 181_000
    assert view["oi"]["value"] == 12.5
    fetcher.close()


def test_okx_gap_above_0_3_percent_is_an_anomaly(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("交叉核对不需要再请求")

    _seed(tmp_path / "superE.sqlite", metric_ts=NOW - 300_000, kline_close=NOW - 1_000, funding_at=NOW)
    fetcher = _fetcher(tmp_path, handler)
    fetcher.cache.upsert_point("okx", "ETHUSDT", NOW, {"price": 1003.1}, NOW, "OKX")
    view = fetcher.view()
    assert view["crosscheck"]["status"] == "anomaly"
    assert view["crosscheck"]["pause_plans"] is True
    assert "数据异常" in view["crosscheck"]["message"]
    assert view["crosscheck"]["binance"] == 1000
    assert view["crosscheck"]["okx"] == 1003.1

    exact = crosscheck(1000, 1003.0, 0.3, binance_ok=True, okx_ok=True)
    assert exact["status"] == "ok"
    missing = crosscheck(1000, None, 0.3, binance_ok=True, okx_ok=False)
    assert missing["status"] == "missing"
    assert missing["pause_plans"] is False
    assert missing["message"] == "交叉核对未获取"
    fetcher.close()


def test_limit_message_and_health(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    fetcher = _fetcher(tmp_path, handler)
    penalty = parse_penalty(
        status=429,
        headers={"Retry-After": "120"},
        body={"code": -1003},
        now_ms=NOW,
        strikes=0,
        source_url="https://www.binance.com/fapi/v1/openInterest",
    )
    fetcher.ban.record(penalty)
    view = fetcher.view()
    assert view["limit"]["active"] is True
    assert "限流中" in view["limit"]["message"]
    assert view["limit"]["message"].endswith("后恢复")
    app = create_app(tmp_path, _analysis(), schedule=False, fetcher=fetcher)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["role"] == "board"
        payload = client.get("/api/board/data").json()
        assert payload["limit"]["active"] is True
        assert client.get("/api/board/config").json()["board"]["endpoints"]["binance"] == "https://www.binance.com"
    fetcher.close()
