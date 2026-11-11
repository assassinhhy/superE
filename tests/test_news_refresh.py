"""Hourly news refresh, manual cooldown, dedup and the LLM fallback."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from supere.news import live, llm, refresh as news_refresh
from supere.news.refresh import HourlyPlan, NewsRefresher, hour_floor_ms, next_hour_ms, run_schedule
from supere.news.summary import bias_word, headline
from supere.storage.db import Database

SGT = ZoneInfo("Asia/Singapore")
NOW = 1_800_000_000_000
SECRET = "unit-test-llm-key"


def _ms(hour: int, minute: int, second: int = 0) -> int:
    return int(datetime(2026, 11, 11, hour, minute, second, tzinfo=SGT).timestamp() * 1000)


def _item(i: int, **extra) -> dict:
    row = {
        "source": "coindesk",
        "source_tier": 2,
        "lang": "en",
        "title": f"Ethereum development update number {i}",
        "body": "ETH developers shipped a change.",
        "url": f"https://www.coindesk.com/eth-{i}",
        "ts_published": NOW - i * 60_000,
    }
    row.update(extra)
    return row


class _Resp:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _llm_payload(rows: list[dict]) -> dict:
    results = []
    for row in rows:
        index = int(row["id"])
        label = ("利多", "利空", "中性")[index % 3]
        important = index < 2
        results.append({
            "id": index,
            "summary_zh": f"以太坊出现新的进展，当前判断为{label}。",
            "label": label,
            "reason_zh": "这条标题描述的事件可能影响 ETH 的市场情绪。",
            "importance": 3 if important else 1,
            "impact_zh": "这件事可能改变市场对以太坊的风险偏好。" if important else "",
            "direction": {"利多": 0.6, "利空": -0.6, "中性": 0.0}[label],
            "urgency": 0.8 if important else 0.2,
            "relevance": 0.9,
        })
    return {"content": [{"type": "text", "text": json.dumps({"results": results}, ensure_ascii=False)}]}


class FakeLLM:
    item_calls = 0
    digest_calls = 0

    def __init__(self, *args, **kwargs):
        self.proxy = kwargs.get("proxy")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, **kwargs):
        assert SECRET not in (url or "")
        payload = kwargs.get("json")
        user = (payload or {})["messages"][0]["content"]
        if "rss_summary" in user or user.startswith("条目"):
            FakeLLM.item_calls += 1
            start = user.find("[")
            rows, _end = json.JSONDecoder().raw_decode(user[start:])
            return _Resp(200, _llm_payload(rows))
        FakeLLM.digest_calls += 1
        return _Resp(200, {"content": [{"type": "text", "text": json.dumps({
            "prose_1h": "最近 1 小时以太坊消息偏空，风险偏好下降。",
            "prose_24h": "最近 24 小时多空都有，方向不一致。",
        }, ensure_ascii=False)}]})


def test_hourly_plan_fires_on_the_singapore_hour():
    plan = HourlyPlan()
    assert plan.on_start(_ms(10, 15)) is True
    assert hour_floor_ms(_ms(10, 15)) == _ms(10, 0)
    assert next_hour_ms(_ms(10, 15)) == _ms(11, 0)
    assert plan.on_tick(_ms(10, 40)) is False
    assert plan.on_tick(_ms(11, 0)) is True
    assert plan.on_tick(_ms(11, 30)) is False
    assert next_hour_ms(_ms(0, 0)) == _ms(1, 0)


def test_schedule_loop_fetches_at_startup_and_the_following_hour():
    clock = {"ms": _ms(10, 15)}
    calls = []
    state = {"waits": 0}

    class Stop:
        def is_set(self):
            return state["waits"] >= 2

        def wait(self, delay):
            state["waits"] += 1
            assert delay > 60
            if state["waits"] >= 2:
                return True
            clock["ms"] = next_hour_ms(clock["ms"]) + 50
            return False

    stop = Stop()
    run_schedule(stop, lambda: clock["ms"], HourlyPlan(), lambda now: calls.append(now), stop.wait)
    assert calls == [_ms(10, 15), next_hour_ms(_ms(10, 15)) + 50]


def test_manual_refresh_cooldown_and_single_flight(tmp_path: Path):
    calls = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    def fetcher(now):
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            assert release.wait(2)
        return {"items": [_item(1, ts_published=now - 1000)], "sources": {"coindesk": {"ok": True, "count": 1, "error": None}},
                "status": "ok", "fetched_at": now}

    refresher = NewsRefresher(
        tmp_path / "superE.sqlite",
        fetcher=fetcher,
        scorer=lambda snap: snap,
        now_ms=lambda: NOW,
    )
    worker = threading.Thread(target=lambda: refresher.refresh(NOW, manual=True))
    worker.start()
    assert started.wait(2)
    busy = refresher.refresh(NOW, manual=True)
    assert busy["accepted"] is False
    assert busy["refreshing"] is True
    release.set()
    worker.join(2)
    assert calls["n"] == 1
    cooling = refresher.refresh(NOW, manual=True)
    assert cooling["accepted"] is False
    assert cooling["cooling"] is True
    assert cooling["cooldown_remaining_s"] == 60
    assert calls["n"] == 1
    again = refresher.refresh(NOW + 61_000, manual=True)
    assert again["accepted"] is True
    assert calls["n"] == 2


def test_dedup_merges_url_and_similar_titles():
    rows = [
        {"title": "Ethereum ETF sees record inflow", "url": "https://www.CoinDesk.com/a?utm_source=tw",
         "source_tier": 2, "body": "details about the inflow", "ts_published": 2, "source": "coindesk"},
        {"title": "Ethereum ETF sees record inflows", "url": "https://cointelegraph.com/b",
         "source_tier": 2, "body": "", "ts_published": 1, "source": "cointelegraph"},
        {"title": "Ethereum ETF sees record inflow", "url": "https://coindesk.com/a",
         "source_tier": 2, "body": "", "ts_published": 3, "source": "coindesk"},
        {"title": "Fed holds rates steady", "url": "https://decrypt.co/fed",
         "source_tier": 2, "body": "", "ts_published": 4, "source": "decrypt"},
    ]
    kept = live._dedupe(rows)
    titles = [row["title"] for row in kept]
    assert len(kept) == 2
    assert any("ETF" in title for title in titles)
    assert any("Fed" in title for title in titles)
    assert live.canonical_url("https://www.CoinDesk.com/a?utm_source=tw") == live.canonical_url("https://coindesk.com/a")


def test_google_is_capped_and_binance_requires_eth_and_gdelt_is_absent():
    assert live.MAX_AGE_MS == 24 * 3_600_000
    assert len(live.limit_source("google_news_eth", [{"title": str(i)} for i in range(30)])) == 12
    assert len(live.limit_source("coindesk", [{"title": str(i)} for i in range(30)])) == 30
    assert live._relevant("ETH wallet maintenance", "", eth_only=True)
    assert not live._relevant("Bitcoin wallet maintenance", "", eth_only=True)
    assert not live._relevant("Fed rate decision", "", eth_only=True)
    assert live._relevant("Fed rate decision", "", eth_only=False)
    names = [name for name, _url in live.RSS_SOURCES]
    assert "theblock" in names and "gdelt" not in names
    assert "ethereum" in live.GOOGLE_NEWS_URL
    assert all("gdelt" not in url for _name, url in live.RSS_SOURCES)


def test_proxy_comes_from_https_proxy(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    assert live.resolve_proxy(None) == "http://127.0.0.1:7890"
    assert live.resolve_proxy("") is None
    seen = {}

    class Client:
        def __init__(self, *args, **kwargs):
            seen["proxy"] = kwargs.get("proxy")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url):
            class Response:
                status_code = 200
                content = b"<rss></rss>"

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": {"catalogs": []}}

            return Response()

    monkeypatch.setattr(live.httpx, "Client", Client)
    live.fetch_now(NOW)
    assert seen["proxy"] == "http://127.0.0.1:7890"


def test_llm_parse_failure_falls_back_to_rule_and_hides_the_key(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setenv("SUPERE_LLM_BASE_URL", "https://opencode.ai/zen/go")
    monkeypatch.setenv("SUPERE_LLM_API_KEY", SECRET)
    monkeypatch.setenv("SUPERE_LLM_MODEL", "deepseek-v4.1-flash")
    db_path = tmp_path / "superE.sqlite"
    items = [_item(0)]
    live.persist(items, db_path, NOW)

    class Bad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            raise RuntimeError(SECRET)

    with caplog.at_level(logging.DEBUG):
        result = llm.score_panel_items(
            db_path, items, proxy="", sleeper=lambda _delay: None, client_factory=lambda: Bad(),
        )
    score = next(iter(result["scores"].values()))
    assert score.scorer == "rule"
    assert score.summary_zh
    assert result["counts"]["failed"] == 1
    assert SECRET not in caplog.text
    assert SECRET not in json.dumps(result["counts"])


def test_llm_retries_429_then_accepts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERE_LLM_BASE_URL", "https://opencode.ai/zen/go")
    monkeypatch.setenv("SUPERE_LLM_API_KEY", SECRET)
    monkeypatch.setenv("SUPERE_LLM_MODEL", "deepseek-v4.1-flash")
    db_path = tmp_path / "superE.sqlite"
    items = [_item(3)]
    live.persist(items, db_path, NOW)
    state = {"n": 0, "sleeps": []}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            state["n"] += 1
            if state["n"] < 3:
                return _Resp(429, {})
            return _Resp(200, _llm_payload([{"id": 0}]))

    result = llm.score_panel_items(
        db_path, items, proxy="", sleeper=lambda delay: state["sleeps"].append(delay),
        client_factory=lambda: Client(),
    )
    score = next(iter(result["scores"].values()))
    assert state["n"] == 3
    assert state["sleeps"] == pytest.approx([0.8, 1.6])
    assert score.scorer == "llm"
    assert score.label == "利多"
    assert "以太坊" in score.summary_zh


def test_llm_timeout_retries_then_rules(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setenv("SUPERE_LLM_BASE_URL", "https://opencode.ai/zen/go")
    monkeypatch.setenv("SUPERE_LLM_API_KEY", SECRET)
    db_path = tmp_path / "superE.sqlite"
    items = [_item(4)]
    live.persist(items, db_path, NOW)
    sleeps = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            raise httpx.TimeoutException(SECRET)

    with caplog.at_level(logging.DEBUG):
        result = llm.score_panel_items(
            db_path, items, proxy="", sleeper=lambda delay: sleeps.append(delay),
            client_factory=lambda: Client(),
        )
    assert len(sleeps) == 2
    assert next(iter(result["scores"].values())).scorer == "rule"
    assert SECRET not in caplog.text


def test_llm_success_is_not_scored_again(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPERE_LLM_BASE_URL", "https://opencode.ai/zen/go")
    monkeypatch.setenv("SUPERE_LLM_API_KEY", SECRET)
    monkeypatch.setenv("SUPERE_LLM_MODEL", "deepseek-v4.1-flash")
    db_path = tmp_path / "superE.sqlite"
    items = [_item(i) for i in range(12)]
    live.persist(items, db_path, NOW)
    FakeLLM.item_calls = 0
    first = llm.score_panel_items(db_path, items, proxy="", sleeper=lambda _d: None, client_factory=FakeLLM)
    assert first["counts"]["llm"] == 12
    assert FakeLLM.item_calls == 2
    again_items = [_item(i) for i in range(12)]
    live.persist(again_items, db_path, NOW)
    second = llm.score_panel_items(db_path, again_items, proxy="", sleeper=lambda _d: None, client_factory=FakeLLM)
    assert second["counts"]["cached"] == 12
    assert second["counts"]["llm"] == 0
    assert FakeLLM.item_calls == 2


def test_messages_url_appends_anthropic_path():
    assert llm.messages_url("https://opencode.ai/zen/go") == "https://opencode.ai/zen/go/v1/messages"
    assert llm.messages_url("https://opencode.ai/zen/go/v1") == "https://opencode.ai/zen/go/v1/messages"


def test_bias_headline_matches_the_panel_example():
    counts = {"positive": 2, "negative": 5, "neutral": 3}
    assert bias_word(counts) == "偏空"
    assert headline("偏空", counts) == "最近 1 小时偏空 · 利多 2 / 利空 5 / 中性 3"


def test_get_does_not_fetch_and_predict_uses_a_one_hour_cache(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SUPERE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SUPERE_LLM_BASE_URL", raising=False)
    from fastapi.testclient import TestClient
    from supere.train.serve import create_app

    calls = []
    clock = {"s": NOW / 1000}

    def fetcher(now):
        calls.append(now)
        return {
            "items": [_item(1, ts_published=now - 1000)],
            "sources": {"coindesk": {"ok": True, "count": 1, "error": None}},
            "status": "ok",
            "fetched_at": now,
        }

    app = create_app(tmp_path, news_snapshot=fetcher, clock=lambda: clock["s"], schedule=False)
    client = TestClient(app)
    assert client.get("/api/news").status_code == 200
    assert calls == []
    first = client.post("/api/predict")
    assert first.status_code == 200
    assert len(calls) == 1
    assert first.json()["news"]["items"]
    clock["s"] += 30 * 60
    client.post("/api/predict")
    assert len(calls) == 1
    clock["s"] += 31 * 60
    client.post("/api/predict")
    assert len(calls) == 2
    refreshed = client.post("/api/news/refresh")
    assert refreshed.status_code == 200
    body = refreshed.json()
    assert body["accepted"] is True
    assert body["next_refresh_at"]
    cooled = client.post("/api/news/refresh")
    assert cooled.status_code == 200
    assert cooled.json()["cooling"] is True
    assert cooled.json()["cooldown_remaining_s"] > 0
    assert len(calls) == 3


def test_startup_fetches_once(tmp_path: Path):
    from fastapi.testclient import TestClient
    from supere.train.serve import create_app

    calls = []

    def fetcher(now):
        calls.append(now)
        return {"items": [], "sources": {"coindesk": {"ok": True, "count": 0, "error": None}},
                "status": "empty", "fetched_at": now}

    app = create_app(tmp_path, news_snapshot=fetcher, clock=lambda: NOW / 1000, schedule=True)
    with TestClient(app) as client:
        for _ in range(50):
            if calls:
                break
            time.sleep(0.02)
        assert len(calls) == 1
        client.get("/api/news")
        assert len(calls) == 1


def test_refresh_end_to_end_with_mock_llm(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setenv("SUPERE_LLM_BASE_URL", "https://opencode.ai/zen/go")
    monkeypatch.setenv("SUPERE_LLM_API_KEY", SECRET)
    monkeypatch.setenv("SUPERE_LLM_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setattr(llm.httpx, "Client", FakeLLM)
    FakeLLM.item_calls = 0
    from fastapi.testclient import TestClient
    from supere.train.serve import create_app

    def fetcher(now):
        return {
            "items": [_item(i, ts_published=now - (i + 1) * 60_000) for i in range(12)],
            "sources": {
                "coindesk": {"ok": True, "count": 12, "error": None},
                "theblock": {"ok": False, "count": 0, "error": "timeout"},
            },
            "status": "partial",
            "fetched_at": now,
        }

    app = create_app(tmp_path, news_snapshot=fetcher, clock=lambda: NOW / 1000, schedule=False)
    client = TestClient(app)
    with caplog.at_level(logging.DEBUG):
        refreshed = client.post("/api/news/refresh")
        listed = client.get("/api/news")
    assert refreshed.status_code == 200
    body = refreshed.json()
    assert len(body["items"]) >= 10
    assert all(item["summary_zh"] and "以太坊" in item["summary_zh"] for item in body["items"])
    assert all(item["sentiment_label"] in ("利多", "利空", "中性") for item in body["items"])
    assert all(item["scored_by"] == "llm" for item in body["items"])
    assert body["summary"]["headline"].startswith("最近 1 小时")
    assert body["last_refresh_at"]
    assert body["next_refresh_at"] > body["last_refresh_at"]
    assert body["sources"]["coindesk"]["ok"] is True
    assert body["sources"]["theblock"]["ok"] is False
    assert any(item["importance"] >= 2 for item in body["items"])
    got = listed.json()
    assert len(got["items"]) >= 10
    assert got["items"][0]["summary_zh"]
    blob = refreshed.text + listed.text + caplog.text
    assert SECRET not in blob
    # The key must not land in the database either.
    db = Database(tmp_path / "superE.sqlite")
    try:
        stored = " ".join(str(row["summary_zh"] or "") + str(row["score_json"] or "")
                          for row in db.conn.execute("SELECT summary_zh, score_json FROM news"))
    finally:
        db.close()
    assert SECRET not in stored


def _insert(db: Database, source: str, title: str, *, relevance: float = 0.9,
            scored_by: str = "rule_fallback", now: int = NOW) -> None:
    db.insert_news({
        "ts_published": now - 60_000, "ts_ingested": now - 60_000, "ts_local": now - 60_000,
        "source": source, "source_tier": 2, "lang": "en", "title": title, "body": "",
        "url": f"https://example.com/{source}/{title[:12]}", "url_hash": f"h-{source}-{title[:8]}",
        "text_simhash": "0", "status": "scored", "available_at": now - 60_000,
        "timing_quality": "ingested", "direction_score": 0.2, "eth_relevance": relevance,
        "urgency": 0.3, "scored_by": scored_by, "score_version": "old",
        "content_hash": f"c-{source}-{title[:8]}",
        "summary_zh": f"{title}。这条消息被判断为中性。", "sentiment_label": "中性",
    })


def test_panel_hides_gdelt_and_low_relevance(tmp_path: Path):
    from supere.train.serve import _news_payload

    db = Database(tmp_path / "superE.sqlite")
    _insert(db, "gdelt", "Miami-Dade tax office accepts bitcoin", relevance=0.9)
    _insert(db, "coindesk", "Ethereum staking queue grows", relevance=0.9)
    _insert(db, "coindesk", "Solana wallet teams with SpaceX", relevance=0.1)
    _insert(db, "google_news", "Fed holds rates as inflation cools", relevance=0.6)
    db.close()
    payload = _news_payload(tmp_path, 80, 24, now_ms=NOW)
    titles = [row["title"] for row in payload["items"]]
    other = [row["title"] for row in payload["other"]]
    assert "Ethereum staking queue grows" in titles
    assert "Fed holds rates as inflation cools" in titles
    assert "Miami-Dade tax office accepts bitcoin" not in titles
    assert "Miami-Dade tax office accepts bitcoin" not in other
    assert "Solana wallet teams with SpaceX" in other
    assert payload["aggregate"]["count"] == len(payload["items"])


def test_fetch_keeps_macro_and_bitcoin_and_drops_unrelated():
    assert live._relevant("Fed holds rates, dollar firms", "")
    assert live._relevant("Bitcoin ETF sees record inflows", "")
    assert live._relevant("ETH liquidations spike", "")
    assert not live._relevant("Solana wallet teams with SpaceX", "")
    assert not live._relevant("US home price index rises", "")
    assert not live._relevant("Dogecoin meme of the day", "")


def test_backfill_scores_rule_rows_and_skips_gdelt(tmp_path: Path, monkeypatch):
    from supere.news.scoring import pending_llm_items

    monkeypatch.setenv("SUPERE_LLM_BASE_URL", "https://opencode.ai/zen/go")
    monkeypatch.setenv("SUPERE_LLM_API_KEY", SECRET)
    db = Database(tmp_path / "superE.sqlite")
    _insert(db, "gdelt", "Ontario crypto founder kidnapped", scored_by="rule_fallback")
    _insert(db, "coindesk", "Ethereum client release", scored_by="rule_fallback")
    _insert(db, "decrypt", "Already translated ETH note", scored_by="llm")
    db.close()
    pending = pending_llm_items(
        tmp_path / "superE.sqlite", tuple(sorted(live.PANEL_SOURCES)), NOW - 24 * 3_600_000, limit=8,
    )
    assert [row["source"] for row in pending] == ["coindesk"]
    seen = []

    def scorer(snap):
        seen.extend(item["title"] for item in snap["items"])
        return snap

    refresher = NewsRefresher(
        tmp_path / "superE.sqlite", fetcher=lambda now: {"items": []}, scorer=scorer,
        now_ms=lambda: NOW,
    )
    assert refresher.backfill_once() == 1
    assert seen == ["Ethereum client release"]
    refresher._refreshing = True
    assert refresher.backfill_once() == 0
