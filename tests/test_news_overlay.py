"""Click-time news scoring, overlay math and the passive prediction route."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from supere.news import live, overlay, scoring
from supere.storage.db import Database
from supere.train.serve import create_app

NOW = 1_800_000_000_000


def item(**kwargs):
    base = {
        "id": 1,
        "title": "Ethereum ETF inflow",
        "source": "test",
        "url": "https://example.com/a",
        "available_at": NOW - 60_000,
        "ts_published": NOW - 60_000,
        "direction": 0.8,
        "urgency": 0.8,
        "relevance": 0.9,
        "event_type": "flows",
        "summary_zh": "ETH ETF 资金流入增加，可能提升买盘。",
        "sentiment_label": "利多",
        "scored_by": "deepseek",
        "score_version": "ds-v2:p2",
    }
    base.update(kwargs)
    return base


def forecast():
    return {"median": 100.0, "p10": 98.0, "p25": 99.0, "p75": 101.0, "p90": 102.0,
            "width50_bps": 200.0, "width80_bps": 400.0, "ref_close": 100.0}


def test_empty_news_uses_documented_defaults():
    agg = overlay.aggregate([], NOW, 3_600_000)
    assert (agg["u"], agg["d"], agg["r"]) == (0.0, 0.0, 0.0)
    assert agg["gate"]["triggered"] is False
    out = overlay.apply_overlay(forecast(), agg)
    assert out["applied"] is False
    assert out["width_multiplier"] == 1.0
    assert out["median_delta"] == 0.0
    assert out["adjusted"] == out["model"]


def test_gate_below_threshold_does_not_adjust():
    agg = overlay.aggregate([item(urgency=0.2, relevance=0.5)], NOW, 3_600_000)
    assert agg["gate"]["u_r"] == pytest.approx(0.1)
    assert agg["gate"]["triggered"] is False
    out = overlay.apply_overlay(forecast(), agg)
    assert out["applied"] is False
    assert out["width_multiplier"] == 1.0


def test_positive_direction_moves_median_up_and_widens_only():
    agg = overlay.aggregate([item(direction=0.9)], NOW, 3_600_000)
    raw = forecast()
    out = overlay.apply_overlay(raw, agg)
    assert out["applied"] is True
    assert out["adjusted"]["median"] > raw["median"]
    assert out["width_multiplier"] > 1.0
    assert out["adjusted"]["p75"] - out["adjusted"]["p25"] > raw["p75"] - raw["p25"]
    assert out["adjusted"]["p10"] <= out["adjusted"]["p25"] <= out["adjusted"]["median"]
    assert out["adjusted"]["median"] <= out["adjusted"]["p75"] <= out["adjusted"]["p90"]


def test_negative_direction_moves_median_down():
    out = overlay.apply_overlay(forecast(), overlay.aggregate([item(direction=-0.9)], NOW, 3_600_000))
    assert out["applied"] is True
    assert out["adjusted"]["median"] < forecast()["median"]


def test_zero_relevance_never_adjusts():
    agg = overlay.aggregate([item(relevance=0.0, urgency=1.0)], NOW, 3_600_000)
    assert agg["gate"]["triggered"] is False
    assert overlay.apply_overlay(forecast(), agg)["applied"] is False


def test_max_width_is_35_percent():
    agg = overlay.aggregate([item(direction=0.0, urgency=1.0, relevance=1.0)], NOW, 3_600_000)
    out = overlay.apply_overlay(forecast(), agg)
    assert out["width_multiplier"] == pytest.approx(1.35)
    assert out["width_multiplier"] <= 1.35 + 1e-12


def test_median_delta_is_capped_and_quantiles_never_cross():
    raw = {"median": 100.0, "p10": 99.9, "p25": 99.95, "p75": 100.05, "p90": 100.1,
           "ref_close": 100.0}
    agg = overlay.aggregate([item(direction=1.0, urgency=1.0, relevance=1.0)], NOW, 3_600_000)
    out = overlay.apply_overlay(raw, agg)
    assert abs(out["median_delta"]) <= out["median_delta_cap"] + 1e-12
    values = [out["adjusted"][key] for key in ("p10", "p25", "median", "p75", "p90")]
    assert values == sorted(values)


def test_future_news_is_not_visible():
    future = item(available_at=NOW + 10_000, ts_published=NOW + 10_000)
    agg = overlay.aggregate([future], NOW, 3_600_000)
    assert agg["count"] == 0
    assert agg["gate"]["triggered"] is False


def test_top_news_are_ranked_and_limited_to_three():
    rows = [
        item(id=index, title=f"ETH news {index}", urgency=0.3 + index * 0.2,
             ts_published=NOW - index * 60_000, available_at=NOW - index * 60_000)
        for index in range(6)
    ]
    agg = overlay.aggregate(rows, NOW, 3_600_000)
    assert len(agg["top_news"]) == 3
    assert agg["top_news"][0]["urgency"] >= agg["top_news"][1]["urgency"]


def test_alert_empty_and_high_are_plain_chinese():
    empty = overlay.aggregate([], NOW, 3_600_000)["alert"]
    assert empty["title"] == "暂无显著异常"
    high = overlay.aggregate([item(event_type="security", urgency=0.95)], NOW, 3_600_000)["alert"]
    assert high["level"] == "high"
    assert high["urgency"] == "高"
    assert "安全" in high["why"]
    assert high["what"]


def test_apply_overlay_does_not_mutate_model_values():
    raw = forecast()
    before = dict(raw)
    overlay.apply_overlay(raw, overlay.aggregate([item()], NOW, 3_600_000))
    assert raw == before


def test_rule_summary_is_natural_and_has_no_removed_wording():
    text = scoring.rule_summary_zh("Ethereum ETF approved", "regulation", 0.8, 0.9, 0.9)
    assert "ETH" in text
    assert "利多" in text
    for term in ("仅参考", "供参考", "仅供参考", "非建议"):
        assert term not in text
    assert text.count("。") <= 2


def test_scoring_cache_reuses_content_and_version(tmp_path: Path):
    db_path = tmp_path / "superE.sqlite"
    db = Database(db_path)
    title = "Ethereum ETF inflow"
    digest = scoring.content_hash(title, "")
    news_id = db.insert_news({
        "ts_published": NOW - 1000, "ts_ingested": NOW - 1000, "ts_local": NOW - 1000,
        "source": "test", "source_tier": 2, "lang": "en", "title": title, "body": "",
        "url": "u", "url_hash": "h", "text_simhash": "0", "status": "scored",
        "available_at": NOW - 1000, "timing_quality": "click_fetch",
        "direction_score": 0.9, "eth_relevance": 0.9, "urgency": 0.8,
        "scored_by": "deepseek", "score_version": scoring.score_version(),
        "content_hash": digest, "summary_zh": "缓存摘要。", "sentiment_label": "利多",
        "score_json": json.dumps({"direction": 0.9, "eth_relevance": 0.9, "urgency": 0.8}),
    })
    db.close()
    result = scoring.score_items(db_path, [{"id": news_id, "title": title, "body": ""}],
                                 use_deepseek=False)
    assert result["counts"]["cached"] == 1
    assert result["scores"][digest].summary_zh == "缓存摘要。"


def test_news_api_has_alert_and_summaries(tmp_path: Path):
    from supere.train.serve import _news_payload
    db = Database(tmp_path / "superE.sqlite")
    now = int(time.time() * 1000)
    db.insert_news({
        "ts_published": now - 1000, "ts_ingested": now - 1000, "ts_local": now - 1000,
        "source": "test", "source_tier": 2, "lang": "en", "title": "Ethereum ETF inflow",
        "body": "", "url": "u", "url_hash": "h", "text_simhash": "0", "status": "scored",
        "available_at": now - 1000, "timing_quality": "ingested",
        "direction_score": 0.9, "eth_relevance": 0.9, "urgency": 0.8,
        "event_type": "flows",
    })
    db.close()
    payload = _news_payload(tmp_path, 50, 72)
    assert payload["status"] == "ok"
    assert payload["summary"]["label"] == "利多"
    assert payload["items"][0]["sentiment_label"] == "利多"
    assert payload["items"][0]["summary_zh"]
    assert payload["alert"]["level"] in ("watch", "high")


def test_click_route_fetches_news_then_returns_model_and_logs_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "super-secret-test-key")
    monkeypatch.setattr(scoring, "score_batch_deepseek", lambda *args, **kwargs: None)

    def snapshot(now_ms):
        return {
            "items": [item(id=None, available_at=now_ms, ts_published=now_ms - 1000)],
            "sources": {"test": {"ok": True, "count": 1, "error": None}},
            "status": "ok",
            "fetched_at": now_ms,
        }

    app = create_app(tmp_path, news_snapshot=snapshot, clock=lambda: NOW / 1000)
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.post("/api/predict")
    assert response.status_code == 200
    payload = response.json()
    assert payload["trigger"] == "user_click"
    assert payload["news"]["items"][0]["summary_zh"]
    assert payload["news"]["alert"]["title"]
    assert len(payload["items"]) == 5
    log = tmp_path / "logs" / "predict_clicks.jsonl"
    assert log.exists()
    text = log.read_text(encoding="utf-8")
    assert "super-secret-test-key" not in text
    record = json.loads(text.splitlines()[-1])
    assert record["trigger"] == "user_click"
    assert record["news"]["sources"]["test"]["ok"] is True
    assert record["news"]["scorer_version"]
    assert len(record["horizons"]) == 5
    assert "content_hash" in record["news"]["items"][0]


def test_news_failure_does_not_block_the_click(tmp_path: Path):
    def broken(now_ms):
        raise TimeoutError("news timeout")

    app = create_app(tmp_path, news_snapshot=broken, clock=lambda: NOW / 1000)
    from fastapi.testclient import TestClient
    response = TestClient(app).post("/api/predict")
    assert response.status_code == 200
    payload = response.json()
    assert payload["news"]["status"] == "error"
    assert len(payload["items"]) == 5
    assert all("news_overlay" in row for row in payload["items"])


def test_rss_parser_extracts_eth_items_only():
    rss = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Ethereum upgrade ships</title><link>https://example.com/eth</link>
      <description>ETH validators adopt the change.</description>
      <pubDate>Wed, 05 Nov 2026 10:00:00 GMT</pubDate></item>
      <item><title>Dogecoin meme of the day</title><link>https://example.com/doge</link>
      <description>Nothing relevant here.</description>
      <pubDate>Wed, 05 Nov 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    # 用与 RSS pubDate 同一时刻的 now，避免 MAX_AGE_MS（24 小时）随真实时间过期。
    from datetime import datetime, timezone

    now_ms = int(datetime(2026, 11, 5, 10, 5, tzinfo=timezone.utc).timestamp() * 1000)
    rows = live._rss_items("cointelegraph", rss, now_ms)
    assert len(rows) == 1
    assert rows[0]["title"] == "Ethereum upgrade ships"
    assert rows[0]["url"] == "https://example.com/eth"
    assert rows[0]["ts_published"] is not None
