from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from supere.intervals import INTERVAL_MS
from supere.storage.schema import init_schema
from supere.train import newsfeat, quantile_v3, rv
from supere.train.data import Bars

START = 1_700_000_000_000 // 300_000 * 300_000  # aligned to 5m


def _insert_klines(conn, market: str, symbol: str, interval: str,
                   open_time: np.ndarray, o, h, l, c, v, qv, trades, tb, tbq,
                   ingested: int) -> None:
    rows = []
    step = INTERVAL_MS[interval]
    for i in range(open_time.shape[0]):
        rows.append((market, symbol, interval, int(open_time[i]), int(open_time[i]) + step - 1,
                     f"{o[i]:.8f}", f"{h[i]:.8f}", f"{l[i]:.8f}", f"{c[i]:.8f}",
                     f"{v[i]:.8f}", f"{qv[i]:.8f}", int(trades[i]), f"{tb[i]:.8f}",
                     f"{tbq[i]:.8f}", "test", ingested, ingested))
    conn.executemany("INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def _make_market_db(path: Path, n_5m: int = 4000, seed: int = 5) -> Path:
    rng = np.random.default_rng(seed)
    n_1m = n_5m * 5
    r1 = rng.normal(0.0, 0.0008, n_1m)
    close = 2000.0 * np.exp(np.cumsum(r1))
    opn = np.concatenate([[2000.0], close[:-1]])
    high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0, 0.0004, n_1m)))
    low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0, 0.0004, n_1m)))
    vol = rng.uniform(1.0, 20.0, n_1m)
    qv = vol * close
    trades = rng.integers(1, 200, n_1m)
    tb = vol * rng.uniform(0.35, 0.65, n_1m)
    tbq = tb * close
    t1 = START + np.arange(n_1m, dtype=np.int64) * 60_000

    n = n_5m
    t5 = t1.reshape(n, 5)[:, 0]
    o5 = opn.reshape(n, 5)[:, 0]
    c5 = close.reshape(n, 5)[:, -1]
    h5 = high.reshape(n, 5).max(axis=1)
    l5 = low.reshape(n, 5).min(axis=1)
    v5 = vol.reshape(n, 5).sum(axis=1)
    qv5 = qv.reshape(n, 5).sum(axis=1)
    tr5 = trades.reshape(n, 5).sum(axis=1)
    tb5 = tb.reshape(n, 5).sum(axis=1)
    tbq5 = tbq.reshape(n, 5).sum(axis=1)

    conn = sqlite3.connect(path)
    try:
        init_schema(conn)
        ingested = int(time.time() * 1000)
        _insert_klines(conn, "um", "ETHUSDT", "1m", t1, opn, high, low, close,
                       vol, qv, trades, tb, tbq, ingested)
        _insert_klines(conn, "um", "ETHUSDT", "5m", t5, o5, h5, l5, c5,
                       v5, qv5, tr5, tb5, tbq5, ingested)
        _insert_klines(conn, "um_mark", "ETHUSDT", "5m", t5, o5, h5, l5, c5,
                       v5, qv5, tr5, tb5, tbq5, ingested)
        _insert_klines(conn, "um_index", "ETHUSDT", "5m", t5, o5, h5, l5, c5,
                       v5, qv5, tr5, tb5, tbq5, ingested)
        _insert_klines(conn, "um_premium", "ETHUSDT", "5m", t5, o5, h5, l5, c5,
                       v5, qv5, tr5, tb5, tbq5, ingested)
        conn.commit()
    finally:
        conn.close()
    return path


def _bars_1m(n: int = 20_000, seed: int = 9) -> Bars:
    rng = np.random.default_rng(seed)
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))
    opn = np.concatenate([[2000.0], close[:-1]])
    high = np.maximum(opn, close) * 1.0002
    low = np.minimum(opn, close) * 0.9998
    vol = rng.uniform(1.0, 20.0, n)
    t = START + np.arange(n, dtype=np.int64) * 60_000
    return Bars(interval="1m", open_time=t, o=opn, h=high, l=low, c=close,
                volume=vol, quote_volume=vol * close, trades=rng.integers(1, 100, n).astype(float),
                taker_buy_base=vol * 0.5, taker_buy_quote=vol * close * 0.5)


def test_rv_features_are_causal_under_truncation():
    bars = _bars_1m()
    decisions = bars.open_time[2000:8000:5] + 60_000
    full = rv.build(bars, decisions, 300_000)
    cut = bars.slice_to(7000)
    part = rv.build(cut, decisions, 300_000)
    assert full.names == part.names
    mask = decisions <= cut.close_time[-1]
    np.testing.assert_allclose(full.X[mask], part.X[mask], equal_nan=True, rtol=1e-10, atol=1e-12)
    # Future label is only a label and can differ after truncation.
    assert np.isfinite(full.rv_future).any()
    idx = {n: i for i, n in enumerate(full.names)}
    for name in ("rv_5m", "logrv_5m", "logrv_360m", "bpv_5m", "micro_taker_imb_5m"):
        assert np.isfinite(full.X[:, idx[name]]).any(), name


def test_har_matrix_and_qlike_are_finite():
    bars = _bars_1m()
    decisions = bars.open_time[3000:9000:5] + 60_000
    rvf = rv.build(bars, decisions, 300_000)
    names, z = rv.har_matrix(rvf, "har_cj", 5)
    assert z.shape[1] == len(names)
    assert np.isfinite(z).any()
    score = rv.qlike(np.array([1e-8, 4e-8]), np.array([2e-8, 2e-8]))
    assert np.isfinite(score)


def test_news_features_respect_available_at(tmp_path: Path):
    from supere.storage.db import Database
    db_path = tmp_path / "superE.sqlite"
    db = Database(db_path)
    t0 = START
    t1 = START + 30 * 60_000
    for ts, title in ((t0, "ETH ETF approval"), (t1, "exchange hack")):
        db.insert_news({
            "ts_published": ts, "ts_ingested": ts, "ts_local": ts, "source": "test",
            "source_tier": 2, "lang": "en", "title": title, "body": "", "url": title,
            "url_hash": title, "text_simhash": "0", "status": "scored",
            "available_at": ts, "timing_quality": "ingested", "direction_score": 0.8,
            "eth_relevance": 0.9, "urgency": 0.9, "magnitude": 0.8, "event_type": "test",
        })
    db.close()
    news = newsfeat.load_news(db_path)
    assert news.n == 2
    decisions = np.array([t0 - 1, t0, t1], dtype=np.int64)
    cols = newsfeat.build_columns(decisions, 300_000, news)
    assert np.isnan(cols["news_count_1h"][0])
    assert cols["news_count_1h"][1] == 1.0
    assert cols["news_count_1h"][2] == 2.0
    assert cols["news_coverage"][0] == 0.0
    assert cols["news_coverage"][1] == 1.0


def test_v3_fit_and_inference_contract(tmp_path: Path):
    data_root = tmp_path
    db_path = data_root / "superE.sqlite"
    _make_market_db(db_path, n_5m=4000)
    overrides = {
        "snapshot_ms": None, "limit_bars": None, "n_folds": 2,
        "n_estimators": 8, "min_child_samples": 20, "features": "rv",
        "sigma_model": "har", "shape": "quantile", "conformal": "norm",
        "n_jobs": 2,
    }
    cfg = quantile_v3.default_config("5m", overrides)
    S = quantile_v3.load_samples(cfg, db_path, data_root)
    assert S.m > 2000
    base = quantile_v3.q.baseline_quantiles(S, quantile_v3._qcfg(cfg))
    cv = quantile_v3.run_cv(S, cfg, base, lambda _msg: None)
    assert "gates" in cv and "model" in cv
    assert cv["model"]["cov50"] == pytest.approx(cv["model"]["cov50"])
    hold = quantile_v3.run_holdout(S, cfg, base, bool(cv.get("median_fallback")))
    assert "model" in hold and "baselines" in hold
    assert hold["model"]["pinball_combined_bps"] > 0
    fit = quantile_v3.fit_block(S, np.arange(S.m, dtype=np.int64), cfg)
    quantile_v3.save_artifacts(data_root, S, cfg, fit,
                               {"status": "仅参考", "gates": cv["gates"]}, False)
    art = quantile_v3.load_v3(data_root, "5m")
    assert art is not None and art.meta["schema_version"] == 3
    now_ms = int(S.decision_ms[-1]) + 1
    out = quantile_v3.predict_latest(data_root, "5m", art, now_ms)
    pf = out["price_forecast"]
    assert pf is not None
    for key in ("median", "p25", "p75", "p10", "p90", "width50_bps", "width80_bps"):
        assert pf[key] is not None and np.isfinite(pf[key])
    assert pf["p10"] <= pf["p25"] <= pf["median"] <= pf["p75"] <= pf["p90"]


def test_news_api_payload(tmp_path: Path):
    from supere.storage.db import Database
    from supere.train.serve import _news_payload
    db_path = tmp_path / "superE.sqlite"
    db = Database(db_path)
    now = int(time.time() * 1000)
    db.insert_news({
        "ts_published": now - 1000, "ts_ingested": now - 1000, "ts_local": now - 1000,
        "source": "test", "source_tier": 2, "lang": "en", "title": "ETH rally",
        "body": "", "url": "u", "url_hash": "h", "text_simhash": "0", "status": "scored",
        "available_at": now - 1000, "timing_quality": "ingested",
        "direction_score": 0.9, "eth_relevance": 0.9, "urgency": 0.8,
    })
    db.close()
    payload = _news_payload(tmp_path, 50, 72)
    assert payload["status"] == "ok"
    assert payload["summary"]["label"] == "利多"
    assert payload["summary"]["high_urgency"] == 1
    assert payload["items"][0]["available_at_utc8"]
    from fastapi.testclient import TestClient
    from supere.train.serve import create_app
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/news?limit=10&since_hours=72")
    assert response.status_code == 200
    assert response.json()["items"][0]["title"] == "ETH rally"
