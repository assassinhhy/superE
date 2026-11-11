"""Live v6 CQR attach-point tests (plan 计划-第六版上线 item 7).

Pure synthetic state machines plus one HTTP contract test.  No shared data,
no model directory writes, no real database is modified.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from supere.news import overlay
from supere.train import calib_live as live
from supere.train import calib_v6
from supere.train import serve

H = 3_600_000
STEP = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": H, "4h": 4 * H}
START = 1_700_000_000_000
REF = 2000.0
SIGMA = 0.004


class FakeArt:
    directory = None

    def __init__(self, model_id: str = "fake-model", symbol: str = "ETHUSDT"):
        self.model_id = model_id
        self.meta = {"config": {"symbol": symbol}, "model_id": model_id}


class FakeLabels:
    """In-memory label store; mirrors SqliteLabelSource's ``label`` contract."""

    def __init__(self):
        self.data: dict[tuple[str, int], float] = {}
        self.calls: list[tuple[str, int, int]] = []

    def set(self, interval: str, decision_ms: int, y: float) -> None:
        self.data[(interval, int(decision_ms))] = float(y)

    def label(self, interval: str, decision_ms: int, now_ms: int) -> float | None:
        self.calls.append((interval, int(decision_ms), int(now_ms)))
        return self.data.get((interval, int(decision_ms)))


def _pf(decision_ms: int, *, ref: float = REF, sigma: float = SIGMA,
        half_width: float = 0.004) -> dict:
    return {
        "ref_close": ref,
        "median": ref,
        "p10": ref * math.exp(-1.7 * half_width),
        "p25": ref * math.exp(-0.9 * half_width),
        "p75": ref * math.exp(0.9 * half_width),
        "p90": ref * math.exp(1.7 * half_width),
        "sigma_hat": sigma,
        "decision_ms": int(decision_ms),
        "asof_utc8": "x",
    }


def _cal(tmp_path: Path, labels: FakeLabels, art: FakeArt, clock_ms: list[int]):
    return live.LiveCalibrator(tmp_path, clock=lambda: clock_ms[0] / 1000.0,
                               label_source=labels), art


def _feed(cal, art, interval, labels, bars: int, *, start=START, delay=60_000,
          sigma=SIGMA, now_extra=60_000):
    """Feed ``bars`` distinct decisions; at bar i the label of bar i-1 is due."""
    step = STEP[interval]
    outs = []
    for i in range(bars):
        d = start + i * step
        if i > 0:
            k = (d - START) // step
            labels.set(interval, d - step, 1e-4 * math.sin(k))
        now = d + now_extra
        pf = _pf(d, sigma=sigma)
        outs.append(cal.decorate(interval, art, pf, now))
    return outs


# ----------------------------------------------------------------- policy

def test_interval_policy_five_periods(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    out = {}
    for iv in ("5m", "15m", "30m", "1h", "4h"):
        out[iv] = cal.decorate(iv, art, _pf(START), START)
    assert out["15m"]["calibration"]["calib"] == "none"
    assert out["30m"]["calibration"]["calib"] == "none"
    assert out["15m"]["calibration"]["reason"] == "interval_policy"
    for iv in ("5m", "1h", "4h"):
        assert out[iv]["calibration"]["policy"] == "cqr"
    status = cal.status()
    assert set(status["intervals"]) == {"5m", "1h", "4h"}
    assert status["policy"]["cqr"] == ["5m", "1h", "4h"]
    assert status["policy"]["none"] == ["15m", "30m"]


def test_none_intervals_do_not_create_state(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    cal.decorate("15m", FakeArt(), _pf(START), START)
    cal.decorate("30m", FakeArt(), _pf(START), START)
    assert cal.status()["intervals"] == {}
    assert not list(tmp_path.glob("calib_v6/**/*.json"))


# ------------------------------------------------------------------- cold

def test_cold_below_200_stays_none_then_cqr(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    outs = _feed(cal, art, "1h", labels, 151)
    assert outs[-1]["calibration"]["calib"] == "none"
    assert outs[-1]["calibration"]["reason"] == "cold"
    assert outs[-1]["calibration"]["n_mature"] == 150
    outs = _feed(cal, art, "1h", labels, 60, start=START + 151 * H)
    final = outs[-1]["calibration"]
    assert final["calib"] == "cqr" and final["active"] is True
    assert final["n_mature"] >= 200
    st = cal.status()["intervals"]["1h"]
    assert st["n_mature"] >= 200
    assert st["window_days"] == 28.0


def test_4h_window_is_90_days(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    _feed(cal, FakeArt(), "4h", labels, 2)
    assert cal.status()["intervals"]["4h"]["window_days"] == 90.0


# --------------------------------------------------------------- maturity

@pytest.mark.parametrize("interval", ["1h", "4h"])
def test_label_not_learned_before_60s_delay(tmp_path, interval):
    step = STEP[interval]
    labels = FakeLabels()
    clock = [START]
    cal = live.LiveCalibrator(tmp_path, clock=lambda: clock[0] / 1000.0,
                              label_source=labels)
    art = FakeArt()
    d0 = START
    cal.decorate(interval, art, _pf(d0), d0)
    labels.set(interval, d0, 0.001)
    d1 = d0 + step
    # Exactly at the next decision the label is 60 s early: it must be ignored.
    cal.decorate(interval, art, _pf(d1), d1)
    assert cal.status()["intervals"][interval]["n_mature"] == 0
    # 60 s later the same decision bar is a duplicate; the state learns once.
    clock[0] = d1 + 60_000
    cal.decorate(interval, art, _pf(d1), d1 + 60_000)
    st = cal.status()["intervals"][interval]
    assert st["n_mature"] == 1
    assert st["live_rows"] == 2  # d0 and d1, never the duplicate


def test_late_label_consumed_in_order(tmp_path):
    labels = FakeLabels()
    clock = [START]
    cal = live.LiveCalibrator(tmp_path, clock=lambda: clock[0] / 1000.0,
                              label_source=labels)
    art = FakeArt()
    d0, d1 = START, START + H
    cal.decorate("1h", art, _pf(d0), d0)
    cal.decorate("1h", art, _pf(d1), d1)
    state = cal._state("1h", art)
    # Only the later label exists; the earlier one must block ordered use.
    labels.set("1h", d1, 0.002)
    assert state.advance(d1 + 60_000, labels) is False
    assert state.engine.band50.count == 0
    labels.set("1h", d0, 0.001)
    assert state.advance(d1 + H + 61_000, labels) is True
    assert state.engine.band50.count == 2
    order = list(state.engine.band50.decision[:2])
    assert order == [d0, d1]


# -------------------------------------------------------- idempotence/restart

def test_duplicate_request_returns_identical_output_once(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    _feed(cal, art, "1h", labels, 220)
    d = START + 219 * H
    labels.set("1h", d, 0.001)
    now = d + H + 60_000
    a = cal.decorate("1h", art, _pf(d), now)
    n_before = cal.status()["intervals"]["1h"]["n_mature"]
    seen_before = len(cal._state("1h", art).engine.seen_observe)
    b = cal.decorate("1h", art, _pf(d), now)  # same decision, same clock
    assert a["calibrated_price_forecast"] == b["calibrated_price_forecast"]
    assert b["calibration"]["reason"] == "duplicate"
    assert cal.status()["intervals"]["1h"]["n_mature"] == n_before
    assert len(cal._state("1h", art).engine.seen_observe) == seen_before


def test_restart_matches_uninterrupted(tmp_path):
    rows = 240
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    art = FakeArt()

    la = FakeLabels()
    ca = live.LiveCalibrator(a_dir, clock=lambda: 1.0, label_source=la)
    continuous = _feed(ca, art, "1h", la, rows)

    lb = FakeLabels()
    cb = live.LiveCalibrator(b_dir, clock=lambda: 1.0, label_source=lb)
    first = _feed(cb, art, "1h", lb, 120)
    cb2 = live.LiveCalibrator(b_dir, clock=lambda: 1.0, label_source=lb)
    # Seed the label of the last pre-restart bar, as the continuous run had it.
    lb.set("1h", START + 119 * H, 1e-4 * math.sin(120.0))
    second = _feed(cb2, art, "1h", lb, rows - 120, start=START + 120 * H)
    for got, want in zip(first + second, continuous):
        assert got["calibrated_price_forecast"] == want["calibrated_price_forecast"]
    assert (cb2.status()["intervals"]["1h"]["n_mature"]
            == ca.status()["intervals"]["1h"]["n_mature"])


def test_concurrent_same_decision_is_learned_once(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    _feed(cal, art, "1h", labels, 210)
    d = START + 209 * H
    now = d + 60_000  # same clock as the original request: nothing matures
    before = cal.status()["intervals"]["1h"]["n_mature"]
    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            results.append(cal.decorate("1h", art, _pf(d), now))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 8
    assert all(r["calibrated_price_forecast"] == results[0]["calibrated_price_forecast"]
               for r in results)
    assert cal.status()["intervals"]["1h"]["n_mature"] == before
    state = cal._state("1h", art)
    assert state.live_rows == 210


def test_out_of_order_decision_does_not_raise(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    cal.decorate("1h", art, _pf(START + H), START + H)
    out = cal.decorate("1h", art, _pf(START), START)
    assert out["calibration"]["reason"] == "out_of_order"
    assert out["calibration"]["calib"] == "none"
    assert out["calibrated_price_forecast"] == out["raw_price_forecast"]


# ------------------------------------------------------------- degradation

def test_corrupt_snapshot_quarantines_and_degrades(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    _feed(cal, art, "1h", labels, 3)
    path = cal._state("1h", art).path
    path.write_text("{not json", encoding="utf-8")
    labels2 = FakeLabels()
    cal2 = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels2)
    out = cal2.decorate("1h", art, _pf(START + 10 * H), START + 10 * H)
    assert out["calibration"]["calib"] == "none"
    assert "snapshot_quarantined" in (out["calibration"]["alert"] or "")
    assert list(path.parent.glob("state.corrupt-*.json"))
    # A fresh, valid state replaces it; the damaged bytes are not reused.
    assert path.exists()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == live.LIVE_SCHEMA
    assert "engine" in doc


def test_model_hash_change_never_mixes_state(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art_a = FakeArt("model-a")
    _feed(cal, art_a, "1h", labels, 5)
    path_a = cal._state("1h", art_a).path
    assert path_a.exists()

    cal_b = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art_b = FakeArt("model-b")
    out = cal_b.decorate("1h", art_b, _pf(START + 10 * H), START + 10 * H)
    path_b = cal_b._state("1h", art_b).path
    assert path_a.exists() and path_b.exists()
    assert path_a != path_b
    assert out["calibration"]["n_mature"] == 0


def test_key_mismatch_at_same_path_is_quarantined(tmp_path):
    art = FakeArt("same")
    fp = live.artifact_fingerprint(None, model_id="same", meta=art.meta)
    path = live.state_path(live.state_root_for(tmp_path), "ETHUSDT", "1h",
                           fp["model_hash"])
    path.parent.mkdir(parents=True, exist_ok=True)
    wrong = {"schema": live.LIVE_SCHEMA, "state_version": live.LIVE_STATE_VERSION,
             "key": {"symbol": "OTHER", "horizon": "1h",
                     "model_hash": fp["model_hash"], "calib_version": live.CALIB_VERSION}}
    path.write_text(json.dumps(wrong), encoding="utf-8")
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=FakeLabels())
    out = cal.decorate("1h", art, _pf(START), START)
    assert out["calibration"]["calib"] == "none"
    assert "snapshot_quarantined" in (out["calibration"]["alert"] or "")
    assert list(path.parent.glob("state.corrupt-*.json"))


def test_atomic_save_leaves_no_tmp(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    _feed(cal, art, "1h", labels, 3)
    path = cal._state("1h", art).path
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_missing_decision_or_sigma_degrades(tmp_path):
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=FakeLabels())
    pf = _pf(START)
    del pf["decision_ms"]
    out = cal.decorate("1h", FakeArt(), pf, START)
    assert out["calibration"]["reason"] == "missing_sigma_or_decision"
    assert out["calibrated_price_forecast"] == out["raw_price_forecast"]


# ----------------------------------------------------- quantiles / ordering

def test_q50_untouched_and_nested(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    _feed(cal, art, "1h", labels, 230)
    d = START + 229 * H
    labels.set("1h", d, 0.001)
    out = cal.decorate("1h", art, _pf(d), d + H + 60_000)
    raw, calq = out["raw_price_forecast"], out["calibrated_price_forecast"]
    assert out["median"] == raw["median"] == calq["median"]
    for block in (raw, calq):
        assert block["p10"] <= block["p25"] <= block["median"] <= block["p75"] <= block["p90"]
    assert out["p10"] == calq["p10"] and out["p90"] == calq["p90"]
    assert out["calibration"]["calib"] == "cqr"


def test_calibration_runs_before_news_overlay(tmp_path):
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: 1.0, label_source=labels)
    art = FakeArt()
    _feed(cal, art, "1h", labels, 230)
    d = START + 229 * H
    labels.set("1h", d, 0.001)
    pf = cal.decorate("1h", art, _pf(d), d + H + 60_000)
    now = d + H + 60_000
    item = {"available_at": now - 1000, "ts_published": now - 1000,
            "relevance": 0.9, "urgency": 0.9, "direction": 0.8,
            "title": "t", "source": "s", "content_hash": "h"}
    agg = overlay.aggregate([item], now, H)
    assert agg["gate"]["triggered"] is True
    ov = overlay.apply_overlay(pf, agg)
    assert ov["model"]["median"] == pytest.approx(
        pf["calibrated_price_forecast"]["median"], abs=1e-6)
    assert ov["model"]["p10"] == pytest.approx(
        pf["calibrated_price_forecast"]["p10"], abs=1e-6)
    assert ov["adjusted"]["median"] != pf["calibrated_price_forecast"]["median"]
    # The pre-news final block is the calibrated block; the post-news final is
    # the overlay output.  Neither overwrites the raw or calibrated blocks.
    assert pf["final_price_forecast"] == pf["calibrated_price_forecast"]
    assert pf["raw_price_forecast"] != ov["adjusted"]

    record = serve._click_record(
        tmp_path, now,
        {"items": [{"interval": "1h", "model_id": "m", "status": "ok",
                    "price_forecast": pf, "news_overlay": ov,
                    "final_price_forecast": ov["adjusted"]}]},
        {"fetched_at": now, "sources": {}, "score_counts": {}},
        {"status": "ok", "items": [], "aggregate": {}, "scorer_version": "test"})
    row = record["horizons"][0]
    assert row["raw_price_forecast"] == pf["raw_price_forecast"]
    assert row["calibrated_price_forecast"] == pf["calibrated_price_forecast"]
    assert row["final_price_forecast"] == ov["adjusted"]
    assert row["calibration"]["calib"] == "cqr"
    assert row["calibration"]["calib_version"] == live.CALIB_VERSION
    assert row["calibration"]["model_hash"]
    assert row["decision_ms"] == d


# --------------------------------------------------------- service plumbing

class _FakeService:
    def __init__(self, root, *, clock=None, **kwargs):
        self.root = root
        self.clock = clock
        self.calls = []

    def predict_all(self, intervals):
        items = []
        for iv in intervals:
            pf = {
                "interval": iv, "status": "ok", "ref_close": REF,
                # Display fields already point at the calibrated block.
                "median": REF, "p10": REF * 0.98, "p25": REF * 0.996,
                "p75": REF * 1.004, "p90": REF * 1.02,
                "width50_bps": 60.0, "width80_bps": 200.0,
                "decision_ms": START, "asof_utc8": "2026-11-11 00:00",
                "raw_price_forecast": {"median": REF, "p10": REF * 0.99,
                                       "p25": REF * 0.997, "p75": REF * 1.003,
                                       "p90": REF * 1.01},
                "calibrated_price_forecast": {"median": REF, "p10": REF * 0.98,
                                              "p25": REF * 0.996, "p75": REF * 1.004,
                                              "p90": REF * 1.02},
                "final_price_forecast": {"median": REF},
                "calibration": {"calib": "cqr" if iv in ("5m", "1h", "4h") else "none",
                                "calib_version": live.CALIB_VERSION,
                                "n_mature": 250, "model_hash": "mh",
                                "source": "bootstrap"},
            }
            items.append({"interval": iv, "status": "ok", "model_id": "m",
                          "price_forecast": pf})
        return {"debug": True, "items": items}

    def calibration_status(self):
        return {"enabled": True, "loaded": True, "marker": "fake"}


def test_app_post_get_and_health_plumbing(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "PredictService", _FakeService)
    monkeypatch.setattr(serve, "score_items", lambda *a, **k: {
        "counts": {}, "version": "test", "scores": {}})
    now_ms = START + 10 * H
    news_item = {"id": 1, "title": "ETH news", "source": "test",
                 "available_at": now_ms - 1000, "ts_published": now_ms - 1000,
                 "relevance": 0.9, "urgency": 0.9, "direction": 0.8,
                 "content_hash": "abc", "summary_zh": "摘要"}

    def snapshot(_now):
        return {"items": [dict(news_item)], "sources": {"test": 1},
                "status": "ok", "fetched_at": now_ms}

    app = serve.create_app(tmp_path, news_snapshot=snapshot,
                           clock=lambda: now_ms / 1000.0)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["calibration"]["marker"] == "fake"

        got = client.get("/api/predict").json()
        assert len(got["items"]) == 5

        post = client.post("/api/predict")
        assert post.status_code == 200
        payload = post.json()
        assert len(payload["items"]) == 5
        calib = {it["interval"]: it["price_forecast"]["calibration"]["calib"]
                 for it in payload["items"]}
        assert calib == {"5m": "cqr", "15m": "none", "30m": "none",
                         "1h": "cqr", "4h": "cqr"}
        for item in payload["items"]:
            assert item["news_overlay"] is not None
            assert item["final_price_forecast"] is not None
            assert (item["price_forecast"]["calibrated_price_forecast"]
                    == item["news_overlay"]["model"])
        log = (tmp_path / "logs" / "predict_clicks.jsonl").read_text(encoding="utf-8")
        record = json.loads(log.strip().splitlines()[-1])
        assert len(record["horizons"]) == 5
        for row in record["horizons"]:
            assert row["raw_price_forecast"] is not None
            assert row["calibrated_price_forecast"] is not None
            assert row["final_price_forecast"] is not None
            assert row["calibration"]["calib_version"] == live.CALIB_VERSION


def test_predict_service_attach_point(tmp_path, monkeypatch):
    """The real QuantilePredictService calls the calibrator for v3 artifacts."""
    from supere.train import predict as predict_mod
    from supere.train import quantile_v3 as v3

    class FakeV3Art:
        directory = None
        model_id = "v3-fake"
        meta = {"config": {"symbol": "ETHUSDT"}, "model_id": "v3-fake"}

    monkeypatch.setattr(v3, "load_v3", lambda root, interval: FakeV3Art())
    monkeypatch.setattr(v3, "predict_latest",
                        lambda root, interval, art, now: {
                            "interval": interval, "status": "ok",
                            "price_forecast": _pf(START),
                            "model_id": "v3-fake"})
    labels = FakeLabels()
    cal = live.LiveCalibrator(tmp_path, clock=lambda: START / 1000.0,
                              label_source=labels)
    svc = predict_mod.QuantilePredictService(tmp_path, clock=lambda: START / 1000.0,
                                             calibrator=cal)
    out = svc.predict("1h")
    assert out["price_forecast"]["calibration"]["policy"] == "cqr"
    assert out["price_forecast"]["raw_price_forecast"]["median"] == REF
    assert svc.calibration_status()["loaded"] is True
    assert svc.predict("15m")["price_forecast"]["calibration"]["calib"] == "none"
