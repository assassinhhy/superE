"""Debug model artifacts and the latest-bar predictor: save/load, no-leak, empty states, routes."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
from synthetic import START_MS, build_synthetic_db

from supere.intervals import INTERVAL_MS
from supere.train import artifacts, config as tcfg
from supere.train.__main__ import main
from supere.train.predict import PredictService, direction, latest_probs, closed_count
from supere.train.runner import run
from supere.train import data, features

INTERVAL = "15m"
STEP = INTERVAL_MS[INTERVAL]
N = 600
FAST = ["--n-estimators", "30", "--n-jobs", "1"]


def _quiet_logger() -> logging.Logger:
    log = logging.getLogger("test.predict")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


def _train(tmp_path: Path, db: Path):
    cfg = tcfg.build(interval=INTERVAL, file_values={},
                     overrides={"model": {"n_estimators": 30, "n_jobs": 1}})
    return run(cfg, db, _quiet_logger(), save_root=tmp_path)


def _service(tmp_path: Path, now_ms: int) -> PredictService:
    return PredictService(tmp_path, clock=lambda: now_ms / 1000.0)


def test_run_saves_model_folder_and_it_loads(tmp_path: Path):
    build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    rc = main(["--config", str(Path(__file__).resolve().parents[1] / "config.toml"),
               "--train-config", str(Path(__file__).resolve().parents[1] / "train.toml"),
               "--data-dir", str(tmp_path), "--interval", INTERVAL, *FAST])
    assert rc == 0

    folder = tmp_path / "models" / INTERVAL
    for name in (artifacts.MODEL_FILE, artifacts.FEATURES_FILE, artifacts.META_FILE):
        assert (folder / name).is_file(), name
    assert not list(folder.glob("*.tmp"))

    art = artifacts.load(tmp_path, INTERVAL)
    assert art is not None
    meta = art.meta
    assert meta["debug"] is True
    assert meta["interval"] == INTERVAL
    assert meta["horizon"] == 4 and meta["threshold"] == 0.40
    assert meta["trained_at_utc8"]
    assert meta["oos"]["samples"] > 0
    assert art.booster.num_feature() == len(art.names) == meta["n_features"]
    json.loads((folder / artifacts.META_FILE).read_text(encoding="utf-8"))  # strict JSON


def test_no_arg_run_still_saves_and_no_save_flag_skips(tmp_path: Path):
    build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    repo = Path(__file__).resolve().parents[1]
    base = ["--config", str(repo / "config.toml"), "--train-config", str(repo / "train.toml"),
            "--data-dir", str(tmp_path), "--interval", INTERVAL, *FAST]
    assert main(base + ["--no-save"]) == 0
    assert artifacts.load(tmp_path, INTERVAL) is None


def test_missing_model_and_bad_interval_are_empty_states(tmp_path: Path):
    svc = _service(tmp_path, now_ms=START_MS)
    out = svc.predict("15m")
    assert out["status"] == "no_model" and out["direction"] is None and out["debug"] is True
    bad = svc.predict("1d")
    assert bad["status"] == "bad_interval"


def test_partial_model_folder_is_reported_not_loaded(tmp_path: Path):
    folder = artifacts.model_dir(tmp_path, INTERVAL)
    folder.mkdir(parents=True)
    (folder / artifacts.META_FILE).write_text("{}", encoding="utf-8")
    out = _service(tmp_path, now_ms=START_MS).predict(INTERVAL)
    assert out["status"] == "bad_model"
    assert "缺少" in out["message"]


def test_direction_rule_matches_backtest_signal():
    assert direction({"up": 0.5, "flat": 0.2, "down": 0.3}, 0.40) == "up"
    assert direction({"up": 0.1, "flat": 0.2, "down": 0.7}, 0.40) == "down"
    assert direction({"up": 0.35, "flat": 0.40, "down": 0.25}, 0.40) == "flat"
    # Both sides above the bar: the larger one wins, as in signal_dirs.
    assert direction({"up": 0.45, "flat": 0.0, "down": 0.55}, 0.40) == "down"


def _probs_at(tmp_path: Path, db: Path, now_ms: int) -> dict:
    out = _service(tmp_path, now_ms).predict(INTERVAL)
    assert out["status"] == "ok", out
    return out["probs"]


def test_prediction_ignores_forming_and_future_bars(tmp_path: Path):
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)

    cut = 480  # bars [0, cut) are closed at `now`; bar `cut` has just opened
    now_just_opened = START_MS + cut * STEP
    now_mid_forming = now_just_opened + STEP // 2
    reference = _probs_at(tmp_path, db, now_just_opened)
    assert _probs_at(tmp_path, db, now_mid_forming) == reference

    # Overwrite every bar from `cut` onwards with absurd prices. Nothing before
    # `now` may change, so the output must stay identical.
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE klines SET o='1000000', h='1000000', l='1000000', c='1000000', volume='1' "
            "WHERE market='um' AND symbol='ETHUSDT' AND interval=? AND open_time >= ?",
            (INTERVAL, now_just_opened),
        )
        conn.commit()
    finally:
        conn.close()
    assert _probs_at(tmp_path, db, now_mid_forming) == reference


def test_latest_probs_uses_only_closed_bars(tmp_path: Path):
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)
    art = artifacts.load(tmp_path, INTERVAL)
    conn = data.open_readonly(db)
    try:
        bars = data.load_bars(conn, "um", "ETHUSDT", INTERVAL)
    finally:
        conn.close()
    now = int(bars.open_time[300])  # bar 300 has just opened: bars 0..299 are closed
    assert closed_count(bars.close_time, now) == 300
    got = latest_probs(art, bars, data.Funding.empty(), None, now)
    assert not isinstance(got, str)
    assert got.last == 299
    assert got.decision_ms == now


def test_routes_return_debug_banner_and_batch_shape(tmp_path: Path):
    from fastapi.testclient import TestClient

    from supere.train.serve import create_app

    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/api/health").json()
        assert health["ok"] is True and health["debug"] is True

        one = client.get("/api/predict", params={"interval": "15m"}).json()
        assert one["interval"] == "15m" and one["status"] == "no_model" and one["debug"] is True

        batch = client.get("/api/predict").json()
        assert [item["interval"] for item in batch["items"]] == list(tcfg.PRODUCT_INTERVALS)
        assert batch["debug"] is True

        bad = client.get("/api/predict", params={"interval": "1d"})
        assert bad.status_code == 400
        assert bad.json()["status"] == "bad_interval"


def test_product_intervals_are_trainable_and_labelled():
    for interval in tcfg.PRODUCT_INTERVALS:
        assert interval in tcfg.TRAINABLE_INTERVALS
        assert interval in tcfg.DEFAULT_LABEL
    assert tcfg.PRODUCT_INTERVALS == ("5m", "15m", "30m", "1h", "4h")


def test_strict_json_helper_turns_nan_into_null():
    out = artifacts.strict({"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3), "d": [np.float32(1.5)]})
    assert out == {"a": None, "b": None, "c": 3, "d": [1.5]}
    json.dumps(out, allow_nan=False)


def test_interrupted_save_is_refused_not_served(tmp_path: Path):
    """A new features.json or model.txt without a matching meta.json must not load silently."""
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)
    folder = artifacts.model_dir(tmp_path, INTERVAL)

    names = json.loads((folder / artifacts.FEATURES_FILE).read_text(encoding="utf-8"))
    swapped = [names[1], names[0], *names[2:]]  # same count, different order
    (folder / artifacts.FEATURES_FILE).write_text(json.dumps(swapped), encoding="utf-8")

    try:
        artifacts.load(tmp_path, INTERVAL)
    except artifacts.ArtifactError as exc:
        assert "保存到一半" in str(exc)
    else:  # pragma: no cover - the test is meant to fail here
        raise AssertionError("mismatched features.json was loaded")
    out = _service(tmp_path, now_ms=START_MS + N * STEP).predict(INTERVAL)
    assert out["status"] == "bad_model" and out["direction"] is None


def test_malformed_meta_gives_bad_model_status(tmp_path: Path):
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)
    folder = artifacts.model_dir(tmp_path, INTERVAL)
    for body in ("null", "[]", '{"schema_version": 1, "interval": "15m"}'):
        (folder / artifacts.META_FILE).write_text(body, encoding="utf-8")
        out = _service(tmp_path, now_ms=START_MS + N * STEP).predict(INTERVAL)
        assert out["status"] == "bad_model", (body, out)
        assert out["direction"] is None


def test_top_features_are_shap_contributions_that_sum_to_raw_score(tmp_path: Path):
    """The explanation must be the booster's own decomposition, not a made-up ranking."""
    from supere.train.predict import CLASS_ORDER, EXPLAIN_TOP, explain_row

    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)
    art = artifacts.load(tmp_path, INTERVAL)
    conn = data.open_readonly(db)
    try:
        bars = data.load_bars(conn, "um", "ETHUSDT", INTERVAL)
    finally:
        conn.close()
    now = int(bars.open_time[400])
    got = latest_probs(art, bars, data.Funding.empty(), None, now)
    assert not isinstance(got, str)

    # Full decomposition for the explained class: segments sum to the raw score.
    names, X = features.build(bars.slice_to(400), data.Funding.empty() if art.meta["use_funding"] else None, None)
    row = X[399]
    full = np.asarray(art.booster.predict(row.reshape(1, -1), pred_contrib=True)).reshape(-1)
    raw = np.asarray(art.booster.predict(row.reshape(1, -1), raw_score=True)).reshape(-1)
    n = len(art.names)
    assert full.shape[0] == len(CLASS_ORDER) * (n + 1)
    for k in range(len(CLASS_ORDER)):
        assert np.isclose(full[k * (n + 1): (k + 1) * (n + 1)].sum(), raw[k])

    assert got.explain_for == max(CLASS_ORDER, key=lambda c: got.probs[c])
    assert len(got.top_features) == EXPLAIN_TOP
    mags = [abs(f["contribution"]) for f in got.top_features]
    assert mags == sorted(mags, reverse=True)
    k = CLASS_ORDER.index(got.explain_for)
    for f in got.top_features:
        i = art.names.index(f["name"])
        assert np.isclose(f["contribution"], round(full[k * (n + 1) + i], 6))
        assert f["value"] == float(row[i]) or (f["value"] is None and not np.isfinite(row[i]))

    # The same explanation reaches the JSON the UI reads.
    out = _service(tmp_path, now_ms=now).predict(INTERVAL)
    assert out["status"] == "ok"
    assert out["explain_for"] == got.explain_for
    assert len(out["top_features"]) == EXPLAIN_TOP
    assert out["explain_for"] == max(CLASS_ORDER, key=lambda c: out["probs"][c])
    assert all(f["name"] in art.names for f in out["top_features"])
    json.dumps(out, allow_nan=False)
    assert explain_row(art.booster, art.names, row, k)[0]["name"] == got.top_features[0]["name"]


def test_trade_levels_use_the_label_barrier_width():
    from supere.train.predict import trade_levels

    up, why = trade_levels("up", ref=100.0, atr=2.0, k_atr=1.0)
    assert why is None
    assert up["side"] == "long" and up["entry"] == 100.0
    assert up["stop"] == 98.0 and up["targets"][0] == 102.0  # 1R = the label's upper barrier
    assert up["targets"][1] == 104.0 and up["backtested_targets"] == [True, False]
    assert abs(up["stop_pct"] + 0.02) < 1e-12

    dn, _ = trade_levels("down", ref=100.0, atr=2.0, k_atr=1.5)
    assert dn["side"] == "short" and dn["stop"] == 103.0 and dn["targets"][0] == 97.0

    flat, reason = trade_levels("flat", ref=100.0, atr=2.0, k_atr=1.0)
    assert flat is None and "观望" in reason
    none_atr, reason = trade_levels("up", ref=100.0, atr=float("nan"), k_atr=1.0)
    assert none_atr is None and reason


def test_served_prediction_carries_levels_consistent_with_direction(tmp_path: Path):
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)
    out = _service(tmp_path, START_MS + 480 * STEP + STEP // 2).predict(INTERVAL)
    assert out["status"] == "ok"
    if out["direction"] == "flat":
        assert out["levels"] is None and out["levels_reason"]
    else:
        lv = out["levels"]
        assert lv is not None
        assert (lv["stop"] < lv["entry"] < lv["targets"][0]) == (out["direction"] == "up")
        risk = abs(lv["entry"] - lv["stop"])
        assert abs(abs(lv["targets"][0] - lv["entry"]) - risk) < 1e-6  # 1R on each side


def test_served_prediction_context_brackets_close_and_funding_is_before_decision(tmp_path: Path):
    from supere.train.predict import CONTEXT_WINDOW, _context

    db = build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval=INTERVAL)
    _train(tmp_path, db)
    out = _service(tmp_path, START_MS + 480 * STEP + STEP // 2).predict(INTERVAL)
    assert out["status"] == "ok"
    ctx = out["context"]
    assert ctx["window_bars"] == CONTEXT_WINDOW
    assert ctx["low"] <= ctx["ref_close"] <= ctx["high"]
    assert ctx["funding"] is None or ctx["funding"]["rate"] is not None

    # Funding settled at or after the decision moment must never be picked.
    bars = data.load_bars(data.open_readonly(db), "um", "ETHUSDT", INTERVAL, 1500)
    funding = data.Funding(time=np.array([10, 20, 30], dtype=np.int64),
                           rate=np.array([0.1, 0.2, 0.3]))
    picked = _context(bars, funding, last=50, decision_ms=25)
    assert picked["funding"]["rate"] == 0.2
    assert _context(bars, funding, last=50, decision_ms=10)["funding"] is None
    assert _context(bars, data.Funding.empty(), last=50, decision_ms=25)["funding"] is None
