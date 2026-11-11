"""Engineering tests for the v6 calibration state machine (stage 1).

These tests are pure synthetic: no database, no service, no shared data.
They cover the plan's engineering list: truncation, maturity delay (including a
4h horizon), batch == per-event, duplicate requests, missing labels,
checkpoint/restore, monotonicity with an unchanged q50, crossing fallback and
the <=20 ms calibration compute budget.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from supere.train import calib_v6 as c
from supere.train import v6proto

H = 3_600_000


def _frame(n: int = 800, step: int = H, sigma: float = 0.002, seed: int = 7,
           fold: int = 1, horizon_mult: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    decision = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * step
    sig = np.full(n, sigma)
    y = rng.normal(0.0, 1.0, n) * sig
    q = np.column_stack([-1.2816, -0.6745, 0.0, 0.6745, 1.2816]) * sig[:, None]
    return pd.DataFrame({
        "interval": "1h", "fold": np.full(n, fold, dtype=np.int64),
        "sample_pos": np.arange(n, dtype=np.int64),
        "decision_ms": decision,
        "horizon_end_ms": decision + step * horizon_mult,
        "y_logret": y, "ref_close": 2000.0, "sigma_model": sig,
        "q10": q[:, 0], "q25": q[:, 1], "q50": q[:, 2], "q75": q[:, 3], "q90": q[:, 4],
        "har_q10": q[:, 0], "har_q25": q[:, 1], "har_q50": q[:, 2],
        "har_q75": q[:, 3], "har_q90": q[:, 4],
        "rv_past": np.abs(rng.normal(0, 0.01, n)),
        "rv_future": np.abs(rng.normal(0, 0.01, n)),
        "model_hash": "m1", "feature_hash": "f1", "config_sha256": "c1",
    })


def _engine(candidate="cqr", *, min_samples=1, delay_ms=0, gamma=0.005,
            window_days=28.0, cap=1000, eps=1e-6):
    cfg = c.EngineConfig(candidate=candidate, window_days=window_days,
                         min_samples=min_samples, gamma=gamma, delay_ms=delay_ms)
    return c.CalibrationEngine(cfg, cap, eps)


def _qcols(cand):
    return [f"q{k}_{cand}" for k in ("10", "25", "50", "75", "90")]


def test_order_stat_finite_sample_and_clamp():
    vals = np.arange(10, dtype=float)  # 0..9
    out, flags = c.order_stats(vals, [0.5, 0.0, 1.0])
    # ceil((10+1)*0.5)=6 -> 6th smallest = 5
    assert out[0] == 5.0 and flags[0] == ""
    assert out[1] == 0.0 and flags[1] == "low"
    assert out[2] == 9.0 and flags[2] == "high"
    out2, flags2 = c.order_stats(np.array([]), [0.5])
    assert out2 == [0.0] and flags2 == ["empty"]


def test_q50_untouched_and_nested():
    df = _frame(600)
    out, _ = c.replay_interval(df, "1h", "cqr", window_days=28, min_samples=1)
    q = out[_qcols("cqr")].to_numpy()
    assert np.allclose(q[:, 2], df["q50"].to_numpy())
    assert np.all(np.diff(q, axis=1) >= 0)


def test_maturity_delay_blocks_learning():
    # horizon 1h, +60 s publication delay: at the second decision the first
    # label is not yet available, so the second row must be cold.
    df = _frame(12, step=H, seed=3)
    out, s = c.replay_interval(df, "1h", "cqr", window_days=28, min_samples=1,
                               delay_ms=60_000)
    assert out["ncal_cqr"].iloc[0] == 0
    assert out["ncal_cqr"].iloc[1] == 0
    assert out["ncal_cqr"].iloc[2] == 1
    # With no delay the first label is available at the next decision.
    out0, _ = c.replay_interval(df, "1h", "cqr", window_days=28, min_samples=1,
                                delay_ms=0)
    assert out0["ncal_cqr"].iloc[1] == 1


def test_4h_label_never_used_before_maturity():
    df = _frame(30, step=4 * H, horizon_mult=1, seed=4)
    out, _ = c.replay_interval(df, "4h", "cqr", window_days=90, min_samples=1)
    # Row i can only know labels of rows <= i-1, and row i-1 matures exactly at
    # decision_i; a 60 s delay pushes it one more row.
    out60, _ = c.replay_interval(df, "4h", "cqr", window_days=90, min_samples=1,
                                 delay_ms=60_000)
    assert out["ncal_cqr"].iloc[1] == 1
    assert out60["ncal_cqr"].iloc[2] == 1


def test_duplicate_request_does_not_learn():
    df = _frame(50, seed=5)
    eng = _engine(min_samples=1)
    row = df.iloc[0]
    pid = "p0"
    eng.set_label(pid, float(row.y_logret))
    _, d1 = eng.predict(int(row.decision_ms), row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float),
                        float(row.sigma_model), pid, int(row.horizon_end_ms))
    count = eng.band50.count
    # Advance past maturity so the item is consumed once.
    eng.advance(int(row.horizon_end_ms) + 1)
    after = eng.band50.count
    assert after == count + 1
    _, d2 = eng.predict(int(row.decision_ms), row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float),
                        float(row.sigma_model), pid, int(row.horizon_end_ms))
    assert d2["reason"] == "duplicate"
    assert eng.band50.count == after
    assert d1["reason"] == "cold"


def test_missing_label_is_skipped_and_counted():
    eng = _engine(min_samples=1)
    row = _frame(10, seed=6).iloc[0]
    q = row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float)
    eng.predict(int(row.decision_ms), q, float(row.sigma_model), "missing",
                int(row.horizon_end_ms))
    eng.advance(int(row.horizon_end_ms) + 1)
    assert eng.band50.count == 0
    assert eng.missing_labels == ["missing"]


def test_snapshot_restore_identical():
    df = _frame(300, seed=8)
    cfg = c.EngineConfig(candidate="cqr_aci", window_days=28, min_samples=1, gamma=0.005)
    eng = c.CalibrationEngine(cfg, 1000, 1e-6)
    dec = df.decision_ms.to_numpy()
    raw = df[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float)
    y = df.y_logret.to_numpy(float)
    sig = df.sigma_model.to_numpy(float)
    end = df.horizon_end_ms.to_numpy()
    outs = []
    for i in range(150):
        pid = f"p{i}"
        eng.set_label(pid, y[i])
        o, _ = eng.predict(int(dec[i]), raw[i], float(sig[i]), pid, int(end[i]))
        outs.append(o)
    snap = json.loads(json.dumps(eng.snapshot(), default=float))
    eng2 = c.CalibrationEngine.restore(cfg, 1000, 1e-6, snap)
    outs2 = []
    for i in range(150, 300):
        pid = f"p{i}"
        eng.set_label(pid, y[i])
        o, _ = eng.predict(int(dec[i]), raw[i], float(sig[i]), pid, int(end[i]))
        outs.append(o)
        eng2.set_label(pid, y[i])
        o2, _ = eng2.predict(int(dec[i]), raw[i], float(sig[i]), pid, int(end[i]))
        outs2.append(o2)
    assert np.allclose(np.asarray(outs[150:]), np.asarray(outs2))


def test_truncation_does_not_change_past():
    df = _frame(500, seed=9)
    full, _ = c.replay_interval(df, "1h", "cqr", window_days=28, min_samples=1)
    trunc, _ = c.replay_interval(df.iloc[:300].copy(), "1h", "cqr",
                                 window_days=28, min_samples=1)
    assert np.allclose(full[_qcols("cqr")].to_numpy()[:300],
                       trunc[_qcols("cqr")].to_numpy())


def test_batch_equals_event_by_event():
    df = _frame(200, seed=10)
    out, _ = c.replay_interval(df, "1h", "cqr", window_days=28, min_samples=1)
    eng = _engine(min_samples=1)
    manual = []
    for i, row in df.iterrows():
        pid = f"1h:1:{i}"
        eng.set_label(pid, float(row.y_logret))
        o, _ = eng.predict(int(row.decision_ms),
                           row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float),
                           float(row.sigma_model), pid, int(row.horizon_end_ms))
        manual.append(o)
    assert np.allclose(np.asarray(manual), out[_qcols("cqr")].to_numpy())


def test_crossing_falls_back_to_raw():
    eng = _engine(min_samples=1)
    # All past labels are far above q25, so the lower score quantile is very
    # negative; the q25 correction then pushes the bound above q50 -> crossing.
    q = np.array([-0.010, -0.001, 0.0, 0.001, 0.010])
    sigma = 0.002
    eng.band50.append(0, (-0.001 - 0.05) / sigma, (0.05 - 0.001) / sigma)
    eng.band50.append(1, (-0.001 - 0.05) / sigma, (0.05 - 0.001) / sigma)
    eng.band80.append(0, (-0.010 - 0.05) / sigma, (0.05 - 0.010) / sigma)
    eng.band80.append(1, (-0.010 - 0.05) / sigma, (0.05 - 0.010) / sigma)
    out, diag = eng.predict(10_000, q, sigma, "x", 20_000)
    assert diag["reason"] == "crossing"
    assert np.allclose(out, q)
    assert eng.fallbacks["crossing"] == 1


def test_calibration_compute_budget_p95_under_20ms():
    eng = _engine(min_samples=1, candidate="cqr_aci", cap=9000)
    rng = np.random.default_rng(11)
    n = 8064  # 28 days of 5m rows
    eng.band50.decision[:n] = np.arange(n)
    eng.band50.s_lo[:n] = rng.normal(0, 1, n)
    eng.band50.s_hi[:n] = rng.normal(0, 1, n)
    eng.band50.tail = n
    eng.band80.decision[:n] = np.arange(n)
    eng.band80.s_lo[:n] = rng.normal(0, 1, n)
    eng.band80.s_hi[:n] = rng.normal(0, 1, n)
    eng.band80.tail = n
    q = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
    times = []
    for _ in range(300):
        t = time.perf_counter()
        eng._calibrate(q, 0.002)
        times.append((time.perf_counter() - t) * 1000)
    p95 = float(np.quantile(times, 0.95))
    assert p95 < 20.0, f"p95={p95:.2f}ms"


def test_acceptance_rule_keeps_none_when_cost_too_high():
    summary = {
        "candidates": {
            "none": {"outer_confirmatory": {
                "pinball_combined_bps": 10.0, "IS50_bps": 100.0, "w50_bps": 50.0,
                "cov50": 0.50, "cov80": 0.80,
                "rolling_cov50_error": {"mean_abs": 0.01},
                "worst_valid_dev50_pp": 1.0}},
            "cqr": {"outer_confirmatory": {
                "pinball_combined_bps": 10.1, "IS50_bps": 101.0, "w50_bps": 52.0,
                "cov50": 0.50, "cov80": 0.80,
                "rolling_cov50_error": {"mean_abs": 0.002},
                "worst_valid_dev50_pp": 0.2}},
        }
    }
    res = c.compare_to_none(summary)["cqr"]
    assert res["coverage_ok"] and res["improve_ok"]
    assert not res["cost_ok"]
    assert res["keep_none"]


def test_http_read_contract_does_not_advance_state():
    df = _frame(40, seed=12)
    eng = _engine(min_samples=1)
    row = df.iloc[0]
    pid = "p0"
    eng.set_label(pid, float(row.y_logret))
    eng.predict(int(row.decision_ms),
                row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float),
                float(row.sigma_model), pid, int(row.horizon_end_ms))
    view = c.ReadOnlyView(eng, symbol="ETHUSDT", horizon="1h")
    before = view.fingerprint()
    snap_before = json.dumps(eng.snapshot(), sort_keys=True, default=float)
    responses = [view.get() for _ in range(100)]
    after = view.fingerprint()
    snap_after = json.dumps(eng.snapshot(), sort_keys=True, default=float)
    assert before == after
    assert snap_before == snap_after
    assert all(r["read_only"] for r in responses)
    assert all(r["pending"] == 1 for r in responses)


def test_out_of_order_predict_is_rejected():
    df = _frame(10, seed=13)
    eng = _engine(min_samples=1)
    row = df.iloc[5]
    q = row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float)
    eng.predict(int(row.decision_ms), q, float(row.sigma_model), "late", int(row.horizon_end_ms))
    earlier = df.iloc[4]
    with pytest.raises(ValueError):
        eng.predict(int(earlier.decision_ms),
                    earlier[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float),
                    float(earlier.sigma_model), "early", int(earlier.horizon_end_ms))


def test_atomic_snapshot_file_roundtrip(tmp_path):
    df = _frame(20, seed=14)
    eng = _engine(min_samples=1, candidate="cqr_aci")
    row = df.iloc[0]
    eng.set_label("p0", float(row.y_logret))
    eng.predict(int(row.decision_ms),
                row[["q10", "q25", "q50", "q75", "q90"]].to_numpy(float),
                float(row.sigma_model), "p0", int(row.horizon_end_ms))
    path = c.write_snapshot(tmp_path / "state.json", eng.snapshot())
    assert path.exists() and not (tmp_path / "state.json.tmp").exists()
    snap = c.read_snapshot(path)
    eng2 = c.CalibrationEngine.restore(eng.cfg, 1000, 1e-6, snap)
    assert eng2.snapshot() == eng.snapshot()
    assert eng2.last_decision_ms == eng.last_decision_ms


def test_protocol_frozen_once(tmp_path):
    doc = {"eval_protocol": "v6"}
    p = tmp_path / "proto.json"
    v6proto.write_protocol(doc, p)
    with pytest.raises(FileExistsError):
        v6proto.write_protocol(doc, p)
    assert json.loads(p.read_text(encoding="utf-8"))["eval_protocol"] == "v6"
