"""BTC / DVOL / macro-calendar features: alignment, as-of rules, no lookahead, serving contract."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import numpy as np

from synthetic import START_MS, build_synthetic_db

from supere.train import altdata, data
from supere.train.quantile import (QConfig, QuantileError, default_config, feature_groups,
                                   fit_block, load_samples, save_artifacts)

STEP_15M = 900_000
N = 4000


def test_feature_groups_parsing_and_validation():
    assert feature_groups(QConfig(interval="1h")) == set()
    assert feature_groups(QConfig(interval="1h", feature_groups="btc, dvol")) == {"btc", "dvol"}
    assert feature_groups(default_config("1h", {"feature_groups": "all"})) == {"btc", "dvol", "macro"}
    try:
        default_config("1h", {"feature_groups": "btc,foo"})
    except QuantileError:
        pass
    else:
        raise AssertionError("unknown feature group should be rejected")




def _bars(interval: str, closes: np.ndarray, start_ms: int, step_ms: int) -> data.Bars:
    n = closes.shape[0]
    open_time = start_ms + np.arange(n, dtype=np.int64) * step_ms
    return data.Bars(
        interval=interval, open_time=open_time, o=closes.copy(), h=closes.copy(),
        l=closes.copy(), c=closes.copy(), volume=np.ones(n), quote_volume=np.ones(n),
        trades=np.ones(n), taker_buy_base=np.ones(n) * 0.5,
    )


def _eth_btc(n: int = 300, seed: int = 3) -> tuple[data.Bars, data.Bars]:
    rng = np.random.default_rng(seed)
    eth = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    btc = 30000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0015, n)))
    return _bars("5m", eth, START_MS, 300_000), _bars("5m", btc, START_MS, 300_000)


def _write_dvol(path: Path, rows: list[tuple[int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "open", "high", "low", "close"])
        for ts, close in rows:
            writer.writerow([ts, close, close, close, close])


def _write_macro(path: Path, rows: list[tuple[int, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_time", "type", "title", "importance", "source"])
        for ts, kind in rows:
            writer.writerow([ts, kind, kind, 1, "test"])


def test_btc_returns_match_close_ratios_and_are_causal():
    eth, btc = _eth_btc()
    cols = altdata.btc_columns(eth, btc)
    i = 120
    assert np.isclose(cols["btc_ret_1"][i], np.log(btc.c[i] / btc.c[i - 1]))
    assert np.isclose(cols["btc_ret_5"][i], np.log(btc.c[i] / btc.c[i - 5]))
    log_e = np.log(eth.c)
    log_b = np.log(btc.c)
    assert np.isclose(cols["eth_btc_rs_5"][i], (log_e[i] - log_e[i - 5]) - (log_b[i] - log_b[i - 5]))
    # Truncating the input to bars 0..i must not change row i.
    short = altdata.btc_columns(eth.slice_to(i + 1), btc.slice_to(i + 1))
    for name in altdata.BTC_FEATURES:
        a, b = cols[name][i], short[name][i]
        assert np.isfinite(a) and np.isfinite(b)
        assert np.isclose(a, b), name


def test_btc_missing_bars_are_nan_not_zero():
    eth, _ = _eth_btc()
    cols = altdata.btc_columns(eth, None)
    assert np.all(np.isnan(cols["btc_ret_1"]))
    assert np.all(np.isnan(cols["eth_btc_rs_5"]))
    assert np.all(np.isnan(cols["eth_btc_corr_20"]))


def test_dvol_csv_is_asof_on_bucket_close(tmp_path: Path):
    path = tmp_path / "dvol.csv"
    t0 = 1_600_000_000_000
    _write_dvol(path, [(t0, 50.0), (t0 + 3_600_000, 60.0)])
    available, close, z = altdata.load_dvol_csv(path)
    assert available.tolist() == [t0 + 3_600_000, t0 + 7_200_000]
    assert close.tolist() == [50.0, 60.0]
    assert np.all(np.isnan(z))  # not enough history for the 30d window

    alt = altdata.AltData(dvol_available=available, dvol_close=close, dvol_z30=z)
    sig = np.full(3, 0.01)
    # One millisecond before the first candle closes the value is unknown.
    cols = altdata.dvol_columns(np.array([t0 + 3_600_000 - 1, t0 + 3_600_000, t0 + 7_200_000]), STEP_15M, sig, alt)
    assert np.isnan(cols["dvol_level"][0])
    assert np.isclose(cols["dvol_level"][1], np.log(0.5))
    assert np.isclose(cols["dvol_chg_1h"][2], np.log(60.0 / 50.0))
    assert "dvol_rv_ratio" in cols


def test_macro_distances_flags_and_unknown_types(tmp_path: Path):
    path = tmp_path / "macro.csv"
    t0, t1 = 1_600_000_000_000, 1_600_000_000_000 + 3_600_000
    _write_macro(path, [(t1, "FOMC"), (t0, "CPI"), (t0, "NOT_A_TYPE")])
    times, codes = altdata.load_macro_csv(path)
    assert times.tolist() == [t0, t1]
    assert codes.tolist() == [altdata.MACRO_CODE["CPI"], altdata.MACRO_CODE["FOMC"]]

    alt = altdata.AltData(macro_time=times, macro_code=codes)
    mid = t0 + 1_800_000
    cols = altdata.macro_columns(np.array([mid]), 3_600_000, alt)
    assert np.isclose(cols["macro_min_to_next"][0], np.log1p(30.0))
    assert np.isclose(cols["macro_min_since_prev"][0], np.log1p(30.0))
    assert cols["macro_next_is_fomc"][0] == 1.0
    assert cols["macro_prev_is_cpi"][0] == 1.0
    assert cols["macro_in_next_window"][0] == 1.0
    assert cols["macro_in_prev_window"][0] == 1.0
    assert cols["macro_next_in_5m"][0] == 0.0  # 30 minutes away
    assert cols["macro_next_in_30m"][0] == 1.0


def test_missing_alt_files_keep_named_all_nan_columns(tmp_path: Path):
    eth, _ = _eth_btc(n=80)
    empty = altdata.build_columns(eth, eth.close_time, eth.interval_ms, np.full(80, 0.01),
                                  altdata.AltData(), {"dvol", "macro"})
    assert list(empty) == list(altdata.DVOL_FEATURES + altdata.MACRO_FEATURES)
    assert all(np.all(np.isnan(v)) for v in empty.values())

    loaded = altdata.load_alt(tmp_path, {"dvol", "macro"})
    same = altdata.build_columns(eth, eth.close_time, eth.interval_ms, np.full(80, 0.01),
                                 loaded, {"dvol", "macro"})
    assert list(same) == list(empty)


def _copy_markets(db_path: Path, markets: tuple[str, ...] = ("um_mark", "um_index", "um_premium")) -> None:
    """Duplicate the ETH klines into the derived-price markets the feature code reads."""
    conn = sqlite3.connect(db_path)
    try:
        for market in markets:
            conn.execute(
                "INSERT INTO klines SELECT ?, symbol, interval, open_time, close_time, o, h, l, c, "
                "volume, quote_volume, trades, taker_buy_base, taker_buy_quote, source, ingested_at, "
                "ingested_local FROM klines WHERE market = 'um' AND symbol = 'ETHUSDT' AND interval = '15m'",
                (market,),
            )
        conn.commit()
    finally:
        conn.close()


def _build_data_root(tmp_path: Path) -> Path:
    build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval="15m")
    # Extra intervals/symbols: no funding/metrics rows, so their primary keys stay unique.
    build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval="1h", symbol="ETHUSDT",
                       with_funding=False, with_metrics=False)
    build_synthetic_db(tmp_path / "superE.sqlite", n=N, interval="15m", symbol="BTCUSDT",
                       with_funding=False, with_metrics=False, seed=11)
    _copy_markets(tmp_path / "superE.sqlite")
    step = STEP_15M
    _write_dvol(tmp_path / "dvol_ETH_1h.csv",
                [(START_MS + k * 3_600_000, 40.0 + (k % 24)) for k in range(N * step // 3_600_000)])
    _write_macro(tmp_path / "macro_calendar.csv",
                 [(START_MS + k * 86_400_000, kind)
                  for k, kind in enumerate(["FOMC", "CPI", "NFP", "PCE"] * 8)])
    return tmp_path


def test_predict_service_contract_with_alt_groups(tmp_path: Path):
    from supere.train.predict import QuantilePredictService

    root = _build_data_root(tmp_path)
    cfg = QConfig(interval="15m", snapshot_ms=None, feature_groups="all",
                  n_estimators=5, n_jobs=1, n_folds=2)
    S = load_samples(cfg, root / "superE.sqlite")
    assert any(n.startswith("btc_") for n in S.names)
    assert any(n.startswith("dvol_") for n in S.names)
    assert any(n.startswith("macro_") for n in S.names)
    fit = fit_block(S, np.arange(S.m, dtype=np.int64), cfg)
    save_artifacts(root, S, cfg, fit, S.names, {})

    now_ms = int(S.decision_ms[-1]) + STEP_15M
    service = QuantilePredictService(root, clock=lambda: now_ms / 1000.0)
    out = service.predict("15m")
    assert out["status"] == "ok", out
    pf = out["price_forecast"]
    for key in ("interval", "asof_utc8", "ref_close", "median", "p10", "p25", "p75", "p90",
                "width50_bps", "width80_bps", "coverage_ok", "status", "explain", "top_features"):
        assert key in pf, key
    assert pf["p10"] <= pf["p25"] <= pf["median"] <= pf["p75"] <= pf["p90"]
    assert out["model_id"]
