"""Alternative data for the point/interval forecast (plan 追加 2026-11-05 15:00).

Three optional feature groups, all causal:

* ``btc``: BTCUSDT UM klines aligned bar-for-bar to the ETH series. Returns,
  volatility, ETH/BTC relative strength, rolling correlation/beta. A row reads
  only bars whose exclusive close is at or before the same decision moment.
* ``dvol``: Deribit ETH DVOL hourly candles. A candle's close is only known at
  ``bucket_open + 1h``; ``asof`` uses that publication time, never the bucket
  open. Level, changes over 1h/4h/1d/7d, 30-day z-score and DVOL/realised ratio.
* ``macro``: US FOMC/CPI/NFP/PCE release times. The schedule is known in
  advance, so distance-to-next and event-in-horizon flags may look forward. The
  calendar CSV is built by ``scripts/fetch_macro_calendar.py``.

The CSV loaders are pure (no network), so tests and the read-only predict
service can use them without touching the database or an API key.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from supere.train import data, features

DVOL_RESOLUTION_MS = 3_600_000
DVOL_Z_WINDOW = 24 * 30  # 30 days of hourly values
YEAR_MS = 365 * 86_400_000
MACRO_TYPES: tuple[str, ...] = ("FOMC", "CPI", "NFP", "PCE")
MACRO_CODE = {name: i for i, name in enumerate(MACRO_TYPES)}
MACRO_WINDOWS: tuple[tuple[str, int], ...] = (
    ("5m", 5 * 60_000), ("15m", 15 * 60_000), ("30m", 30 * 60_000),
    ("1h", 3_600_000), ("4h", 4 * 3_600_000),
)

BTC_FEATURES: tuple[str, ...] = (
    "btc_ret_1", "btc_ret_3", "btc_ret_5", "btc_ret_20", "btc_rv_20", "btc_rv_50",
    "eth_btc_rs_1", "eth_btc_rs_3", "eth_btc_rs_5", "eth_btc_rs_20",
    "eth_btc_corr_20", "eth_btc_corr_60", "eth_btc_beta_20", "eth_btc_vol_ratio_20",
)
DVOL_FEATURES: tuple[str, ...] = (
    "dvol_level", "dvol_chg_1h", "dvol_chg_4h", "dvol_chg_1d", "dvol_chg_7d",
    "dvol_z_30d", "dvol_rv_ratio",
)
MACRO_FEATURES: tuple[str, ...] = (
    "macro_min_to_next", "macro_min_since_prev",
    *(f"macro_next_is_{name.lower()}" for name in MACRO_TYPES),
    *(f"macro_prev_is_{name.lower()}" for name in MACRO_TYPES),
    "macro_in_next_window", "macro_in_prev_window",
    *(f"macro_next_in_{label}" for label, _ in MACRO_WINDOWS),
    *(f"macro_prev_in_{label}" for label, _ in MACRO_WINDOWS),
)


def empty_columns(names: tuple[str, ...], n: int) -> dict[str, np.ndarray]:
    return {name: np.full(n, np.nan) for name in names}


@dataclass
class AltData:
    btc: data.Bars | None = None
    dvol_available: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dvol_close: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    dvol_z30: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    macro_time: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    macro_code: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def has_dvol(self) -> bool:
        return self.dvol_available.shape[0] > 0

    def has_macro(self) -> bool:
        return self.macro_time.shape[0] > 0


def load_dvol_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(available_ms, close, z30)`` sorted by publication time. NaN z before a full window."""
    if not Path(path).exists():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    stamps: list[int] = []
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                ts = int(row["ts"])
                close = float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(close) and close > 0:
                stamps.append(ts)
                values.append(close)
    order = np.argsort(np.asarray(stamps, dtype=np.int64), kind="stable")
    ts = np.asarray(stamps, dtype=np.int64)[order]
    close = np.asarray(values, dtype=np.float64)[order]
    available = ts + DVOL_RESOLUTION_MS
    level = np.log(close)
    z = np.full(close.shape[0], np.nan)
    if close.shape[0] >= DVOL_Z_WINDOW:
        mean = features._roll(level, DVOL_Z_WINDOW, np.mean)
        std = features._roll(level, DVOL_Z_WINDOW, np.std)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (level - mean) / std
        z = np.where(std > 0, z, np.nan)
    return available, close, z


def load_macro_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(event_ms, type_code)`` sorted, unknown types dropped."""
    if not Path(path).exists():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    times: list[int] = []
    codes: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = (row.get("type") or "").strip().upper()
            if kind not in MACRO_CODE:
                continue
            try:
                times.append(int(row["event_time"]))
            except (KeyError, TypeError, ValueError):
                continue
            codes.append(MACRO_CODE[kind])
    order = np.argsort(np.asarray(times, dtype=np.int64), kind="stable")
    return (np.asarray(times, dtype=np.int64)[order], np.asarray(codes, dtype=np.int64)[order])


def load_alt(data_root: Path, groups: set[str], btc: data.Bars | None = None,
             dvol_file: str | None = None, macro_file: str | None = None) -> AltData:
    root = Path(data_root)
    dvol_path = Path(dvol_file) if dvol_file else root / "dvol_ETH_1h.csv"
    macro_path = Path(macro_file) if macro_file else root / "macro_calendar.csv"
    out = AltData(btc=btc if "btc" in groups else None)
    if "dvol" in groups:
        out.dvol_available, out.dvol_close, out.dvol_z30 = load_dvol_csv(dvol_path)
    if "macro" in groups:
        out.macro_time, out.macro_code = load_macro_csv(macro_path)
    return out


# ---------------------------------------------------------------- helpers

def _asof(sorted_ms: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the last value available at each query time; -1 when none."""
    if sorted_ms.shape[0] == 0:
        return np.full(query.shape[0], -1, dtype=np.int64)
    return np.searchsorted(sorted_ms, query, side="right") - 1


def _lag(values: np.ndarray, k: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan)
    if values.shape[0] > k:
        out[k:] = values[:-k]
    return out


def _align_close(bars: data.Bars, other: data.Bars) -> np.ndarray:
    """Other's close at the same open time; NaN when the bar is missing."""
    out = np.full(bars.n, np.nan)
    if other is None or other.n == 0:
        return out
    idx = np.searchsorted(other.open_time, bars.open_time)
    ok = (idx >= 0) & (idx < other.n)
    hit = np.zeros(bars.n, dtype=bool)
    hit[ok] = other.open_time[idx[ok]] == bars.open_time[ok]
    if hit.any():
        out[hit] = other.c[idx[hit]]
    return out


def _roll_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < w:
        return out
    pair = np.column_stack([x, y])
    windows = np.lib.stride_tricks.sliding_window_view(pair, w, axis=0)  # (n-w+1, 2, w)
    wx = windows[:, 0, :]
    wy = windows[:, 1, :]
    mx = wx.mean(axis=1)
    my = wy.mean(axis=1)
    cov = ((wx - mx[:, None]) * (wy - my[:, None])).mean(axis=1)
    sx = wx.std(axis=1)
    sy = wy.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (sx * sy)
    out[w - 1 :] = np.where((sx > 0) & (sy > 0), corr, np.nan)
    return out


def _roll_beta(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < w:
        return out
    pair = np.column_stack([x, y])
    windows = np.lib.stride_tricks.sliding_window_view(pair, w, axis=0)
    wx = windows[:, 0, :]
    wy = windows[:, 1, :]
    mx = wx.mean(axis=1)
    my = wy.mean(axis=1)
    cov = ((wx - mx[:, None]) * (wy - my[:, None])).mean(axis=1)
    var = ((wy - my[:, None]) ** 2).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov / var
    out[w - 1 :] = np.where(var > 0, beta, np.nan)
    return out


# ---------------------------------------------------------------- groups

def btc_columns(bars: data.Bars, btc: data.Bars | None) -> dict[str, np.ndarray]:
    """BTC context. Missing BTC bars become NaN, never silently zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        eth_close = bars.c
        btc_close = _align_close(bars, btc)
        log_e = np.log(eth_close)
        log_b = np.log(btc_close)
        r_e = np.full(bars.n, np.nan)
        r_b = np.full(bars.n, np.nan)
        r_e[1:] = np.diff(log_e)
        r_b[1:] = np.diff(log_b)
        cols: dict[str, np.ndarray] = {}
        for k in (1, 3, 5, 20):
            e_ret = log_e - _lag(log_e, k)
            b_ret = log_b - _lag(log_b, k)
            cols[f"btc_ret_{k}"] = b_ret
            cols[f"eth_btc_rs_{k}"] = e_ret - b_ret
        cols["btc_rv_20"] = features._roll(r_b, 20, np.std)
        cols["btc_rv_50"] = features._roll(r_b, 50, np.std)
        cols["eth_btc_corr_20"] = _roll_corr(r_e, r_b, 20)
        cols["eth_btc_corr_60"] = _roll_corr(r_e, r_b, 60)
        cols["eth_btc_beta_20"] = _roll_beta(r_e, r_b, 20)
        eth_rv20 = features._roll(r_e, 20, np.std)
        cols["eth_btc_vol_ratio_20"] = np.log(eth_rv20 / cols["btc_rv_20"])
    return cols


def dvol_columns(decision_ms: np.ndarray, interval_ms: int, sig: np.ndarray,
                 alt: AltData) -> dict[str, np.ndarray]:
    n = decision_ms.shape[0]
    if not alt.has_dvol():
        return empty_columns(DVOL_FEATURES, n)
    avail = alt.dvol_available
    values = alt.dvol_close
    j = _asof(avail, decision_ms)
    ok = j >= 0
    level = np.full(n, np.nan)
    level[ok] = np.log(values[j[ok]] / 100.0)
    cols: dict[str, np.ndarray] = {"dvol_level": level}
    for label, window_ms in (("1h", 3_600_000), ("4h", 4 * 3_600_000),
                             ("1d", 86_400_000), ("7d", 7 * 86_400_000)):
        j0 = _asof(avail, decision_ms - window_ms)
        prev = np.full(n, np.nan)
        prev_ok = j0 >= 0
        prev[prev_ok] = np.log(values[j0[prev_ok]] / 100.0)
        cols[f"dvol_chg_{label}"] = level - prev
    z = np.full(n, np.nan)
    z[ok] = alt.dvol_z30[j[ok]]
    cols["dvol_z_30d"] = z
    # DVOL is an annualised percentage; compare it with annualised realised vol.
    rv_annual = sig * np.sqrt(YEAR_MS / float(interval_ms))
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["dvol_rv_ratio"] = level - np.log(rv_annual)
    return cols


def macro_columns(decision_ms: np.ndarray, horizon_ms: int, alt: AltData) -> dict[str, np.ndarray]:
    n = decision_ms.shape[0]
    if not alt.has_macro():
        return empty_columns(MACRO_FEATURES, n)
    events = alt.macro_time
    codes = alt.macro_code
    nxt = np.searchsorted(events, decision_ms, side="right")
    prv = nxt - 1
    has_next = nxt < events.shape[0]
    has_prev = prv >= 0

    def minutes_to_next() -> np.ndarray:
        out = np.full(n, np.nan)
        out[has_next] = (events[nxt[has_next]] - decision_ms[has_next]) / 60_000.0
        return np.log1p(np.maximum(out, 0.0))

    def minutes_since_prev() -> np.ndarray:
        out = np.full(n, np.nan)
        out[has_prev] = (decision_ms[has_prev] - events[prv[has_prev]]) / 60_000.0
        return np.log1p(np.maximum(out, 0.0))

    cols: dict[str, np.ndarray] = {
        "macro_min_to_next": minutes_to_next(),
        "macro_min_since_prev": minutes_since_prev(),
    }
    next_at = np.where(has_next, nxt, 0)
    prev_at = np.where(has_prev, prv, 0)
    for code, name in enumerate(MACRO_TYPES):
        cols[f"macro_next_is_{name.lower()}"] = (has_next & (codes[next_at] == code)).astype(np.float64)
        cols[f"macro_prev_is_{name.lower()}"] = (has_prev & (codes[prev_at] == code)).astype(np.float64)
    soon = has_next & (events[next_at] <= decision_ms + horizon_ms)
    recent = has_prev & (events[prev_at] >= decision_ms - horizon_ms)
    cols["macro_in_next_window"] = soon.astype(np.float64)
    cols["macro_in_prev_window"] = recent.astype(np.float64)
    for label, window_ms in MACRO_WINDOWS:
        nxt_flag = has_next & (events[next_at] <= decision_ms + window_ms)
        prev_flag = has_prev & (events[prev_at] >= decision_ms - window_ms)
        cols[f"macro_next_in_{label}"] = nxt_flag.astype(np.float64)
        cols[f"macro_prev_in_{label}"] = prev_flag.astype(np.float64)
    return cols


def build_columns(bars: data.Bars, decision_ms: np.ndarray, interval_ms: int, sig: np.ndarray,
                  alt: AltData | None, groups: set[str]) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {}
    if alt is None:
        alt = AltData()
    if "btc" in groups:
        cols.update(btc_columns(bars, alt.btc))
    if "dvol" in groups:
        cols.update(dvol_columns(decision_ms, interval_ms, sig, alt))
    if "macro" in groups:
        cols.update(macro_columns(decision_ms, interval_ms, alt))
    return cols
