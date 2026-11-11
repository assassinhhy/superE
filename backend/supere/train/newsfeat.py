"""Causal news features (plan v3, stage 5).

A news item is visible only at ``available_at`` (conservatively the later of
publication and collector/GDELT-seen time).  Every feature for a decision at
``t`` uses items with ``available_at <= t`` only.  Before the first available
item the features are NaN, not zero: "no coverage" is not the same as "no news".

The module is read-only; the collector/scorer writes the table elsewhere.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
EPS = 1e-12
HIGH_URGENCY = 0.7
WINDOWS: tuple[tuple[str, int], ...] = (
    ("15m", 15 * MINUTE_MS), ("1h", HOUR_MS), ("4h", 4 * HOUR_MS), ("24h", 24 * HOUR_MS),
)
TAUS: tuple[tuple[str, int], ...] = (("1h", HOUR_MS), ("4h", 4 * HOUR_MS), ("24h", 24 * HOUR_MS))


@dataclass
class NewsData:
    avail: np.ndarray
    direction: np.ndarray
    relevance: np.ndarray
    urgency: np.ndarray
    magnitude: np.ndarray
    event_type: np.ndarray | None = None
    ids: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.avail.shape[0])

    @classmethod
    def empty(cls) -> "NewsData":
        z = np.zeros(0, dtype=np.int64)
        f = np.zeros(0, dtype=np.float64)
        return cls(avail=z, direction=f, relevance=f, urgency=f, magnitude=f)


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(news)")}


def load_news(db_path: Path, end_ms: int | None = None) -> NewsData:
    """Rows with a usable visibility time, sorted by that time."""
    path = Path(db_path)
    if not path.exists():
        return NewsData.empty()
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = _columns(conn)
        if not cols:
            return NewsData.empty()
        avail_expr = "COALESCE(available_at, ts_ingested)"
        direction_expr = "COALESCE(direction_score, direction, 0.0)"
        urgency_expr = "COALESCE(urgency, CASE WHEN impact IS NOT NULL THEN impact / 3.0 ELSE 0.0 END)"
        relevance_expr = "COALESCE(eth_relevance, 0.0)"
        magnitude_expr = "COALESCE(magnitude, 0.0)"
        type_expr = "COALESCE(event_type, '')" if "event_type" in cols else "''"
        where = "1=1"
        args: list = []
        if end_ms is not None:
            where += f" AND {avail_expr} <= ?"
            args.append(int(end_ms))
        sql = (f"SELECT id, {avail_expr} AS avail, {direction_expr}, {relevance_expr}, "
               f"{urgency_expr}, {magnitude_expr}, {type_expr} "
               f"FROM news WHERE {where} ORDER BY avail")
        rows = conn.execute(sql, tuple(args)).fetchall()
    finally:
        conn.close()
    if not rows:
        return NewsData.empty()
    cols = list(zip(*rows, strict=True))
    return NewsData(
        avail=np.asarray(cols[1], dtype=np.int64),
        direction=np.nan_to_num(np.asarray(cols[2], dtype=np.float64), nan=0.0),
        relevance=np.nan_to_num(np.asarray(cols[3], dtype=np.float64), nan=0.0),
        urgency=np.nan_to_num(np.asarray(cols[4], dtype=np.float64), nan=0.0),
        magnitude=np.nan_to_num(np.asarray(cols[5], dtype=np.float64), nan=0.0),
        event_type=np.asarray(cols[6], dtype=object),
        ids=np.asarray(cols[0], dtype=np.int64),
    )


def _cum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])


def _window(cum: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    return cum[np.clip(hi, 0, cum.shape[0] - 1)] - cum[np.clip(lo, 0, cum.shape[0] - 1)]


def _decay_sum(avail: np.ndarray, values: np.ndarray, decision_ms: np.ndarray,
               tau: int) -> np.ndarray:
    """Stable Σ values_i exp(-(t-avail_i)/tau) over a local six-tau window.

    A global ``exp(avail/tau)`` cumulative sum overflows because the archive
    spans years; the local window keeps every exponent <= 0.
    """
    n = decision_ms.shape[0]
    out = np.zeros(n, dtype=np.float64)
    horizon = int(min(30 * DAY_MS, max(6 * tau, DAY_MS)))
    lo = np.searchsorted(avail, decision_ms - horizon, side="right")
    hi = np.searchsorted(avail, decision_ms, side="right")
    for j in range(n):
        a, b = int(lo[j]), int(hi[j])
        if b <= a:
            continue
        age = decision_ms[j] - avail[a:b]
        out[j] = float(np.sum(values[a:b] * np.exp(-age / float(tau))))
    return out


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling median, excluding the current element."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n == 0:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    if n > w:
        out[w:] = np.median(sliding_window_view(x, w)[:-1], axis=1)
    return out


def feature_names() -> list[str]:
    names: list[str] = []
    for label, _ in WINDOWS:
        names += [f"news_count_{label}", f"news_score_{label}", f"news_pos_{label}",
                  f"news_neg_{label}", f"news_max_urgency_{label}", f"news_high_count_{label}"]
    for label, _ in TAUS:
        names += [f"news_decay_score_{label}", f"news_decay_count_{label}",
                  f"news_decay_weight_{label}"]
    names += ["news_net_sentiment", "news_anomaly_1h", "news_time_since_high", "news_coverage"]
    return names


def empty_columns(n: int) -> dict[str, np.ndarray]:
    return {name: np.full(n, np.nan) for name in feature_names()} | {
        "news_coverage": np.zeros(n, dtype=np.float64)
    }


def build_columns(decision_ms: np.ndarray, interval_ms: int, news: NewsData) -> dict[str, np.ndarray]:
    """News features for each decision, strictly ``available_at <= decision``."""
    decision_ms = np.asarray(decision_ms, dtype=np.int64)
    n = decision_ms.shape[0]
    if news.n == 0:
        return empty_columns(n)

    avail = news.avail
    direction = np.clip(news.direction, -1.0, 1.0)
    relevance = np.clip(news.relevance, 0.0, 1.0)
    urgency = np.clip(news.urgency, 0.0, 1.0)
    magnitude = np.clip(news.magnitude, 0.0, 1.0)
    score = direction * relevance * (0.25 + 0.75 * urgency)
    weight = relevance * (0.25 + 0.75 * urgency)
    pos = np.maximum(direction, 0.0) * weight
    neg = np.maximum(-direction, 0.0) * weight
    high = (urgency >= HIGH_URGENCY).astype(np.float64)

    cum_count = _cum(np.ones(news.n))
    cum_score = _cum(score)
    cum_pos = _cum(pos)
    cum_neg = _cum(neg)
    cum_urg = _cum(urgency * weight)
    cum_high = _cum(high)
    cum_mag = _cum(magnitude * weight)

    cols: dict[str, np.ndarray] = {}
    for label, w in WINDOWS:
        hi = np.searchsorted(avail, decision_ms, side="right")
        lo = np.searchsorted(avail, decision_ms - w, side="right")
        cnt = _window(cum_count, hi, lo)
        cols[f"news_count_{label}"] = cnt
        cols[f"news_score_{label}"] = _window(cum_score, hi, lo)
        cols[f"news_pos_{label}"] = _window(cum_pos, hi, lo)
        cols[f"news_neg_{label}"] = _window(cum_neg, hi, lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            cols[f"news_max_urgency_{label}"] = np.where(
                cnt > 0, _window(cum_urg, hi, lo) / np.maximum(_window(cum_count, hi, lo), 1.0), 0.0)
        cols[f"news_high_count_{label}"] = _window(cum_high, hi, lo)

    # Exponential time-decay sums, computed per decision over a local window so
    # exp() never overflows on a multi-year archive.
    for label, tau in TAUS:
        cols[f"news_decay_weight_{label}"] = _decay_sum(avail, weight, decision_ms, tau)
        cols[f"news_decay_score_{label}"] = _decay_sum(avail, score, decision_ms, tau)
        cols[f"news_decay_count_{label}"] = _decay_sum(avail, np.ones(news.n), decision_ms, tau)

    with np.errstate(divide="ignore", invalid="ignore"):
        cols["news_net_sentiment"] = cols["news_decay_score_4h"] / np.maximum(
            cols["news_decay_weight_4h"], EPS)

    # 1h count versus the typical hourly count over the previous 7 days.
    hour0 = int(avail[0] // HOUR_MS)
    hour_ids = (avail // HOUR_MS - hour0).astype(np.int64)
    n_hours = int(hour_ids.max()) + 1 if hour_ids.size else 1
    hourly = np.bincount(hour_ids, minlength=n_hours).astype(np.float64)
    typical = _rolling_median(hourly, 168)
    dh = (decision_ms // HOUR_MS - hour0).astype(np.int64)
    typical_at = np.where((dh >= 0) & (dh < typical.shape[0]), typical[np.clip(dh, 0, typical.shape[0] - 1)], np.nan)
    cols["news_anomaly_1h"] = cols["news_count_1h"] / np.maximum(typical_at, 1.0)

    # Time since the most recent high-urgency item.
    high_avail = avail[high > 0]
    if high_avail.shape[0]:
        j = np.searchsorted(high_avail, decision_ms, side="right") - 1
        since = np.full(n, np.nan)
        ok = j >= 0
        since[ok] = (decision_ms[ok] - high_avail[j[ok]]) / float(MINUTE_MS)
        cols["news_time_since_high"] = np.log1p(np.maximum(since, 0.0))
    else:
        cols["news_time_since_high"] = np.full(n, np.nan)

    # Coverage: from the first item until a week after the last known item.
    first = int(avail[0])
    last = int(avail[-1])
    covered = (decision_ms >= first) & (decision_ms <= last + 7 * DAY_MS)
    cols["news_coverage"] = covered.astype(np.float64)
    for name in feature_names():
        if name == "news_coverage":
            continue
        if name in cols:
            cols[name] = np.where(covered, cols[name], np.nan)
    return cols
