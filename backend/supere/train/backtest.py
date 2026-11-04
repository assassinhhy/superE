"""Simple out-of-sample backtest (plan 7.4).

Rules:

* Only out-of-sample rows are passed in. The model never trades on rows it
  was trained on.
* A signal decided at bar ``i`` opens at the open of bar ``i+1``, the same
  price the label uses. It closes at the first barrier touched, or at the
  close of bar ``i+H``.
* One position at a time. A new signal is ignored while a trade is open.
* Costs are charged per side. Two fee scenarios are reported:
  ``taker_both`` (0.05% in and out, the default because it is the conservative
  reading of the owner's rule) and ``maker_entry`` (0.02% in as a resting
  limit, 0.05% out as a taker). Slippage is a flat per-side fraction, zero by
  default.
* Funding: a long pays and a short receives ``rate`` for every settlement in
  ``(entry, exit]``. Sign is ``-direction * rate``.

The numbers are a process check, not a strategy claim. Sample counts are
small, and the labels overlap (plan 8.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from supere.train.config import CostConfig
from supere.train.data import Funding


@dataclass(frozen=True)
class FeeScenario:
    name: str
    entry_fee: float
    exit_fee: float


def scenarios(cost: CostConfig) -> list[FeeScenario]:
    return [
        FeeScenario("taker_both", cost.taker_fee, cost.taker_fee),
        FeeScenario("maker_entry", cost.maker_fee, cost.taker_fee),
    ]


def signal_dirs(proba: np.ndarray, threshold: float) -> np.ndarray:
    """+1 long, -1 short, 0 no trade. Column order is [-1, 0, +1]."""
    p_dn = proba[:, 0]
    p_up = proba[:, 2]
    long = (p_up >= threshold) & (p_up > p_dn)
    short = (p_dn >= threshold) & (p_dn > p_up)
    return np.where(long, 1, np.where(short, -1, 0)).astype(np.int64)


def simulate(
    *,
    sample_bar: np.ndarray,  # decision bar i of each sample, sorted
    exit_bar: np.ndarray,
    entry: np.ndarray,  # open of bar i+1
    exit_price: np.ndarray,
    ambiguous: np.ndarray,
    dirs: np.ndarray,
    open_time: np.ndarray,
    interval_ms: int,
    funding: Funding,
    scenario: FeeScenario,
    slippage_per_side: float,
) -> list[dict]:
    trades: list[dict] = []
    busy_until = -1
    for p in range(sample_bar.shape[0]):
        d = int(dirs[p])
        if d == 0:
            continue
        entry_bar = int(sample_bar[p]) + 1
        if entry_bar <= busy_until:
            continue
        ex = int(exit_bar[p])
        px_in = float(entry[p])
        px_out = float(exit_price[p])

        gross = d * (px_out - px_in) / px_in
        fees = scenario.entry_fee + scenario.exit_fee
        slip = 2.0 * slippage_per_side

        entry_time = int(open_time[entry_bar])
        exit_time = int(open_time[ex]) + interval_ms  # close of the exit bar
        fund_sum = 0.0
        if funding.time.shape[0]:
            lo = np.searchsorted(funding.time, entry_time, side="right")
            hi = np.searchsorted(funding.time, exit_time, side="right")
            fund_sum = float(np.sum(funding.rate[lo:hi]))
        funding_cost = d * fund_sum

        net = gross - fees - slip - funding_cost
        trades.append(
            {
                "entry_bar": entry_bar,
                "exit_bar": ex,
                "dir": d,
                "gross": gross,
                "fees": fees,
                "slippage": slip,
                "funding": funding_cost,
                "net": net,
                "ambiguous": bool(ambiguous[p]),
            }
        )
        busy_until = ex
    return trades


def summarize(trades: list[dict], n_samples: int) -> dict:
    n = len(trades)
    out: dict = {"trades": n, "coverage": (n / n_samples) if n_samples else float("nan")}
    if n == 0:
        out.update(
            {
                "hit_rate": float("nan"),
                "expectancy": float("nan"),
                "gross_expectancy": float("nan"),
                "payoff_ratio": float("nan"),
                "compounded": 0.0,
                "max_drawdown": 0.0,
                "max_consecutive_losses": 0,
                "ambiguous_trades": 0,
                "funding_total": 0.0,
            }
        )
        return out

    net = np.array([t["net"] for t in trades], dtype=np.float64)
    gross = np.array([t["gross"] for t in trades], dtype=np.float64)
    wins = net[net > 0]
    losses = net[net <= 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = equity / peak - 1.0

    streak = best = 0
    for r in net:
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)

    out.update(
        {
            "hit_rate": float(np.mean(net > 0)),
            "expectancy": float(net.mean()),
            "gross_expectancy": float(gross.mean()),
            "payoff_ratio": (avg_win / abs(avg_loss)) if avg_loss else float("nan"),
            "compounded": float(equity[-1] - 1.0),
            "max_drawdown": float(drawdown.min()),
            "max_consecutive_losses": int(best),
            "ambiguous_trades": int(sum(1 for t in trades if t["ambiguous"])),
            "funding_total": float(sum(t["funding"] for t in trades)),
        }
    )
    return out
