"""Aggregate 1m candles onto a higher Binance UTC interval and compare them."""

from __future__ import annotations

from decimal import Decimal

from supere.intervals import align_open, close_time_of
from supere.storage.models import Kline

_REL = Decimal("1e-6")


def _d(value: str) -> Decimal:
    return Decimal(value)


def aggregate_1m(bars: list[Kline], interval: str, ingested_at: int = 0) -> list[Kline]:
    """Sum complete buckets. A bucket is emitted only when every 1m bar is present."""
    if interval == "1m":
        return list(bars)
    grouped: dict[int, list[Kline]] = {}
    for bar in bars:
        if bar.interval != "1m":
            raise ValueError("聚合输入必须是 1m")
        grouped.setdefault(align_open(interval, bar.open_time), []).append(bar)
    out: list[Kline] = []
    for open_time in sorted(grouped):
        chunk = sorted(grouped[open_time], key=lambda bar: bar.open_time)
        expected = _expected_count(interval, open_time, chunk)
        if len({bar.open_time for bar in chunk}) != expected:
            continue
        first, last = chunk[0], chunk[-1]
        out.append(
            Kline(
                market=first.market,
                symbol=first.symbol,
                interval=interval,
                open_time=open_time,
                close_time=close_time_of(interval, open_time),
                o=first.o,
                h=dec_str(max(_d(bar.h) for bar in chunk)),
                l=dec_str(min(_d(bar.l) for bar in chunk)),
                c=last.c,
                volume=dec_str(sum((_d(bar.volume) for bar in chunk), Decimal(0))),
                quote_volume=dec_str(sum((_d(bar.quote_volume) for bar in chunk), Decimal(0))),
                trades=sum(bar.trades for bar in chunk),
                taker_buy_base=dec_str(sum((_d(bar.taker_buy_base) for bar in chunk), Decimal(0))),
                taker_buy_quote=dec_str(sum((_d(bar.taker_buy_quote) for bar in chunk), Decimal(0))),
                source="aggregate",
                ingested_at=ingested_at,
                ingested_local=ingested_at,
            )
        )
    return out


def _expected_count(interval: str, open_time: int, chunk: list[Kline]) -> int:
    from supere.intervals import INTERVAL_MS, next_open

    if interval in INTERVAL_MS and interval != "1M":
        return INTERVAL_MS[interval] // INTERVAL_MS["1m"]
    # Month length varies. Count 1m slots from open to next open.
    return (next_open(interval, open_time) - open_time) // INTERVAL_MS["1m"]


def dec_str(value: Decimal) -> str:
    return format(value, "f")


def prices_equal(left: str, right: str) -> bool:
    return _d(left) == _d(right)


def volume_close(left: str, right: str, tol: Decimal = _REL) -> bool:
    a, b = _d(left), _d(right)
    if a == b:
        return True
    denom = max(abs(a), abs(b), Decimal(1))
    return abs(a - b) / denom <= tol


def compare_bars(aggregated: Kline, official: Kline) -> list[str]:
    """Return a list of mismatch descriptions. Empty means the bars agree."""
    problems = []
    for field in ("o", "h", "l", "c"):
        if not prices_equal(getattr(aggregated, field), getattr(official, field)):
            problems.append(f"{field} {getattr(aggregated, field)} != {getattr(official, field)}")
    for field in ("volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        if not volume_close(getattr(aggregated, field), getattr(official, field)):
            problems.append(f"{field} {getattr(aggregated, field)} != {getattr(official, field)}")
    if aggregated.trades != official.trades:
        problems.append(f"trades {aggregated.trades} != {official.trades}")
    return problems
