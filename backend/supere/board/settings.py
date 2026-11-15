"""``analysis.toml`` plus optional ``analysis.local.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class BoardEndpoints(BaseModel):
    binance: str = "https://www.binance.com"
    okx_ticker: str = "https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP"


class BoardPoll(BaseModel):
    oi_seconds: int = 60
    okx_seconds: int = 60
    metrics_fallback_after_minutes: int = 12
    funding_fallback_after_seconds: int = 120


class BoardBackoff(BaseModel):
    first_seconds: int = 60
    max_seconds: int = 900
    ban_extra_seconds: int = 30
    ban_parse_fail_seconds: int = 600


class BoardStale(BaseModel):
    oi_seconds: int = 180
    metrics_minutes: int = 12
    funding_seconds: int = 120
    okx_seconds: int = 180
    kline_grace_seconds: int = 90


class BoardCrosscheck(BaseModel):
    max_diff_pct: float = 0.3


class BoardSection(BaseModel):
    port: int = 8770
    symbol: str = "ETHUSDT"
    bars: int = 300
    warmup_bars: int = 600
    recompute_on: str = "5m_close"
    price_refresh_seconds: int = 15
    endpoints: BoardEndpoints = Field(default_factory=BoardEndpoints)
    poll: BoardPoll = Field(default_factory=BoardPoll)
    backoff: BoardBackoff = Field(default_factory=BoardBackoff)
    stale: BoardStale = Field(default_factory=BoardStale)
    crosscheck: BoardCrosscheck = Field(default_factory=BoardCrosscheck)


class IndicatorSection(BaseModel):
    ema: list[int] = Field(default_factory=lambda: [20, 50, 200])
    rsi: int = 14
    macd: list[int] = Field(default_factory=lambda: [12, 26, 9])
    atr: int = 14


class TrendSection(BaseModel):
    frames: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    tangle_pct: float = 0.15
    daily_rsi_ob: float = 70
    daily_rsi_os: float = 30
    macd_fade_bars: int = 2


class SentimentSection(BaseModel):
    frame: str = "15m"
    lookback_bars: int = 4
    dp_pct: float = 0.3
    doi_pct: float = 0.3
    global_lsr_crowded_long: float = 3.0
    global_lsr_crowded_short: float = 0.8
    funding_hot_abs: float = 0.0003
    funding_neutral_abs: float = 0.00005
    vol_spike_mult: float = 3.0
    vol_spike_window: int = 20
    taker_sell: float = 0.7
    taker_buy: float = 1.4


class LevelsSection(BaseModel):
    frame: str = "15m"
    also_1h: bool = False
    merge_atr: float = 0.5
    swing_lookback: int = 48
    fractal_wing: int = 3
    wick_body_mult: float = 2.0
    wick_atr_mult: float = 1.0
    round_step: float = 50
    ma_levels: list[str] = Field(
        default_factory=lambda: ["15m:EMA20", "15m:EMA50", "1h:EMA20", "1h:EMA50", "4h:EMA20"]
    )
    touch_atr: float = 0.2
    touch_window_bars: int = 96
    nearest_each_side: int = 3
    day_start_utc8: str = "00:00"
    include_5m_wicks: bool = True
    wick_5m_lookback: int = 288


class RangeSection(BaseModel):
    bars: int = 12
    shrink_recent: int = 4
    shrink_base: int = 16
    shrink_ratio: float = 0.6
    upper_pct: float = 25
    lower_pct: float = 25
    structure_swings: int = 3


class PlansSection(BaseModel):
    max_plans: int = 4
    entry_band_atr: float = 0.3
    stop_atr15: float = 0.5
    stop_atr1h_factor: float = 0.35
    breakout_retest_atr: float = 0.2
    breakout_stop_atr: float = 0.6
    # Breakout/breakdown fixed entry sits this many ATR15m through the key level.
    breakout_entry_atr: float = 0.1
    # ETHUSDT USDT-M tick. Fixed entries are rounded to this.
    tick: float = 0.01
    counter_size_ratio: float = 0.5
    flip_closes_15m: int = 2
    flip_closes_1h: int = 1
    flip_closes_4h: int = 1
    valid_bars_15m: int = 4


class ConfirmSection(BaseModel):
    frames: list[str] = Field(default_factory=lambda: ["5m", "15m"])
    wick_body_mult: float = 1.5
    wick_atr_mult: float = 0.5
    rsi_short_from: float = 60
    rsi_long_from: float = 40
    rsi_lookback_bars: int = 6


class CalcSection(BaseModel):
    capital_u: float = 100
    max_loss_u: float = 4
    max_loss_pct: float = 0
    leverage: float = 20
    leverage_max: float = 50
    margin_mode: str = "isolated"
    mmr: float = 0.005
    taker_fee: float = 0.0005
    liq_min_atr: float = 1.0
    fee_min_mult: float = 3
    rr_min: float = 1.0


class RiskSection(BaseModel):
    loss_streak_stop: int = 3
    event_window_minutes: int = 15
    event_kinds: list[str] = Field(default_factory=lambda: ["CPI", "PPI", "NFP", "FOMC"])
    weekend_days: list[str] = Field(default_factory=lambda: ["Sat", "Sun"])


class NotifySection(BaseModel):
    system: bool = False
    cooldown_minutes: int = 30


class AnalysisConfig(BaseModel):
    board: BoardSection = Field(default_factory=BoardSection)
    indicators: IndicatorSection = Field(default_factory=IndicatorSection)
    trend: TrendSection = Field(default_factory=TrendSection)
    sentiment: SentimentSection = Field(default_factory=SentimentSection)
    levels: LevelsSection = Field(default_factory=LevelsSection)
    range: RangeSection = Field(default_factory=RangeSection)
    plans: PlansSection = Field(default_factory=PlansSection)
    confirm: ConfirmSection = Field(default_factory=ConfirmSection)
    calc: CalcSection = Field(default_factory=CalcSection)
    risk: RiskSection = Field(default_factory=RiskSection)
    notify: NotifySection = Field(default_factory=NotifySection)


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def find_analysis_path(explicit: str | None = None, config_path: Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    if config_path is not None:
        candidate = config_path.parent / "analysis.toml"
        if candidate.is_file():
            return candidate
    cwd = Path.cwd() / "analysis.toml"
    if cwd.is_file():
        return cwd
    return None


def load_analysis(
    explicit: str | None = None, config_path: Path | None = None
) -> tuple[AnalysisConfig, Path | None]:
    path = find_analysis_path(explicit, config_path)
    raw: dict = {}
    if path is not None and path.is_file():
        raw = _merge(raw, _read_toml(path))
        raw = _merge(raw, _read_toml(path.with_name("analysis.local.toml")))
    return AnalysisConfig.model_validate(raw), path
