"""``config.toml`` plus optional ``config.local.toml``. No secrets in either file."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from supere.errors import ConfigError


class ProxyConfig(BaseModel):
    url: str = "http://127.0.0.1:7890"
    no_proxy: str = "127.0.0.1,localhost"


class ServerConfig(BaseModel):
    api_host: str = "127.0.0.1"
    api_port: int = 8766
    ipc_host: str = "127.0.0.1"
    ipc_port: int = 8767


class EndpointConfig(BaseModel):
    fapi: str = "https://fapi.binance.com"
    spot_rest: str = "https://api.binance.com"
    spot_rest_fallback: str = "https://data-api.binance.vision"
    vision: str = "https://data.binance.vision/data"
    fstream_market: str = "wss://fstream.binance.com/market/ws"
    fstream_public: str = "wss://fstream.binance.com/public/ws"
    spot_ws: str = "wss://stream.binance.com:9443/ws"


class DataConfig(BaseModel):
    dir: str = "data"


class SymbolConfig(BaseModel):
    um: str = "ETHUSDT"
    spot: str = "ETHUSDT"


class RateLimitConfig(BaseModel):
    futures_weight_per_min: int = 2400
    futures_budget_ratio: float = 0.5
    spot_weight_per_min: int = 6000
    spot_budget_ratio: float = 0.5
    futures_data_requests_per_5min: int = 1000
    futures_data_budget_ratio: float = 0.5
    header_slowdown_ratio: float = 0.6
    # Not confirmed from the official text. Kept as a conservative constant.
    agg_trade_weight: int = 20
    funding_weight: int = 1
    open_interest_weight: int = 1


class ClockConfig(BaseModel):
    recalibrate_seconds: int = 600
    samples: int = 3
    warn_abs_ms: int = 2000
    change_alert_ms: int = 500


class IngestConfig(BaseModel):
    reconnect_hours: float = 23
    overlap_confirm_messages: int = 5
    parquet_flush_seconds: float = 30
    jsonl_fsync_seconds: float = 1
    aggtrade_archive_gap_hours: float = 2
    kline_verify_delay_seconds: float = 60
    depth_gap_ms: int = 2000
    metrics_poll_seconds: int = 300
    mark_kline_poll_seconds: int = 60
    wal_checkpoint_seconds: int = 300
    client_ping_seconds: int = 20
    startup_tail_limit: int = 499
    # Kline holes longer than this are filled from the archive before REST.
    archive_gap_days: float = 7


class DiskConfig(BaseModel):
    warn_free_gb: float = 50
    critical_free_gb: float = 20


class AnomalyConfig(BaseModel):
    volume_median_mult: float = 5
    price_atr_mult: float = 4
    price_10s_pct: float = 0.5
    liq_single_usd: float = 1_000_000
    liq_minute_usd: float = 5_000_000
    oi_change_pct: float = 3
    funding_jump_abs: float = 0.0005
    atr_period: int = 14


class NewsConfig(BaseModel):
    enabled_sources: list[str] = Field(default_factory=list)
    fear_greed_poll_seconds: int = 6 * 3600


class BackupConfig(BaseModel):
    hour_utc8: int = 3
    keep_days: int = 14
    restore_drill_days: int = 30


class FeesConfig(BaseModel):
    """Public VIP0-style rates from the plan. The account's real rate wins."""

    maker: float = 0.0002
    taker: float = 0.0005
    default_side: str = "taker"
    slippage_ticks: int = 1


class AppConfig(BaseModel):
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    endpoints: EndpointConfig = Field(default_factory=EndpointConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    symbols: SymbolConfig = Field(default_factory=SymbolConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    clock: ClockConfig = Field(default_factory=ClockConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    disk: DiskConfig = Field(default_factory=DiskConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)


def find_config_path(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigError(f"找不到配置文件: {path}")
        return path
    env = os.environ.get("SUPERE_CONFIG")
    if env:
        path = Path(env)
        if not path.is_file():
            raise ConfigError(f"SUPERE_CONFIG 指向的文件不存在: {path}")
        return path
    cwd = Path.cwd()
    direct = cwd / "config.toml"
    if direct.is_file():
        return direct
    for parent in cwd.resolve().parents:
        candidate = parent / "config.toml"
        if candidate.is_file():
            return candidate
    raise ConfigError("当前目录及上级没有 config.toml")


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


def load_config(explicit: str | None = None) -> tuple[AppConfig, Path]:
    path = find_config_path(explicit)
    raw = _read_toml(path)
    local = path.with_name("config.local.toml")
    raw = _merge(raw, _read_toml(local))
    cfg = AppConfig.model_validate(raw)
    return cfg, path


def data_dir(cfg: AppConfig, config_path: Path) -> Path:
    path = Path(cfg.data.dir)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def proxy_or_none(url: str) -> str | None:
    cleaned = url.strip()
    if cleaned.lower() in {"", "none", "direct", "off"}:
        return None
    return cleaned


def apply_no_proxy(value: str) -> None:
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def assert_loopback(host: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError(f"只允许把 API/IPC 绑在回环地址上，当前是 {host}")
