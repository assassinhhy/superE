"""Read-only payloads for the desktop UI. Nothing here writes SQLite."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from supere.ingest.aggregate import aggregate_1m, compare_bars
from supere.intervals import INTERVAL_MS, UM_INTERVALS
from supere.storage.dal import DataAccess
from supere.storage.models import Kline
from supere.timeutil import format_utc8

# Dates the plan already verified. The training page shows these so a
# training window is not placed before the data exists.
HISTORY_FLOORS: list[dict] = [
    {
        "id": "um",
        "since": "2019-11-27",
        "note": "ETHUSDT 永续上线。1 分钟月度存档从 2020-01 起。",
    },
    {
        "id": "metrics",
        "since": "2021-12-01",
        "note": "持仓量和多空比从这一天起。训练窗口不要更早。数值要到 bar 起点之后 5 分钟才算已知。",
    },
    {
        "id": "funding",
        "since": "2020-01",
        "note": "只有已结算资金费率。预测费率不进这张表。",
    },
    {
        "id": "spot",
        "since": "2017-08-17",
        "note": "现货只录 1 分钟，用来算基差。",
    },
    {
        "id": "depth",
        "since": "自录当天",
        "note": "20 档盘口没有历史。断档补不回来。bookDepth 存档不是 20 档，不导入。",
    },
    {
        "id": "liquidation",
        "since": "自录当天",
        "note": "强平推送是抽样，金额只能当下限。",
    },
]

# The buttons on the data page describe collector commands. The API does not run them.
DATA_COMMANDS: list[dict] = [
    {
        "id": "backfill",
        "label": "初始化历史（存档）",
        "command": "python -m supere backfill",
        "detail": "先停掉采集器。回填要占用数据锁，只读 API 不能代写。",
    },
    {
        "id": "gapfill",
        "label": "补缺口",
        "command": "采集器启动和断线后会自己补",
        "detail": "K 线和 aggTrade 用 REST 或存档补。盘口和强平只记断档，不假装补上。",
    },
    {
        "id": "export",
        "label": "导出 Parquet",
        "command": "原始流已经在 data/raw，训练快照是第二阶段",
        "detail": "这一阶段不从界面往 snapshots 写文件。",
    },
]


def _gap_count(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def _utc8(value) -> str | None:
    if value is None:
        return None
    return format_utc8(int(value))


def catalog(dal: DataAccess) -> dict:
    coverage = []
    for row in dal.coverage():
        item = dict(row)
        item["range_from_utc8"] = _utc8(item.get("range_from"))
        item["range_to_utc8"] = _utc8(item.get("range_to"))
        item["last_check_utc8"] = _utc8(item.get("last_check"))
        item["gap_count"] = _gap_count(item.get("gaps_json"))
        coverage.append(item)
    raw = []
    if dal.db is not None:
        for row in dal.db.raw_coverage_rows():
            item = dict(row)
            item["first_utc8"] = _utc8(item.get("first_ts"))
            item["last_utc8"] = _utc8(item.get("last_ts"))
            item["gap_count"] = int(item.get("gap_count") or 0)
            raw.append(item)
    return {
        "intervals": list(UM_INTERVALS),
        "markets": ["um", "um_mark", "um_index", "um_premium", "spot"],
        "coverage": coverage,
        "raw": raw,
        "floors": HISTORY_FLOORS,
        "commands": DATA_COMMANDS,
    }


def _as_kline(row, interval: str) -> Kline:
    return Kline(
        market=row["market"],
        symbol=row["symbol"],
        interval=interval,
        open_time=int(row["open_time"]),
        close_time=int(row["close_time"]),
        o=row["o"],
        h=row["h"],
        l=row["l"],
        c=row["c"],
        volume=row["volume"],
        quote_volume=row["quote_volume"],
        trades=int(row["trades"]),
        taker_buy_base=row["taker_buy_base"],
        taker_buy_quote=row["taker_buy_quote"],
        source=row["source"],
        ingested_at=int(row["ingested_at"]),
        ingested_local=int(row["ingested_local"]),
    )


def aggregate_check(dal: DataAccess, interval: str, bars: int) -> dict:
    """Compare stored higher-timeframe bars with a 1m aggregate. Read-only."""
    if dal.db is None:
        return {"ok": False, "compared": 0, "mismatches": [], "detail": "数据库还没准备好"}
    if interval == "1m":
        return {"ok": False, "compared": 0, "mismatches": [], "detail": "1 分钟就是基准，不用再聚合"}
    if interval not in UM_INTERVALS:
        return {"ok": False, "compared": 0, "mismatches": [], "detail": "未知周期"}
    step = INTERVAL_MS.get(interval)
    if step is None:
        return {
            "ok": False,
            "compared": 0,
            "mismatches": [],
            "detail": "月线跨度大，用 scripts/accept_1a.py 在本机对完整月份",
        }
    minutes = step // INTERVAL_MS["1m"]
    if bars * minutes > 20_000:
        return {
            "ok": False,
            "compared": 0,
            "mismatches": [],
            "detail": "这次要读的 1 分钟太多，把根数调小",
        }
    official_rows = dal.db.fetch_klines("um", "ETHUSDT", interval, limit=bars)
    if not official_rows:
        return {"ok": False, "compared": 0, "mismatches": [], "detail": "本地没有这个周期的合约 K 线"}
    start = int(official_rows[0]["open_time"])
    end = int(official_rows[-1]["close_time"])
    one_rows = dal.db.fetch_klines("um", "ETHUSDT", "1m", start=start, end=end, limit=20_000)
    aggregated = aggregate_1m([_as_kline(row, "1m") for row in one_rows], interval)
    by_open = {bar.open_time: bar for bar in aggregated}
    mismatches = []
    for row in official_rows:
        official = _as_kline(row, interval)
        got = by_open.get(official.open_time)
        if got is None:
            mismatches.append(
                {
                    "open_time": official.open_time,
                    "open_time_utc8": format_utc8(official.open_time),
                    "problems": ["1 分钟不完整，聚不出来"],
                }
            )
            continue
        problems = compare_bars(got, official)
        if problems:
            mismatches.append(
                {
                    "open_time": official.open_time,
                    "open_time_utc8": format_utc8(official.open_time),
                    "problems": problems,
                }
            )
    return {
        "ok": not mismatches,
        "compared": len(official_rows),
        "mismatches": mismatches,
        "interval": interval,
        "detail": "价格必须完全相等，成交量相对误差不超过 1e-6",
    }


def disk_view(path: Path | None, warn_gb: float, critical_gb: float) -> dict | None:
    if path is None or not path.exists():
        return None
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < critical_gb:
        level = "critical"
    elif free_gb < warn_gb:
        level = "warn"
    else:
        level = "ok"
    return {"free_gb": round(free_gb, 2), "level": level}


def settings_view(cfg, data_path: Path | None) -> dict:
    return {
        "timezone": "UTC+8",
        "proxy": cfg.proxy.url,
        "no_proxy": cfg.proxy.no_proxy,
        "data_dir": str(data_path) if data_path is not None else cfg.data.dir,
        "fees": cfg.fees.model_dump(),
        "anomaly": cfg.anomaly.model_dump(),
        "disk": cfg.disk.model_dump(),
        "symbols": cfg.symbols.model_dump(),
        "api_port": cfg.server.api_port,
        "editable": False,
        "note": "这里只展示配置。要改代理、阈值或费率，编辑 config.local.toml 后重启。界面不写数据库，也不保存密钥。",
    }


MISSING_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>superE</title>
<style>
  body { margin: 0; background: #12140f; color: #e6e2d6; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
  main { max-width: 38rem; margin: 12vh auto; padding: 0 1.5rem; }
  h1 { font-weight: 560; font-size: 1.6rem; }
  code { color: #d7a15f; }
  p { line-height: 1.6; color: #c9c3b0; }
</style>
</head>
<body>
<main>
  <h1>superE 界面还没构建</h1>
  <p>窗口已经连上本机 API。把前端依赖装好并构建之后，刷新这个页面。</p>
  <p>在仓库的 <code>frontend</code> 目录执行 <code>pnpm install</code> 然后 <code>pnpm build</code>。没有 pnpm 时用 <code>npm install</code> 和 <code>npm run build</code>。</p>
  <p>构建结果在 <code>frontend/dist</code>，只从本机读取，不访问外网。</p>
</main>
</body>
</html>
"""
