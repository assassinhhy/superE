from __future__ import annotations

import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from supere.hist.convert import convert_archive
from supere.hist.schema import HIST_SCHEMAS, partition_dir


def _zip(tmp_path: Path, name: str, text: str) -> Path:
    csv = tmp_path / name.replace(".zip", ".csv")
    csv.write_text(text, encoding="utf-8")
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv, arcname=csv.name)
    return archive


def test_agg_trades_headerless_and_multi_day(tmp_path: Path):
    text = (
        "100,2000.5,1.25,10,10,1672531200000,false\n"
        "101,2001.0,0.50,11,11,1672531205000,true\n"
        "102,2002.0,2.00,12,12,1672617600000,false\n"
    )
    archive = _zip(tmp_path, "ETHUSDT-aggTrades-2023-01-01.zip", text)
    result = convert_archive(archive, "aggTrades", tmp_path / "raw", period="daily")
    assert result.rows == 3
    assert set(result.date_counts) == {"2023-01-01", "2023-01-02"}

    first = pq.read_table(partition_dir(tmp_path / "raw", "aggTrades", "2023-01-01") / "part-000.parquet")
    assert first.schema.equals(HIST_SCHEMAS["aggTrades_hist"])
    assert first.num_rows == 2
    row = first.to_pylist()[0]
    assert row["agg_id"] == 100
    assert row["price"] == "2000.5"
    assert row["qty"] == "1.25"
    assert row["ts"] == 1672531200000
    assert row["recv_ms"] == row["ts"]
    assert row["is_buyer_maker"] is False
    assert row["source"] == "binance_vision:daily"

    second = pq.read_table(partition_dir(tmp_path / "raw", "aggTrades", "2023-01-02") / "part-000.parquet")
    assert second.num_rows == 1


def test_agg_trades_with_header(tmp_path: Path):
    text = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.0,2.0,5,6,1704067200000,true\n"
    )
    archive = _zip(tmp_path, "ETHUSDT-aggTrades-2024-01-01.zip", text)
    result = convert_archive(archive, "aggTrades", tmp_path / "raw", period="monthly")
    assert result.rows == 1
    table = pq.read_table(partition_dir(tmp_path / "raw", "aggTrades", "2024-01-01") / "part-000.parquet")
    assert table.to_pylist()[0]["price"] == "100.0"
    assert table.to_pylist()[0]["source"] == "binance_vision:monthly"


def test_book_depth(tmp_path: Path):
    lines = ["timestamp,percentage,depth,notional"]
    for pct in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
        shown = f"{pct:.2f}" if pct == -5 else str(pct)
        lines.append(f"2023-01-01 00:00:05,{shown},{100 + pct}.5,{200 + pct}.25")
    text = "\n".join(lines) + "\n"
    archive = _zip(tmp_path, "ETHUSDT-bookDepth-2023-01-01.zip", text)
    result = convert_archive(archive, "bookDepth", tmp_path / "raw")
    assert result.rows == 10
    table = pq.read_table(partition_dir(tmp_path / "raw", "bookDepth", "2023-01-01") / "part-000.parquet")
    assert table.schema.equals(HIST_SCHEMAS["bookDepth_hist"])
    row = table.to_pylist()[0]
    assert row["ts"] == 1672531205000
    assert row["percentage"] == -5
    assert row["depth"] == "95.5"
    assert row["notional"] == "195.25"


def test_metrics_column_mapping_and_empty_quotes(tmp_path: Path):
    text = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2021-12-01 00:00:00,ETHUSDT,100.5,200.5,1.1,,3.3,4.4\n"
    )
    archive = _zip(tmp_path, "ETHUSDT-metrics-2021-12-01.zip", text)
    result = convert_archive(archive, "metrics", tmp_path / "raw")
    assert result.rows == 1
    table = pq.read_table(partition_dir(tmp_path / "raw", "metrics", "2021-12-01") / "part-000.parquet")
    assert table.schema.equals(HIST_SCHEMAS["metrics_hist"])
    row = table.to_pylist()[0]
    assert row["ts"] == 1638316800000
    assert row["oi"] == "100.5"
    # count_* -> top_acc_lsr, sum_* -> top_pos_lsr (same as parse.py).
    assert row["top_acc_lsr"] == "1.1"
    assert row["top_pos_lsr"] in ("", None)
    assert row["global_lsr"] == "3.3"
    assert row["taker_lsr"] == "4.4"


def test_zero_row_archive_is_a_failure(tmp_path: Path):
    text = "timestamp,percentage,depth,notional\n"
    archive = _zip(tmp_path, "ETHUSDT-bookDepth-2023-01-01.zip", text)
    try:
        convert_archive(archive, "bookDepth", tmp_path / "raw")
    except ValueError as exc:
        assert "空存档" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("零行存档应报错")
    assert not (
        partition_dir(tmp_path / "raw", "bookDepth", "2023-01-01") / "part-000.parquet"
    ).exists()
