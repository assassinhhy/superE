"""End-to-end: the CLI on a synthetic database, no network, no real library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from synthetic import build_synthetic_db

from supere.train.__main__ import main

REPO = Path(__file__).resolve().parents[1]
FAST = ["--n-estimators", "30", "--n-jobs", "1"]


def _common(tmp_path: Path) -> list[str]:
    return [
        "--config", str(REPO / "config.toml"),
        "--train-config", str(REPO / "train.toml"),
        "--data-dir", str(tmp_path),
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_walk_forward_run_writes_log_and_json(tmp_path: Path):
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=500)
    before = _sha(db)

    rc = main(_common(tmp_path) + ["--mode", "walk_forward", "--folds", "2", *FAST])
    assert rc == 0

    assert _sha(db) == before, "training must not modify the database file"

    logs = sorted((tmp_path / "logs").glob("train-*.log"))
    jsons = sorted((tmp_path / "logs").glob("train-*.json"))
    assert logs and jsons
    text = logs[-1].read_text(encoding="utf-8")
    assert "开始离线训练" in text
    assert "回测 taker_both" in text

    result = json.loads(jsons[-1].read_text(encoding="utf-8"))
    assert result["oos_samples"] > 0
    assert result["classification"]["n"] == result["oos_samples"]
    assert set(result["backtest"]["scenarios"]) == {"taker_both", "maker_entry"}
    assert len(result["folds"]) == 2
    assert len(result["feature_hash"]) == 16


def test_holdout_run_with_limit_and_no_metrics(tmp_path: Path):
    build_synthetic_db(tmp_path / "superE.sqlite", n=400, with_metrics=False, with_funding=False)
    rc = main(_common(tmp_path) + ["--limit-bars", "300", "--no-metrics", "--no-funding", *FAST])
    assert rc == 0
    jsons = sorted((tmp_path / "logs").glob("train-*.json"))
    result = json.loads(jsons[-1].read_text(encoding="utf-8"))
    assert result["config"]["use_metrics"] is False
    assert result["config"]["use_funding"] is False
    assert result["oos_samples"] > 0


def test_non_trainable_interval_is_refused_with_exit_2(tmp_path: Path, capsys):
    build_synthetic_db(tmp_path / "superE.sqlite", n=50)
    rc = main(_common(tmp_path) + ["--interval", "1d"])
    assert rc == 2
    assert "不训练方向模型" in capsys.readouterr().err


def test_missing_database_exits_1_and_logs(tmp_path: Path):
    rc = main(_common(tmp_path) + [*FAST])
    assert rc == 1
    logs = sorted((tmp_path / "logs").glob("train-*.log"))
    assert logs and "找不到数据库" in logs[-1].read_text(encoding="utf-8")


def test_print_config_needs_no_database(tmp_path: Path, capsys):
    rc = main(_common(tmp_path) + ["--print-config", "--interval", "15m"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["interval"] == "15m"
    assert out["label"]["horizon"] == 4
