"""Training-scheme iteration driver (2026-11-05 plan: 计划-训练口径迭代).

One call = one round for the intervals in the round config. Read-only: opens the
SQLite file with ``mode=ro``, writes only the round JSON and a log.

Design, and why:

* **Frozen snapshot.** The collector keeps writing. Every bar is taken with
  ``open_time + interval <= snapshot_ms``, and the snapshot is stored in the
  round config, so rounds are comparable and a rerun gives the same numbers.
* **Trade-outcome labels per side** (``trade_labels``): R units, after fees and
  funding. Two binary models per interval, long-vs-not and short-vs-not.
* **Purged walk-forward, >= 5 folds, on the first ``cv_frac`` of samples.** The
  newest ``1 - cv_frac`` is the FINAL HOLDOUT and is never touched here unless
  ``--final`` is passed. Embargo = horizon. A training row is kept only if its
  label's exit bar is before the test window start minus the embargo.
* **Confidence cut-off chosen inside the training fold.** Inside each training
  fold, the last ``1 - inner_frac`` rows are an inner validation block (purged).
  An inner model picks the top-quantile ``q`` from ``q_grid`` that maximises mean
  net R there (needs ``min_inner`` rows and a positive mean). The fold model then
  uses the probability cut-off from that quantile on the test block.
* **Simulation** on the test block: one position at a time, the candidate with
  the higher probability wins when both sides pass. Entry at the next open, exit
  at the label's exit bar. Reset per fold.

Reported per interval and per target variant: trades, win rate, mean net R
(taker both sides and maker entry), profit factor, max drawdown in R, per-fold
sign, and the target check from the plan (section 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from supere.intervals import INTERVAL_MS
from supere.train import data, splits, trade_features, trade_labels
from supere.train.runner import _window_has_gap

WARMUP = data.WARMUP_BARS
DEFAULT_HORIZON = {"5m": 4, "15m": 4, "30m": 4, "1h": 4, "4h": 3}
HOLDOUT_NOTE = "最终 holdout 只在 --final 时计算，迭代过程中不看"


def _log(msg: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


# ---------------------------------------------------------------- inputs

def load_inputs(db: Path, cfg: dict, interval: str, snapshot_ms: int):
    market, symbol = "um", "ETHUSDT"
    higher_name = trade_features.HIGHER_TF.get(interval)
    use_mtf = bool(cfg["features"].get("multi_tf", True))
    conn = data.open_readonly(db)
    try:
        max_bars = cfg.get("max_bars")
        if isinstance(max_bars, dict):
            max_bars = max_bars.get(interval)
        bars = data.load_bars(conn, market, symbol, interval, max_bars, close_by_ms=snapshot_ms)
        funding = data.load_funding(conn, symbol)
        metrics = data.load_metrics(conn, symbol)
        higher = None
        if use_mtf and higher_name:
            higher = data.load_bars(conn, market, symbol, higher_name, None, close_by_ms=snapshot_ms)
    finally:
        conn.close()
    return bars, funding, metrics, higher


# ---------------------------------------------------------------- samples

def build_samples(bars, funding, metrics, higher, interval: str, horizon: int, tp_r: float, cfg: dict):
    f = cfg["features"]
    names, X = trade_features.build(
        bars, funding, metrics, higher,
        drop_calendar=bool(f.get("drop_calendar", True)),
        use_structure=bool(f.get("structure", True)),
        use_lsr=bool(f.get("lsr_oi", True)),
        use_multi_tf=bool(f.get("multi_tf", True)),
    )
    setup = cfg.get("setup")
    lmask = smask = None
    if setup:
        from supere.train import setups

        lmask, smask, s_names, s_block = setups.masks_and_features(
            bars, int(setup.get("lookback", 20)), int(setup.get("recent", 10)), float(setup.get("touch_atr", 0.5)))
        names = names + s_names
        X = np.hstack([X, s_block])

    tl = trade_labels.build(bars, horizon, tp_r, float(cfg.get("k_sl", 1.0)), 14, funding)
    rows = tl.idx
    keep = rows >= WARMUP
    req = trade_features.required_columns(names)
    keep &= np.all(np.isfinite(X[rows][:, req]), axis=1)
    gaps = data.gap_after(bars)
    keep &= ~_window_has_gap(gaps, rows, horizon)
    pos = np.nonzero(keep)[0]
    idx = rows[pos]
    samples = {
        "names": names,
        "idx": idx,
        "X": X[idx],
        "entry_bar": idx + 1,
        "entry_time": bars.open_time[idx + 1],
        "bar_range": (int(bars.open_time[0]), int(bars.open_time[-1])),
        "n_bars": int(bars.n),
        "gap_bars": int(np.sum(gaps)),
        "sides": {},
        "setup_rate": None,
    }
    for name, so in tl.sides.items():
        mask_all = lmask if name == "long" else smask
        samples["sides"][name] = {
            "y": so.y[pos],
            "net_taker": so.net_taker[pos],
            "net_maker": so.net_maker[pos],
            "gross": so.gross[pos],
            "outcome": so.outcome[pos],
            "exit_bar": so.exit_bar[pos],
            "d": so.side,
            # Candidate gate: all True when no setup is configured.
            "mask": (mask_all[idx].copy() if mask_all is not None else np.ones(idx.shape[0], dtype=bool)),
        }
    if setup:
        samples["setup_rate"] = {nm: float(np.mean(v["mask"])) for nm, v in samples["sides"].items()}
    samples["exit_any"] = np.maximum(samples["sides"]["long"]["exit_bar"], samples["sides"]["short"]["exit_bar"])
    samples["label_stats"] = _label_stats(tl, pos)
    return samples


def _label_stats(tl: trade_labels.TradeLabels, pos: np.ndarray) -> dict:
    out = {}
    for name, so in tl.sides.items():
        o = so.outcome[pos]
        n = max(int(o.shape[0]), 1)
        out[name] = {
            "y_rate": float(np.mean(so.y[pos])) if pos.size else float("nan"),
            "target_rate": float(np.sum(o == 1)) / n,
            "stop_rate": float(np.sum(o == -1)) / n,
            "timeout_rate": float(np.sum(o == 0)) / n,
            "mean_net_taker_R": float(np.mean(so.net_taker[pos])) if pos.size else float("nan"),
        }
    return out


# ---------------------------------------------------------------- model

def fit_binary(x: np.ndarray, y: np.ndarray, mcfg: dict):
    if np.unique(y).shape[0] < 2:
        return None
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(mcfg["n_estimators"]),
        learning_rate=float(mcfg["learning_rate"]),
        num_leaves=int(mcfg["num_leaves"]),
        min_child_samples=int(mcfg["min_child_samples"]),
        subsample=float(mcfg["subsample"]),
        subsample_freq=1,
        colsample_bytree=float(mcfg["colsample_bytree"]),
        reg_lambda=float(mcfg.get("reg_lambda", 0.0)),
        random_state=int(mcfg["seed"]),
        n_jobs=int(mcfg["n_jobs"]),
        verbose=-1,
    )
    model.fit(x, y)
    return model


def choose_cutoff(p_valid: np.ndarray, net_valid: np.ndarray, q_grid, min_n: int):
    """Top-quantile cut-off with the best mean net R on the inner validation block. None if none qualifies."""
    best = None
    for q in q_grid:
        thr = float(np.quantile(p_valid, 1.0 - q))
        sel = p_valid >= thr
        n = int(sel.sum())
        if n < min_n:
            continue
        mean = float(net_valid[sel].mean())
        if best is None or mean > best["inner_mean_R"]:
            best = {"q": float(q), "thr": thr, "inner_n": n, "inner_mean_R": mean}
    if best is None or best["inner_mean_R"] <= 0.0:
        return None
    return best


# ---------------------------------------------------------------- folds

def _fold_trades(samples, te: np.ndarray, cand: dict, probs: dict, fold: int) -> list[dict]:
    """One position at a time within the test block. Higher probability wins when both sides pass."""
    trades: list[dict] = []
    busy = -1
    idx = samples["idx"]
    for k, pos in enumerate(te):
        c_l = cand.get("long") is not None and bool(cand["long"][k])
        c_s = cand.get("short") is not None and bool(cand["short"][k])
        if not (c_l or c_s):
            continue
        if c_l and c_s:
            name = "long" if probs["long"][k] >= probs["short"][k] else "short"
        else:
            name = "long" if c_l else "short"
        so = samples["sides"][name]
        entry_bar = int(idx[pos]) + 1
        if entry_bar <= busy:
            continue
        ex = int(so["exit_bar"][pos])
        trades.append({
            "fold": fold,
            "side": name,
            "decision_bar": int(idx[pos]),
            "entry_bar": entry_bar,
            "exit_bar": ex,
            "outcome": int(so["outcome"][pos]),
            "gross": float(so["gross"][pos]),
            "net_taker": float(so["net_taker"][pos]),
            "net_maker": float(so["net_maker"][pos]),
            "p": float(probs[name][k]),
        })
        busy = ex
    return trades


def run_fold(samples, tr: np.ndarray, te: np.ndarray, horizon: int, cfg: dict, fold: int) -> dict:
    """Fit on ``tr`` (already purged of ``te``), choose cut-offs inside ``tr``, predict ``te``."""
    X = samples["X"]
    exit_any = samples["exit_any"]
    idx = samples["idx"]
    mcfg = cfg["model"]
    q_grid = cfg["q_grid"]
    min_inner = int(cfg.get("min_inner", 30))
    cut_in = int(round(tr.shape[0] * float(cfg.get("inner_frac", 0.8))))
    inner_tr_raw, inner_va = tr[:cut_in], tr[cut_in:]
    info = {"fold": fold, "train": int(tr.shape[0]), "test": int(te.shape[0]), "sides": {}}
    if inner_va.shape[0] < min_inner or inner_tr_raw.shape[0] < 50:
        info["skipped"] = "训练段太短，无法内部选阈值"
        return {"info": info, "trades": []}
    first_va_bar = int(idx[inner_va[0]])
    inner_tr = inner_tr_raw[exit_any[inner_tr_raw] < first_va_bar - horizon]

    cand: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    for name, so in samples["sides"].items():
        y = so["y"]
        gate = so["mask"]  # candidate gate: only gated rows are trained on and can trade
        tr_s = tr[gate[tr]]
        inner_tr_s = inner_tr[gate[inner_tr]]
        inner_va_s = inner_va[gate[inner_va]]
        cutoff = None
        if inner_va_s.shape[0] >= min_inner and inner_tr_s.shape[0] >= 50:
            m_in = fit_binary(X[inner_tr_s], y[inner_tr_s], mcfg)
            if m_in is not None:
                p_iv = m_in.predict_proba(X[inner_va_s])[:, 1]
                cutoff = choose_cutoff(p_iv, so["net_taker"][inner_va_s], q_grid, min_inner)
        if cutoff is None:
            info["sides"][name] = {"enabled": False, "reason": "内部选阈值失败（候选少，或内部均值不为正）"}
            continue
        m_full = fit_binary(X[tr_s], y[tr_s], mcfg)
        if m_full is None:
            info["sides"][name] = {"enabled": False, "reason": "训练段只有一类"}
            continue
        p_te = m_full.predict_proba(X[te])[:, 1]
        cand[name] = (p_te >= cutoff["thr"]) & gate[te]
        probs[name] = p_te
        info["sides"][name] = {"enabled": True, **cutoff, "test_candidates": int(cand[name].sum()),
                               "test_gate_rows": int(gate[te].sum())}

    trades = []
    if probs:
        for name in ("long", "short"):
            if name not in probs:
                cand[name] = np.zeros(te.shape[0], dtype=bool)
                probs[name] = np.zeros(te.shape[0])
        trades = _fold_trades(samples, te, cand, probs, fold)
    return {"info": info, "trades": trades}


# ---------------------------------------------------------------- metrics

def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    net_t = np.array([t["net_taker"] for t in trades])
    net_m = np.array([t["net_maker"] for t in trades])
    wins = net_t[net_t > 0].sum()
    losses = -net_t[net_t <= 0].sum()
    cum = np.cumsum(net_t)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = float(np.min(cum - peak))
    return {
        "trades": n,
        "long": sum(1 for t in trades if t["side"] == "long"),
        "short": sum(1 for t in trades if t["side"] == "short"),
        "win_rate": float(np.mean(net_t > 0)),
        "mean_net_R_taker": float(net_t.mean()),
        "mean_net_R_maker_entry": float(net_m.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "max_drawdown_R": dd,
        "sum_net_R_taker": float(net_t.sum()),
        "target_hit_rate": float(np.mean([t["outcome"] == 1 for t in trades])),
        "stop_rate": float(np.mean([t["outcome"] == -1 for t in trades])),
    }


def check_targets(interval: str, pooled: dict, folds: list[dict], n_folds: int) -> dict:
    need_trades = 30 if interval == "4h" else 100
    pos_folds = sum(1 for f in folds if f.get("mean_net_R_taker") is not None and f["mean_net_R_taker"] > 0)
    need_pos = int(np.ceil(0.8 * n_folds))
    pf = pooled.get("profit_factor")
    checks = {
        "folds_positive": {"value": pos_folds, "need": f">= {need_pos} of {n_folds}", "ok": pos_folds >= need_pos},
        "pooled_mean_R": {"value": pooled.get("mean_net_R_taker"), "need": ">= 0.05",
                          "ok": (pooled.get("mean_net_R_taker") or -1) >= 0.05},
        "trades": {"value": pooled.get("trades", 0), "need": f">= {need_trades}", "ok": pooled.get("trades", 0) >= need_trades},
        "profit_factor": {"value": pf, "need": ">= 1.1", "ok": pf is not None and pf >= 1.1},
    }
    checks["all_ok"] = all(v["ok"] for k, v in checks.items() if k != "all_ok")
    return checks


# ---------------------------------------------------------------- drivers

def _side_note(info: dict) -> str:
    parts = []
    for name, v in info["sides"].items():
        parts.append(f"{name}:启用" if v.get("enabled") else f"{name}:停用")
    return "；".join(parts) or info.get("skipped", "无")


def walk_forward(samples, horizon: int, cfg: dict, cv_end: int, interval: str) -> dict:
    n_folds = int(cfg.get("n_folds", 5))
    idx = samples["idx"][:cv_end]
    exit_any = samples["exit_any"][:cv_end]
    folds_out, trades_all, fold_records = [], [], []
    t0 = time.time()
    for fold in splits.iter_folds(idx, exit_any, mode="walk_forward", valid_frac=0.25,
                                  n_folds=n_folds, embargo_bars=horizon):
        res = run_fold(samples, fold.train, fold.test, horizon, cfg, fold.number)
        tr = res["trades"]
        s = summarize(tr)
        rec = {**res["info"], "trades": s["trades"], "mean_net_R_taker": s.get("mean_net_R_taker"),
               "mean_net_R_maker_entry": s.get("mean_net_R_maker_entry"), "win_rate": s.get("win_rate"),
               "test_start_utc8": _fmt(int(samples["entry_time"][fold.test[0]])),
               "test_end_utc8": _fmt(int(samples["entry_time"][fold.test[-1]]))}
        folds_out.append(rec)
        trades_all.extend(tr)
        _log(f"  {interval} 第 {fold.number} 折 训练 {fold.train.shape[0]} 测试 {fold.test.shape[0]} "
             f"交易 {s['trades']} 净期望 {_fmtR(s.get('mean_net_R_taker'))} "
             f"（{_side_note(res['info'])}）")
    _log(f"  {interval} walk-forward 用时 {time.time() - t0:.0f}s")
    pooled = summarize(trades_all)
    return {"folds": folds_out, "pooled": pooled, "trades": trades_all,
            "targets": check_targets(interval, pooled, folds_out, n_folds)}


def holdout(samples, horizon: int, cfg: dict, cv_end: int, interval: str) -> dict:
    """The final check: fit on the CV region only, score the newest block once."""
    m = samples["idx"].shape[0]
    tr_all = np.arange(cv_end, dtype=np.int64)
    te = np.arange(cv_end, m, dtype=np.int64)
    first_te = int(samples["idx"][cv_end])
    tr = tr_all[samples["exit_any"][:cv_end] < first_te - horizon]
    res = run_fold(samples, tr, te, horizon, cfg, 0)
    s = summarize(res["trades"])
    return {"info": res["info"], "pooled": s, "trades": res["trades"],
            "test_start_utc8": _fmt(int(samples["entry_time"][cv_end]))}


def _fmt(ms: int) -> str:
    from supere.timeutil import UTC8

    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")


def _fmtR(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}R"


def run_interval(db: Path, cfg: dict, interval: str, snapshot_ms: int, *, final: bool) -> dict:
    horizon = int(cfg.get("horizon", {}).get(interval, DEFAULT_HORIZON[interval]))
    cv_frac = float(cfg.get("cv_frac", 0.85))
    out: dict = {"interval": interval, "horizon": horizon, "variants": {}}
    _log(f"{interval} 读取数据（快照 {_fmt(snapshot_ms)} 之前，最多 {cfg.get('max_bars')} 根）")
    bars, funding, metrics, higher = load_inputs(db, cfg, interval, snapshot_ms)
    _log(f"{interval} K 线 {bars.n} 根；更高周期 {'无' if higher is None else higher.n}；资金费 {funding.time.shape[0]}；metrics {metrics.ts.shape[0]}")

    for tp in cfg["tp_r"]:
        key = f"tp{tp:g}R"
        _log(f"{interval} 变体 {key}：构建样本与标签")
        samples = build_samples(bars, funding, metrics, higher, interval, horizon, float(tp), cfg)
        m = int(samples["idx"].shape[0])
        cv_end = int(round(m * cv_frac))
        _log(f"{interval} {key} 样本 {m}，CV 区间 {cv_end}，holdout {m - cv_end}；特征 {len(samples['names'])}；"
             f"标签 {samples['label_stats']}")
        v = {"samples": m, "cv_samples": cv_end, "holdout_samples": m - cv_end,
             "features": len(samples["names"]), "label_stats": samples["label_stats"],
             "baseline_all_decisions_mean_R": {
                 nm: float(np.mean(so["net_taker"][:cv_end])) for nm, so in samples["sides"].items()}}
        wf = walk_forward(samples, horizon, cfg, cv_end, interval)
        v["walk_forward"] = {"folds": wf["folds"], "pooled": wf["pooled"], "targets": wf["targets"]}
        _log(f"{interval} {key} 合并：交易 {wf['pooled'].get('trades')} 净期望 "
             f"{_fmtR(wf['pooled'].get('mean_net_R_taker'))} 达标 {wf['targets']['all_ok']}")
        if final:
            ho = holdout(samples, horizon, cfg, cv_end, interval)
            v["holdout"] = {"test_start_utc8": ho["test_start_utc8"], "info": ho["info"], "pooled": ho["pooled"]}
            _log(f"{interval} {key} holdout（只此一次）：交易 {ho['pooled'].get('trades')} "
                 f"净期望 {_fmtR(ho['pooled'].get('mean_net_R_taker'))}")
        out["variants"][key] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m supere.train.iterate")
    ap.add_argument("--config", required=True, help="round config JSON")
    ap.add_argument("--db", required=True, help="superE.sqlite（只读）")
    ap.add_argument("--out", required=True, help="结果 JSON 路径")
    ap.add_argument("--intervals", help="逗号分隔，覆盖配置里的周期")
    ap.add_argument("--final", action="store_true", help="计算最终 holdout（只在最后一次使用）")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    snapshot_ms = int(cfg["snapshot_ms"])
    intervals = args.intervals.split(",") if args.intervals else cfg["intervals"]
    t0 = time.time()
    results = {"round": cfg.get("round"), "config": cfg, "snapshot_utc8": _fmt(snapshot_ms),
               "final_holdout_computed": bool(args.final), "note": HOLDOUT_NOTE, "intervals": {}}
    for interval in intervals:
        results["intervals"][interval] = run_interval(Path(args.db), cfg, interval, snapshot_ms, final=args.final)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(_strict(results), ensure_ascii=False, indent=2, allow_nan=False),
                                  encoding="utf-8")
    results["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(_strict(results), ensure_ascii=False, indent=2, allow_nan=False),
                              encoding="utf-8")
    _log(f"完成，写入 {args.out}，用时 {results['seconds']}s")
    return 0


def _strict(obj):
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    sys.exit(main())
