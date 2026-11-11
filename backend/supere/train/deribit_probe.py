"""Deribit free-public historical IV availability probe (stage 2A).

Hard rules implemented here:

* production host only: ``https://www.deribit.com/api/v2`` (the response
  ``testnet`` flag is recorded and must be false);
* public market-data endpoints only; no login, no private endpoint, no key;
* every request goes through the fixed proxy (default ``http://127.0.0.1:7890``);
  a failure is retried through the same proxy, never directly;
* serial, <= 0.5 req/s, <= 100 requests for the whole probe, connect/read
  timeout 10/30 s, 429/5xx retried at most 3 times with exponential backoff;
* small samples only: no full chain traversal, no multi-year backfill.

Outputs (all under the explicit v6 output tree): per-request JSON samples,
``requests.csv``, ``availability.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx

from supere.train import v6proto

BASE = "https://www.deribit.com/api/v2"
DEFAULT_PROXY = "http://127.0.0.1:7890"
MAX_REQUESTS = 100
MIN_INTERVAL_S = 2.05  # <= 0.5 req/s
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3

DOC_URLS = {
    "options_data_collection": "https://docs.deribit.com/articles/options-data-collection-best-practices",
    "get_instruments": "https://docs.deribit.com/api-reference/market-data/public-get_instruments",
    "ticker": "https://docs.deribit.com/api-reference/market-data/public-ticker",
    "dvol": "https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data",
    "mark_price_history": "https://docs.deribit.com/api-reference/market-data/public-get_mark_price_history",
    "last_trades": "https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_instrument_and_time",
}


# ------------------------------------------------------------------- helpers

def parse_instrument_name(name: str) -> dict:
    """Parse ``ETH-27DEC24-3000-C``; returns empty dicts on unknown shapes."""
    parts = str(name).split("-")
    if len(parts) != 4:
        return {}
    out = {"currency": parts[0], "expiry_token": parts[1], "strike": None,
           "option_type": parts[3]}
    try:
        out["strike"] = float(parts[2])
    except ValueError:
        pass
    try:
        out["expiry_utc"] = datetime.strptime(parts[1], "%d%b%y").replace(
            tzinfo=timezone.utc).isoformat()
    except ValueError:
        out["expiry_utc"] = None
    return out


def iv_null_rate(rows: list[dict], key: str = "iv") -> dict:
    seen = 0
    present = 0
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            continue
        seen += 1
        if row.get(key) not in (None, "", 0):
            present += 1
    return {"n": seen, "non_null": present,
            "null_rate": (None if seen == 0 else round(1.0 - present / seen, 4))}


def infer_availability(probe: dict) -> dict:
    """Turn the measured endpoints into an A/B/C matrix (no optimistic claims)."""
    quotes_now = probe.get("book_summary", {})
    hist_trades = probe.get("historical_trades", {})
    recent_expired = probe.get("recent_expired_trades", []) or []
    old_wide = probe.get("old_wide_window", {}) or {}
    old_meta = probe.get("old_metadata", {})
    mark_hist = probe.get("mark_price_history", {}) or {}
    dvol = probe.get("dvol", {}) or {}
    chain = probe.get("instruments", {})
    dvol_rows = sum(int(v.get("rows") or 0) for v in dvol.values() if isinstance(v, dict))
    mark_rows = sum(int(v.get("rows") or 0) for v in mark_hist.values() if isinstance(v, dict))
    expired_trade_rows = sum(int(x.get("rows") or 0) for x in recent_expired)
    trade_rows = int(hist_trades.get("n") or 0) + expired_trade_rows + int(old_wide.get("rows") or 0)

    def verdict(quoted_iv_12m: bool, trade_iv: bool, meta_12m: bool) -> str:
        if quoted_iv_12m and meta_12m:
            return "A"
        if trade_iv:
            return "B"
        return "C"

    items = []
    items.append({
        "item": "current option chain metadata",
        "endpoint": "public/get_instruments",
        "available": bool(chain.get("current", {}).get("n")),
        "detail": chain.get("current", {}),
        "class": "A" if chain.get("current", {}).get("n") else "C",
    })
    items.append({
        "item": "delivered/expired instrument catalog",
        "endpoint": "public/get_instruments?expired=true",
        "available": bool(chain.get("expired", {}).get("n")),
        "detail": chain.get("expired", {}),
        "class": ("B" if chain.get("expired", {}).get("n") else "C"),
        "note": "recent delivered only; not a multi-year archive (see the measured expiry span)",
    })
    items.append({
        "item": "current full-chain mark IV snapshot",
        "endpoint": "public/get_book_summary_by_currency",
        "available": bool(quotes_now.get("n")),
        "detail": quotes_now,
        "class": "A" if quotes_now.get("n") else "C",
        "note": "current snapshot only; not historical",
    })
    items.append({
        "item": "historical quoted IV snapshot",
        "endpoint": "none found in public API",
        "available": False,
        "detail": {},
        "class": "C",
        "note": "ticker/book_summary are current-only; mark_price_history returns price, not IV",
    })
    items.append({
        "item": "historical options trade IV",
        "endpoint": "public/get_last_trades_by_instrument_and_time",
        "available": trade_rows > 0,
        "detail": {"rows": trade_rows, "old_constructed": int(hist_trades.get("n") or 0),
                   "recent_expired": expired_trade_rows,
                   "old_wide_window": int(old_wide.get("rows") or 0)},
        "class": verdict(False, trade_rows > 0, bool(old_meta.get("n"))),
        "note": "trade IV is an irregular proxy, not a continuous quote surface",
    })
    items.append({
        "item": "historical option metadata (older contracts)",
        "endpoint": "public/get_instrument",
        "available": bool(old_meta.get("n")),
        "detail": old_meta,
        "class": "B" if old_meta.get("n") else "C",
    })
    items.append({
        "item": "historical mark price (DVOL options)",
        "endpoint": "public/get_mark_price_history",
        "available": mark_rows > 0,
        "detail": {"rows": mark_rows, "per_tenor": mark_hist},
        "class": "B" if mark_rows > 0 else "C",
        "note": "mark price only; no IV/forward, cannot rebuild a full surface",
    })
    items.append({
        "item": "DVOL index history",
        "endpoint": "public/get_volatility_index_data",
        "available": dvol_rows > 0,
        "detail": {"rows_first_page": dvol_rows, "per_window": dvol},
        "class": "A" if dvol_rows > 0 else "C",
        "note": "aggregate index only; not ATM/skew/term structure and not new data",
    })
    overall = verdict(
        quoted_iv_12m=any(i["item"] == "historical quoted IV snapshot" and i["class"] == "A" for i in items),
        trade_iv=trade_rows > 0,
        meta_12m=bool(old_meta.get("n")),
    )
    return {"items": items, "overall": overall}


# ------------------------------------------------------------------- client

@dataclass
class Budget:
    max_requests: int = MAX_REQUESTS
    used: int = 0

    def take(self) -> None:
        if self.used >= self.max_requests:
            raise RuntimeError(f"request budget exhausted ({self.max_requests})")
        self.used += 1


class ProbeClient:
    def __init__(self, out_dir: Path, proxy: str = DEFAULT_PROXY,
                 max_requests: int = MAX_REQUESTS):
        self.out_dir = Path(out_dir)
        (self.out_dir / "raw").mkdir(parents=True, exist_ok=True)
        self.proxy = proxy
        self.budget = Budget(max_requests)
        self.records: list[dict] = []
        self._last_start = 0.0
        self._client = httpx.Client(
            proxy=proxy, trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            headers={"User-Agent": "superE-v6-research-probe/1.0"},
        )

    def close(self) -> None:
        self._client.close()

    def _wait(self) -> None:
        now = time.monotonic()
        wait = MIN_INTERVAL_S - (now - self._last_start)
        if wait > 0:
            time.sleep(wait)
        self._last_start = time.monotonic()

    def call(self, endpoint: str, params: dict | None = None) -> dict | None:
        params = params or {}
        url = f"{BASE}/{endpoint}"
        attempt = 0
        while True:
            attempt += 1
            self.budget.take()
            self._wait()
            t0 = time.monotonic()
            status = None
            text = ""
            err = None
            try:
                resp = self._client.get(url, params=params)
                status = resp.status_code
                text = resp.text
            except Exception as exc:  # network/timeout: retried through the same proxy
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
            latency = (time.monotonic() - t0) * 1000.0
            body = None
            if text:
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    body = None
            api_error = None
            if isinstance(body, dict) and body.get("error"):
                api_error = body["error"]
            retryable = (
                err is not None
                or (status in RETRY_STATUS)
                or (isinstance(api_error, dict) and api_error.get("code") in RETRY_STATUS)
            )
            rec = {
                "seq": self.budget.used,
                "ts_utc8": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint": endpoint,
                "params": _sanitize(params),
                "attempt": attempt,
                "http_status": status,
                "api_error_code": (api_error or {}).get("code") if isinstance(api_error, dict) else None,
                "api_error_message": (api_error or {}).get("message") if isinstance(api_error, dict) else None,
                "network_error": err,
                "latency_ms": round(latency, 1),
                "bytes": len(text.encode("utf-8")) if text else 0,
                "testnet": bool(body.get("testnet")) if isinstance(body, dict) else None,
                "usIn": body.get("usIn") if isinstance(body, dict) else None,
                "usOut": body.get("usOut") if isinstance(body, dict) else None,
                "usDiff": body.get("usDiff") if isinstance(body, dict) else None,
            }
            result = body.get("result") if isinstance(body, dict) else None
            rows = _row_count(result)
            rec["result_rows"] = rows
            rec["has_more"] = _has_more(result)
            if attempt > 1 and retryable and attempt <= MAX_RETRIES + 1:
                time.sleep(min(30.0, 2.0 ** attempt))
            if retryable and attempt <= MAX_RETRIES:
                self.records.append(rec)
                continue
            path = self.out_dir / "raw" / f"{self.budget.used:03d}_{endpoint.replace('/', '_')}.json"
            payload = {"request": {"endpoint": endpoint, "params": rec["params"]},
                       "meta": rec,
                       "result": result}
            text_out = json.dumps(payload, ensure_ascii=False)
            if len(text_out) > 3_000_000:
                payload["result"] = _compact(result)
                text_out = json.dumps(payload, ensure_ascii=False)
            path.write_text(text_out, encoding="utf-8")
            rec["file"] = str(path)
            rec["sha256"] = hashlib.sha256(text_out.encode("utf-8")).hexdigest()
            self.records.append(rec)
            if retryable:
                return None
            if api_error is not None or status is None or (status and status >= 400):
                return None
            return result

    def write_log(self) -> Path:
        path = self.out_dir / "requests.csv"
        cols = ["seq", "ts_utc8", "endpoint", "params", "attempt", "http_status",
                "api_error_code", "api_error_message", "network_error", "latency_ms",
                "bytes", "testnet", "usIn", "usOut", "usDiff", "result_rows",
                "has_more", "file", "sha256"]
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for rec in self.records:
                w.writerow(rec)
        return path


def _sanitize(params: dict) -> dict:
    # No credentials are ever used; keep params small and loggable.
    return {k: v for k, v in params.items()}


def _row_count(result) -> int | None:
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("data", "trades", "timestamp", "price", "instruments"):
            if key in result and isinstance(result[key], list):
                return len(result[key])
        return 1
    return 0


def _has_more(result) -> bool | None:
    if isinstance(result, dict):
        if "has_more" in result:
            return bool(result["has_more"])
        if "continuation" in result:
            return result["continuation"] not in (None, "")
    return None


def _compact(result):
    if isinstance(result, list):
        return {"n": len(result), "first": result[:5], "last": result[-5:]}
    return result


# --------------------------------------------------------------- probe plan

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def select_current(chain: list[dict], underlying: float | None, now_ms: int) -> dict:
    """Pick a small set of current instruments around 1d/30d/180d, ATM if possible."""
    live = []
    for inst in chain:
        exp = inst.get("expiration_timestamp")
        if not exp or exp <= now_ms:
            continue
        try:
            strike = float(inst.get("strike"))
        except (TypeError, ValueError):
            continue
        live.append({"name": inst.get("instrument_name"), "expiry": int(exp),
                     "strike": strike, "type": inst.get("option_type", "")})
    if not live:
        return {}
    expiries = sorted({x["expiry"] for x in live})
    targets = {"1d": now_ms + 86_400_000, "30d": now_ms + 30 * 86_400_000,
               "180d": now_ms + 180 * 86_400_000}
    chosen = {}
    for label, want in targets.items():
        exp = min(expiries, key=lambda e: abs(e - want))
        pool = [x for x in live if x["expiry"] == exp]
        if underlying:
            pool.sort(key=lambda x: abs(x["strike"] - underlying))
        calls = [x for x in pool if x["type"].lower() == "call"]
        puts = [x for x in pool if x["type"].lower() == "put"]
        chosen[label] = {
            "expiry": exp,
            "expiry_utc": datetime.fromtimestamp(exp / 1000, tz=timezone.utc).isoformat(),
            "call": calls[0]["name"] if calls else None,
            "put": puts[0]["name"] if puts else None,
            "strike": pool[0]["strike"] if pool else None,
        }
    return chosen


def probe(out_dir: Path, proxy: str = DEFAULT_PROXY) -> dict:
    out_dir = v6proto.assert_experiment_output(Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    client = ProbeClient(out_dir, proxy=proxy)
    now_ms = int(time.time() * 1000)
    result: dict = {"base": BASE, "proxy": proxy, "budget": MAX_REQUESTS,
                    "doc_urls": DOC_URLS, "started_utc8": datetime.now().astimezone().isoformat()}
    try:
        t = client.call("public/get_time")
        result["server_time"] = t
        if isinstance(t, int):
            result["clock_offset_ms"] = int(time.time() * 1000) - t

        cur = client.call("public/get_instruments",
                          {"currency": "ETH", "kind": "option", "expired": "false"})
        exp = client.call("public/get_instruments",
                          {"currency": "ETH", "kind": "option", "expired": "true"})
        result["instruments"] = {
            "current": _chain_summary(cur),
            "expired": _chain_summary(exp),
        }
        summary = client.call("public/get_book_summary_by_currency",
                              {"currency": "ETH", "kind": "option"})
        rows = summary if isinstance(summary, list) else []
        underlying = None
        for row in rows:
            if row.get("underlying_price"):
                underlying = float(row["underlying_price"])
                break
        result["book_summary"] = {
            "n": len(rows),
            "underlying_price": underlying,
            "iv": iv_null_rate(rows, "mark_iv"),
            "bid_iv_present": sum(1 for r in rows if r.get("bid_price")),
            "ask_iv_present": sum(1 for r in rows if r.get("ask_price")),
            "timestamp_range": _ts_range([r.get("creation_timestamp") for r in rows]),
        }
        current = select_current(cur if isinstance(cur, list) else [], underlying, now_ms)
        result["selected"] = current

        # Current tickers (1d call/put, 30d call, 180d call).
        picks = [current.get("1d", {}).get("call"), current.get("1d", {}).get("put"),
                 current.get("30d", {}).get("call"), current.get("180d", {}).get("call")]
        tickers = []
        for name in [p for p in picks if p]:
            tk = client.call("public/ticker", {"instrument_name": name})
            if isinstance(tk, dict):
                tickers.append({k: tk.get(k) for k in
                                ("instrument_name", "timestamp", "mark_iv", "bid_iv",
                                 "ask_iv", "underlying_price", "open_interest",
                                 "mark_price", "index_price")})
        result["tickers"] = {"n": len(tickers), "rows": tickers,
                             "iv": iv_null_rate(tickers, "mark_iv")}

        # DVOL: 1d at 60 s, 30d and 180d at 1 h.
        dvol = {}
        for label, days, res in (("1d", 1, 60), ("30d", 30, 3600), ("180d", 180, 3600)):
            start = now_ms - days * 86_400_000
            r = client.call("public/get_volatility_index_data",
                            {"currency": "ETH", "start_timestamp": start,
                             "end_timestamp": now_ms, "resolution": str(res)})
            dvol[label] = _series_summary(r)
        result["dvol"] = dvol

        # Mark price history for two current instruments (last 7 days).
        mark = {}
        for label in ("1d", "30d"):
            name = current.get(label, {}).get("call")
            if not name:
                continue
            r = client.call("public/get_mark_price_history",
                            {"instrument_name": name,
                             "start_timestamp": now_ms - 7 * 86_400_000,
                             "end_timestamp": now_ms})
            mark[label] = _series_summary(r, price_key="price")
            mark[label]["instrument_name"] = name
        result["mark_price_history"] = mark

        # A handful of constructed older quarterly names (not a mass guess):
        # ETH options use ETH-<DDMMMYY>-<strike>-<C|P> with 08:00 UTC expiry.
        old_names = ["ETH-28JUN24-3000-C", "ETH-29DEC23-2000-C",
                     "ETH-30JUN23-2000-P", "ETH-31MAR23-1500-C",
                     "ETH-24JUN22-2000-C"]
        old_meta = {"n": 0, "attempts": []}
        old_trades = {"n": 0, "per_instrument": []}
        for name in old_names:
            inst = client.call("public/get_instrument", {"instrument_name": name})
            if isinstance(inst, dict):
                old_meta["n"] += 1
                old_meta["attempts"].append({
                    "instrument_name": name, "ok": True,
                    "creation_timestamp": inst.get("creation_timestamp"),
                    "expiration_timestamp": inst.get("expiration_timestamp"),
                    "strike": inst.get("strike"), "option_type": inst.get("option_type"),
                })
                exp_ms = int(inst.get("expiration_timestamp") or 0)
                start = exp_ms - 7 * 86_400_000
                end = exp_ms + 2 * 86_400_000
                r = client.call("public/get_last_trades_by_instrument_and_time",
                                {"instrument_name": name, "start_timestamp": start,
                                 "end_timestamp": end, "count": 1000})
                rows = r.get("trades", []) if isinstance(r, dict) else (r or [])
                entry = {
                    "instrument_name": name, "rows": len(rows),
                    "iv": iv_null_rate(rows),
                    "first_ts": min((x.get("timestamp", 0) for x in rows), default=None),
                    "last_ts": max((x.get("timestamp", 0) for x in rows), default=None),
                    "has_more": r.get("has_more") if isinstance(r, dict) else None,
                }
                if isinstance(r, dict) and r.get("has_more") and r.get("continuation"):
                    r2 = client.call("public/get_last_trades_by_instrument_and_time",
                                     {"instrument_name": name, "start_timestamp": start,
                                      "end_timestamp": end, "count": 1000,
                                      "continuation": r["continuation"]})
                    rows2 = r2.get("trades", []) if isinstance(r2, dict) else (r2 or [])
                    entry["page2_rows"] = len(rows2)
                    old_trades["n"] += len(rows2)
                old_trades["per_instrument"].append(entry)
                old_trades["n"] += len(rows)
            else:
                old_meta["attempts"].append({"instrument_name": name, "ok": False})
        result["old_metadata"] = old_meta
        result["historical_trades"] = old_trades

        # Recently delivered contracts: are their trades still queryable?  This
        # separates "our constructed strike never traded" from "expired trades
        # are archived".
        recent = []
        expired_chain = exp if isinstance(exp, list) else []
        candidates = []
        for c in expired_chain:
            try:
                strike = float(c.get("strike") or 0)
                exp_ts = int(c.get("expiration_timestamp") or 0)
            except (TypeError, ValueError):
                continue
            if exp_ts and underlying:
                candidates.append((abs(strike - underlying), c.get("instrument_name"), exp_ts))
        candidates.sort()
        for _dist, name, exp_ts in candidates[:2]:
            r = client.call("public/get_last_trades_by_instrument_and_time",
                            {"instrument_name": name,
                             "start_timestamp": exp_ts - 3 * 86_400_000,
                             "end_timestamp": exp_ts, "count": 1000})
            rows = r.get("trades", []) if isinstance(r, dict) else (r or [])
            recent.append({"instrument_name": name, "rows": len(rows),
                           "iv": iv_null_rate(rows),
                           "has_more": r.get("has_more") if isinstance(r, dict) else None})
        result["recent_expired_trades"] = recent
        # One wide-window check on the first existing old contract, to rule out
        # a too-narrow ±7 day window as the reason for zero rows.
        first_old = next((a for a in old_meta["attempts"] if a.get("ok")), None)
        if first_old and first_old.get("expiration_timestamp"):
            exp_ts = int(first_old["expiration_timestamp"])
            r = client.call("public/get_last_trades_by_instrument_and_time",
                            {"instrument_name": first_old["instrument_name"],
                             "start_timestamp": exp_ts - 60 * 86_400_000,
                             "end_timestamp": exp_ts,
                             "count": 1000})
            rows = r.get("trades", []) if isinstance(r, dict) else (r or [])
            result["old_wide_window"] = {
                "instrument_name": first_old["instrument_name"],
                "window_days": 60,
                "rows": len(rows),
                "iv": iv_null_rate(rows),
                "has_more": r.get("has_more") if isinstance(r, dict) else None,
            }

        # Current trade tape for one ATM contract, including pagination.
        name = current.get("1d", {}).get("call")
        cur_trades = {"rows": 0}
        if name:
            r = client.call("public/get_last_trades_by_instrument_and_time",
                            {"instrument_name": name,
                             "start_timestamp": now_ms - 7 * 86_400_000,
                             "end_timestamp": now_ms, "count": 1000})
            rows = r.get("trades", []) if isinstance(r, dict) else (r or [])
            cur_trades = {"instrument_name": name, "rows": len(rows),
                          "iv": iv_null_rate(rows),
                          "has_more": r.get("has_more") if isinstance(r, dict) else None}
            if isinstance(r, dict) and r.get("has_more") and r.get("continuation"):
                r2 = client.call("public/get_last_trades_by_instrument_and_time",
                                 {"instrument_name": name,
                                  "start_timestamp": now_ms - 7 * 86_400_000,
                                  "end_timestamp": now_ms, "count": 1000,
                                  "continuation": r["continuation"]})
                rows2 = r2.get("trades", []) if isinstance(r2, dict) else (r2 or [])
                cur_trades["page2_rows"] = len(rows2)
        result["current_trades"] = cur_trades

        result["requests_used"] = client.budget.used
        log = client.write_log()
        result["requests_csv"] = str(log)
        result["availability"] = infer_availability(result)
        result["finished_utc8"] = datetime.now().astimezone().isoformat()
    finally:
        client.close()
    (out_dir / "availability.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def _chain_summary(chain) -> dict:
    if not isinstance(chain, list) or not chain:
        return {"n": 0}
    exps = sorted({int(c["expiration_timestamp"]) for c in chain
                   if c.get("expiration_timestamp")})
    creations = sorted({int(c["creation_timestamp"]) for c in chain
                        if c.get("creation_timestamp")})
    return {
        "n": len(chain),
        "expiries": len(exps),
        "first_expiry_utc": datetime.fromtimestamp(exps[0] / 1000, tz=timezone.utc).isoformat() if exps else None,
        "last_expiry_utc": datetime.fromtimestamp(exps[-1] / 1000, tz=timezone.utc).isoformat() if exps else None,
        "first_creation_utc": datetime.fromtimestamp(creations[0] / 1000, tz=timezone.utc).isoformat() if creations else None,
        "last_creation_utc": datetime.fromtimestamp(creations[-1] / 1000, tz=timezone.utc).isoformat() if creations else None,
        "sample": [c.get("instrument_name") for c in chain[:5]],
    }


def _series_summary(r, price_key: str = "price") -> dict:
    if isinstance(r, list):
        if r and isinstance(r[0], (list, tuple)):
            return {"rows": len(r), "first_ts": r[0][0], "last_ts": r[-1][0],
                    "has_more": None, "continuation": False,
                    "fields": ["[timestamp, price]"]}
        return {"rows": len(r), "first_ts": None, "last_ts": None,
                "has_more": None, "continuation": False, "fields": []}
    if not isinstance(r, dict):
        return {"rows": 0}
    ts = r.get("timestamp") or r.get("ticks") or []
    data = r.get("data")
    price = r.get(price_key) or []
    if isinstance(data, list) and data:
        rows = len(data)
    else:
        rows = max(len(ts), len(price))
    first = last = None
    if isinstance(data, list) and data:
        first, last = data[0][0], data[-1][0]
    elif ts:
        first, last = ts[0], ts[-1]
    return {"rows": rows, "first_ts": first, "last_ts": last,
            "has_more": r.get("has_more"),
            "continuation": bool(r.get("continuation")),
            "fields": sorted(r.keys())}


def _ts_range(values) -> dict:
    vals = [int(v) for v in values if v]
    if not vals:
        return {}
    return {"first": min(vals), "last": max(vals)}


def _report(result: dict) -> str:
    av = result.get("availability", {})
    lines = ["# Deribit free-public IV availability probe (stage 2A)", "",
             f"- base: {result.get('base')}  proxy: {result.get('proxy')}",
             f"- requests used: {result.get('requests_used')} / {result.get('budget')}",
             f"- clock offset (local - server): {result.get('clock_offset_ms')} ms",
             f"- overall availability class: **{av.get('overall')}**", "",
             "## Measured items", "",
             "| item | endpoint | available | class | detail |",
             "|---|---|---|---|---|"]
    for item in av.get("items", []):
        detail = item.get("detail", {})
        brief = ", ".join(f"{k}={v}" for k, v in list(detail.items())[:6] if not isinstance(v, (dict, list)))
        lines.append(f"| {item['item']} | {item['endpoint']} | {item['available']} | "
                     f"{item['class']} | {brief} |")
    lines += ["", "## Constructed older-contract metadata and trades", ""]
    for a in result.get("old_metadata", {}).get("attempts", []):
        lines.append(f"- {a['instrument_name']}: metadata_ok={a['ok']} "
                     f"creation={a.get('creation_timestamp')} expiry={a.get('expiration_timestamp')}")
    for p in result.get("historical_trades", {}).get("per_instrument", []):
        lines.append(f"- {p['instrument_name']}: rows={p['rows']} iv_null={p['iv']['null_rate']} "
                     f"first={p.get('first_ts')} last={p.get('last_ts')} has_more={p.get('has_more')}")
    lines += ["", "## Recently delivered contracts (trade availability)", ""]
    for x in result.get("recent_expired_trades", []) or []:
        lines.append(f"- {x['instrument_name']}: rows={x['rows']} iv_null={x['iv']['null_rate']} "
                     f"has_more={x.get('has_more')}")
    wide = result.get("old_wide_window")
    if wide:
        lines.append(f"- wide window ({wide['window_days']}d) on {wide['instrument_name']}: "
                     f"rows={wide['rows']} iv_null={wide['iv']['null_rate']}")
    lines += ["", "## DVOL / mark-price first pages", ""]
    for label, v in (result.get("dvol") or {}).items():
        lines.append(f"- DVOL {label}: rows={v.get('rows')} first={v.get('first_ts')} "
                     f"last={v.get('last_ts')} has_more={v.get('has_more')}")
    for label, v in (result.get("mark_price_history") or {}).items():
        lines.append(f"- mark price {label} ({v.get('instrument_name')}): rows={v.get('rows')} "
                     f"first={v.get('first_ts')} last={v.get('last_ts')}")
    lines += ["", "## Official documentation used", ""]
    for k, v in result.get("doc_urls", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Boundaries", "",
              "- free public production endpoint only; no login, no private endpoint, no key;",
              "- no full-chain traversal or multi-year backfill; sampled windows only;",
              "- field presence is not proof of complete history; this report only covers the measured windows."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deribit public IV availability probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proxy", default=DEFAULT_PROXY)
    args = ap.parse_args(argv)
    result = probe(Path(args.out), proxy=args.proxy)
    print(json.dumps({"requests_used": result["requests_used"],
                      "availability": result["availability"]["overall"],
                      "report": str(Path(args.out) / "report.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
