"""DeepSeek / rule-based scoring for news (plan v3, stage 5).

The API key is read only through :mod:`supere.secrets`; it is never printed,
logged, stored in the database, or written to a report.  When no key is
configured the scorer falls back to a deterministic keyword model and marks
``scored_by='rule_fallback'`` so downstream ablations can separate the two.

Scores are cached by ``url_hash + scorer_version + prompt_version``: a row that
already has that version is skipped.  Failures are retried and left with
``scored_by='pending'`` (or the rule fallback) rather than silently dropped.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from supere import secrets
from supere.storage.db import Database

SCORER_VERSION = "ds-v1"
PROMPT_VERSION = "p1"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
PROXY = "http://127.0.0.1:7890"

SYSTEM_PROMPT = (
    "You are a crypto-news classifier. Return JSON only, no prose. "
    "For each item rate: eth_relevance (0-1, how relevant to ETH price), "
    "direction (-1 bearish .. +1 bullish for ETH), urgency (0-1, how urgent/market-moving), "
    "magnitude (0-1, expected size of impact), event_type (short snake_case label), "
    "confidence (0-1). Do not use hindsight; classify the headline as it would read at publication."
)

POS_WORDS = {
    "surge", "rally", "soar", "gain", "bullish", "approve", "approval", "upgrade", "adopt",
    "inflow", "record", "high", "partnership", "launch", "buy", "accumulate", "etf",
    "涨", "上涨", "突破", "利好", "获批", "增持", "流入",
}
NEG_WORDS = {
    "crash", "plunge", "drop", "fall", "bearish", "hack", "exploit", "lawsuit", "ban",
    "outflow", "liquidation", "bankruptcy", "fraud", "sec", "reject", "delay", "sell",
    "stolen", "incident", "security", "breach", "attack",
    "跌", "下跌", "暴跌", "利空", "被黑", "清算", "诉讼", "禁止",
}
HIGH_WORDS = {
    "hack", "exploit", "crash", "plunge", "sec", "etf", "fed", "fomc", "cpi", "nfp",
    "liquidation", "bankruptcy", "halt", "emergency", "black swan", "war", "tariff",
    "security incident", "stolen", "breach", "attack",
    "暴跌", "被黑", "清算", "紧急", "暂停", "战争",
}
ETH_WORDS = {"eth", "ethereum", "ether", "vitalik", "staking", "beacon", "layer 2", "l2", "defi"}
EVENT_RULES = (
    ("regulation", {"sec", "lawsuit", "ban", "regulat", "court", "approve", "etf"}),
    ("security", {"hack", "exploit", "breach", "stolen", "attack"}),
    ("macro", {"fed", "fomc", "cpi", "nfp", "inflation", "rate", "tariff", "war"}),
    ("flows", {"inflow", "outflow", "etf", "liquidation", "whale"}),
    ("tech", {"upgrade", "launch", "layer 2", "l2", "fork", "staking"}),
)


@dataclass
class Score:
    eth_relevance: float
    direction: float
    urgency: float
    magnitude: float
    event_type: str
    confidence: float
    scorer: str


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def rule_score(title: str, body: str = "") -> Score:
    text = f"{title} {body}".lower()
    tokens = set(re.findall(r"[a-z0-9']+|[一-鿿]+", text))
    pos = len(tokens & POS_WORDS)
    neg = len(tokens & NEG_WORDS)
    high = len(tokens & HIGH_WORDS)
    eth = len(tokens & ETH_WORDS)
    direction = (pos - neg) / max(pos + neg, 1)
    urgency = _clip(0.30 + 0.35 * min(high, 2) + (0.15 if neg else 0.0)
                    + (0.10 if pos else 0.0), 0.0, 1.0)
    relevance = _clip(0.25 + 0.2 * eth + 0.05 * (pos + neg), 0.0, 1.0)
    magnitude = _clip(0.15 + 0.15 * (pos + neg) + 0.15 * high, 0.0, 1.0)
    event = "general"
    for name, words in EVENT_RULES:
        if any(w in text for w in words):
            event = name
            break
    return Score(eth_relevance=round(relevance, 3), direction=round(_clip(direction, -1, 1), 3),
                 urgency=round(urgency, 3), magnitude=round(magnitude, 3),
                 event_type=event, confidence=0.35, scorer="rule_fallback")


def _parse_scores(payload: dict, ids: list[int]) -> dict[int, Score]:
    out: dict[int, Score] = {}
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return out
    for row in rows:
        try:
            rid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if rid not in ids:
            continue
        try:
            out[rid] = Score(
                eth_relevance=_clip(float(row.get("eth_relevance", 0.5)), 0, 1),
                direction=_clip(float(row.get("direction", 0)), -1, 1),
                urgency=_clip(float(row.get("urgency", 0.3)), 0, 1),
                magnitude=_clip(float(row.get("magnitude", 0.3)), 0, 1),
                event_type=str(row.get("event_type", "general"))[:40],
                confidence=_clip(float(row.get("confidence", 0.5)), 0, 1),
                scorer="deepseek",
            )
        except (TypeError, ValueError):
            continue
    return out


def score_batch_deepseek(items: list[dict], key: str, *, proxy: str = PROXY,
                         retries: int = 3) -> dict[int, Score] | None:
    """One request for a batch of ``{id,title,body}``. ``None`` means the call failed."""
    ids = [int(item["id"]) for item in items]
    user = json.dumps(
        [{"id": int(item["id"]), "title": (item.get("title") or "")[:400],
          "summary": (item.get("body") or "")[:600]} for item in items],
        ensure_ascii=False,
    )
    body = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0.0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Items:\n" + user + "\nReturn {\"results\":[...]}."},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=45.0, proxy=proxy) as client:
                resp = client.post(DEEPSEEK_URL, headers=headers, json=body)
            if resp.status_code == 200:
                parsed = _parse_scores(resp.json(), ids)
                if parsed:
                    return parsed
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def _visible_at(row: dict) -> int:
    published = row.get("ts_published")
    ingested = row.get("ts_ingested") or row.get("ts_local") or 0
    return int(max(int(published or 0), int(ingested or 0)))


def _timing_quality(row: dict) -> str:
    source = (row.get("source") or "").lower()
    if source == "gdelt":
        return "proxy_gdelt_seen"
    return "ingested"


def score_pending(db_path: Path, *, limit: int = 2000, batch_size: int = 10,
                  use_deepseek: bool = True, proxy: str = PROXY,
                  rescore: bool = False) -> dict:
    """Score up to ``limit`` rows. Returns counts, never the key."""
    db = Database(Path(db_path))
    try:
        if rescore:
            rows = db.conn.execute(
                "SELECT * FROM news ORDER BY ts_ingested DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = db.unscored_news(limit)
        pending = [dict(row) for row in rows]
        key = secrets.deepseek_api_key() if use_deepseek else None
        counts = {"pending": len(pending), "deepseek": 0, "rule": 0, "failed": 0,
                  "deepseek_configured": bool(key)}
        if not pending:
            return counts
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            scored = None
            if key:
                scored = score_batch_deepseek(batch, key, proxy=proxy)
            for row in batch:
                item = scored.get(int(row["id"])) if scored else None
                if item is None:
                    item = rule_score(str(row.get("title") or ""), str(row.get("body") or ""))
                    if key:
                        counts["failed"] += 1
                    else:
                        counts["rule"] += 1
                else:
                    counts["deepseek"] += 1
                    if item.scorer != "deepseek":
                        counts["rule"] += 1
                fields = {
                    "available_at": _visible_at(row),
                    "timing_quality": _timing_quality(row),
                    "direction": int(1 if item.direction > 0.15 else (-1 if item.direction < -0.15 else 0)),
                    "direction_score": item.direction,
                    "impact": int(round(item.urgency * 3)),
                    "eth_relevance": item.eth_relevance,
                    "urgency": item.urgency,
                    "magnitude": item.magnitude,
                    "event_type": item.event_type,
                    "confidence": item.confidence,
                    "scored_by": item.scorer,
                    "score_version": f"{SCORER_VERSION}:{PROMPT_VERSION}",
                    "score_json": json.dumps(asdict(item), ensure_ascii=False),
                    "status": "scored",
                }
                db.update_news_score(int(row["id"]), fields)
        return counts
    finally:
        db.close()


def list_news(db_path: Path, *, limit: int = 50, since_ms: int | None = None) -> list[dict]:
    """Read-only view for ``/api/news``. Missing columns degrade to defaults."""
    path = Path(db_path)
    if not path.exists():
        return []
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(news)")}
        if not cols:
            return []
        avail = "COALESCE(available_at, ts_ingested)" if "available_at" in cols else "ts_ingested"
        direction = "COALESCE(direction_score, direction, 0.0)" if "direction_score" in cols else "COALESCE(direction, 0.0)"
        urgency = "COALESCE(urgency, 0.0)" if "urgency" in cols else "0.0"
        magnitude = "COALESCE(magnitude, 0.0)" if "magnitude" in cols else "0.0"
        event = "COALESCE(event_type, '')" if "event_type" in cols else "''"
        timing = "COALESCE(timing_quality, '')" if "timing_quality" in cols else "''"
        relevance = "COALESCE(eth_relevance, 0.0)"
        where = "1=1"
        args: list = []
        if since_ms is not None:
            where += f" AND {avail} >= ?"
            args.append(int(since_ms))
        sql = (f"SELECT id, {avail} AS available_at, ts_published, ts_ingested, source, title, url, "
               f"{relevance}, {direction}, {urgency}, {magnitude}, {event}, {timing}, "
               f"COALESCE(scored_by,''), COALESCE(score_version,'') "
               f"FROM news WHERE {where} ORDER BY available_at DESC, id DESC LIMIT ?")
        rows = conn.execute(sql, (*args, int(limit))).fetchall()
    finally:
        conn.close()
    keys = ["id", "available_at", "ts_published", "ts_ingested", "source", "title", "url",
            "relevance", "direction", "urgency", "magnitude", "event_type",
            "timing_quality", "scored_by", "score_version"]
    return [dict(zip(keys, row, strict=True)) for row in rows]
