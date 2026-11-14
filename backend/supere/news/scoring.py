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

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from supere import secrets
from supere.news.summary import market_channel
from supere.storage.db import Database

SCORER_VERSION = "ds-v2"
PROMPT_VERSION = "p2"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
PROXY = "http://127.0.0.1:7890"

SYSTEM_PROMPT = (
    "You are a crypto-news classifier. Return JSON only, no prose. "
    "For each item rate: eth_relevance (0-1, how relevant to ETH price), "
    "direction (-1 bearish .. +1 bullish for ETH), urgency (0-1, how urgent/market-moving), "
    "magnitude (0-1, expected size of impact), event_type (short snake_case label), "
    "confidence (0-1), summary_zh (1-2 short Chinese sentences: what happened and why it matters to ETH; "
    "conservative, no trading advice). Do not use hindsight; classify the headline as it would read at publication."
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
    summary_zh: str = ""
    label: str = ""
    reason_zh: str = ""
    importance: int = 0
    impact_zh: str = ""


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def sentiment_label(direction: float) -> str:
    d = float(direction)
    if d > 0.12:
        return "利多"
    if d < -0.12:
        return "利空"
    return "中性"


def _urgency_word(urgency: float) -> str:
    if urgency >= 0.7:
        return "高"
    if urgency >= 0.4:
        return "中"
    return "低"


EVENT_CN = {
    "regulation": "监管消息",
    "security": "安全事件",
    "macro": "宏观消息",
    "flows": "资金流消息",
    "tech": "技术/网络消息",
    "general": "市场消息",
}

EVENT_WHY = {
    "regulation": "可能改变监管预期，进而影响 ETH 的风险溢价",
    "security": "可能影响市场对交易所和 ETH 的信心",
    "macro": "可能通过利率预期和风险偏好影响 ETH",
    "flows": "可能反映资金流向变化，进而影响 ETH 的买盘",
    "tech": "可能影响以太坊网络预期与 ETH 情绪",
    "general": "可能通过市场情绪和资金流影响 ETH",
}


def content_hash(title: str, body: str = "") -> str:
    """Stable content identity used to cache scores across sources."""
    text = f"{(title or '').strip()}\n{(body or '').strip()[:1000]}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_display_fields(score: Score) -> Score:
    """Fill label, reason and importance when a scorer left them blank."""
    if score.label not in ("利多", "利空", "中性"):
        score.label = sentiment_label(score.direction)
    if not score.reason_zh:
        score.reason_zh = EVENT_WHY.get(score.event_type, EVENT_WHY["general"])
    if score.importance not in (1, 2, 3):
        if score.urgency >= 0.7:
            score.importance = 3
        elif score.urgency >= 0.45:
            score.importance = 2
        else:
            score.importance = 1
    if score.importance >= 2 and not score.impact_zh:
        score.impact_zh = score.summary_zh
    return score


def stored_score_fields(score: Score, digest: str, version: str) -> dict:
    """Columns written for one scored row. Never includes credentials."""
    ensure_display_fields(score)
    return {
        "direction": int(1 if score.direction > 0.15 else (-1 if score.direction < -0.15 else 0)),
        "direction_score": score.direction,
        "impact": int(round(score.urgency * 3)),
        "eth_relevance": score.eth_relevance,
        "urgency": score.urgency,
        "magnitude": score.magnitude,
        "event_type": score.event_type,
        "confidence": score.confidence,
        "scored_by": score.scorer,
        "score_version": version,
        "content_hash": digest,
        "summary_zh": score.summary_zh,
        "sentiment_label": score.label or sentiment_label(score.direction),
        "reason_zh": score.reason_zh,
        "importance": int(score.importance or 1),
        "impact_zh": score.impact_zh,
        "score_json": json.dumps(asdict(score), ensure_ascii=False),
        "status": "scored",
    }


def rule_summary_zh(title: str, event_type: str, direction: float, urgency: float,
                   relevance: float) -> str:
    """Fallback summary written from the headline only; never invents facts."""
    label = sentiment_label(direction)
    clean = re.sub(r"\s+", " ", (title or "").strip()).rstrip("。.!！?？")
    if len(clean) > 90:
        clean = clean[:87] + "…"
    what = clean or "标题信息不足"
    why = EVENT_WHY.get(event_type, EVENT_WHY["general"])
    uncertainty = "细节和影响幅度仍不确定。" if relevance < 0.6 else "具体影响仍取决于市场反应。"
    return f"{what}。这条{EVENT_CN.get(event_type, EVENT_CN['general'])}被判断为{label}，{why}；{uncertainty}"


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
    direction = _clip(direction, -1, 1)
    if market_channel(title, body):
        relevance = max(relevance, 0.55)
    return ensure_display_fields(Score(
        eth_relevance=round(relevance, 3), direction=round(direction, 3),
        urgency=round(urgency, 3), magnitude=round(magnitude, 3),
        event_type=event, confidence=0.35, scorer="rule_fallback",
        summary_zh=rule_summary_zh(title, event, direction, urgency, relevance),
    ))


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
                summary_zh=str(row.get("summary_zh", "") or "").strip(),
                label=str(row.get("label", "") or "").strip()[:8],
                reason_zh=str(row.get("reason_zh", "") or "").strip()[:400],
                importance=int(row.get("importance") or 0),
                impact_zh=str(row.get("impact_zh", "") or "").strip()[:400],
            )
        except (TypeError, ValueError):
            continue
    return out


def score_batch_deepseek(items: list[dict], key: str, *, proxy: str = PROXY,
                         retries: int = 3, timeout: float = 45.0) -> dict[int, Score] | None:
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
            with httpx.Client(timeout=timeout, proxy=proxy) as client:
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
                if not item.summary_zh:
                    item.summary_zh = rule_summary_zh(
                        str(row.get("title") or ""), item.event_type, item.direction,
                        item.urgency, item.eth_relevance,
                    )
                fields = stored_score_fields(
                    item,
                    content_hash(str(row.get("title") or ""), str(row.get("body") or "")),
                    f"{SCORER_VERSION}:{PROMPT_VERSION}",
                )
                fields["available_at"] = _visible_at(row)
                fields["timing_quality"] = _timing_quality(row)
                db.update_news_score(int(row["id"]), fields)
        return counts
    finally:
        db.close()


def score_version() -> str:
    return f"{SCORER_VERSION}:{PROMPT_VERSION}"


def _row_value(row, name: str, default):
    try:
        value = row[name]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def _score_from_row(row) -> Score:
    raw = {}
    try:
        raw = json.loads(row["score_json"] or "{}")
    except (TypeError, ValueError, KeyError):
        raw = {}
    direction = row["direction_score"] if row["direction_score"] is not None else raw.get("direction", 0.0)
    return Score(
        eth_relevance=float(row["eth_relevance"] if row["eth_relevance"] is not None
                            else raw.get("eth_relevance", 0.0)),
        direction=float(direction or 0.0),
        urgency=float(row["urgency"] if row["urgency"] is not None else raw.get("urgency", 0.0)),
        magnitude=float(row["magnitude"] if row["magnitude"] is not None else raw.get("magnitude", 0.0)),
        event_type=str(row["event_type"] or raw.get("event_type", "general")),
        confidence=float(row["confidence"] if row["confidence"] is not None else raw.get("confidence", 0.35)),
        scorer=str(row["scored_by"] or raw.get("scorer", "rule_fallback")),
        summary_zh=str(row["summary_zh"] or raw.get("summary_zh", "") or ""),
        label=str(_row_value(row, "sentiment_label", "") or raw.get("label", "") or ""),
        reason_zh=str(_row_value(row, "reason_zh", "") or raw.get("reason_zh", "") or ""),
        importance=int(_row_value(row, "importance", 0) or raw.get("importance") or 0),
        impact_zh=str(_row_value(row, "impact_zh", "") or raw.get("impact_zh", "") or ""),
    )


def score_items(db_path: Path, items: list[dict], *, use_deepseek: bool = True,
                proxy: str = PROXY, timeout: float = 15.0,
                batch_size: int = 8) -> dict:
    """Score freshly fetched rows before a click-time overlay.

    ``items`` are dicts with ``id``, ``title``, ``body``, ``url`` and optional
    ``content_hash``.  Results are keyed by content hash.  Scores are reused when
    the same content already has the current scorer/prompt version, so repeated
    clicks do not call DeepSeek again.  A missing key or a failed call degrades to
    the deterministic rule scorer; neither can block the forecast.
    """
    version = score_version()
    result: dict[str, Score] = {}
    counts = {"items": len(items), "cached": 0, "deepseek": 0, "rule": 0, "failed": 0,
              "deepseek_configured": secrets.deepseek_configured()}
    if not items:
        return {"scores": result, "counts": counts, "version": version}
    key = secrets.deepseek_api_key() if use_deepseek else None
    db = Database(Path(db_path))
    try:
        pending: list[tuple[dict, str]] = []
        seen: set[str] = set()
        for item in items:
            title = str(item.get("title") or "")
            body = str(item.get("body") or "")
            ch = str(item.get("content_hash") or content_hash(title, body))
            if ch in seen:
                continue
            seen.add(ch)
            cached = db.news_by_content_version(ch, version)
            if cached is not None:
                score = _score_from_row(cached)
                if not score.summary_zh:
                    score.summary_zh = rule_summary_zh(
                        title, score.event_type, score.direction, score.urgency, score.eth_relevance
                    )
                result[ch] = score
                counts["cached"] += 1
                continue
            pending.append((item, ch))

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            deepseek_rows: dict[int, Score] | None = None
            if key:
                payload = [{"id": index, "title": str(item.get("title") or "")[:400],
                            "body": str(item.get("body") or "")[:600]}
                           for index, (item, _ch) in enumerate(batch)]
                deepseek_rows = score_batch_deepseek(
                    payload, key, proxy=proxy, retries=1, timeout=timeout
                )
            for index, (item, ch) in enumerate(batch):
                title = str(item.get("title") or "")
                body = str(item.get("body") or "")
                score = deepseek_rows.get(index) if deepseek_rows else None
                if score is None:
                    score = rule_score(title, body)
                    if key:
                        counts["failed"] += 1
                    else:
                        counts["rule"] += 1
                else:
                    counts["deepseek"] += 1
                if not score.summary_zh:
                    score.summary_zh = rule_summary_zh(
                        title, score.event_type, score.direction, score.urgency, score.eth_relevance
                    )
                result[ch] = score
                news_id = item.get("id")
                if news_id is not None:
                    db.update_news_score(int(news_id), stored_score_fields(score, ch, version))
        return {"scores": result, "counts": counts, "version": version}
    finally:
        db.close()


def list_news(db_path: Path, *, limit: int = 50, since_ms: int | None = None,
              sources: tuple[str, ...] | None = None) -> list[dict]:
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
        summary = "COALESCE(summary_zh, '')" if "summary_zh" in cols else "''"
        label = "COALESCE(sentiment_label, '')" if "sentiment_label" in cols else "''"
        content = "COALESCE(content_hash, '')" if "content_hash" in cols else "''"
        reason = "COALESCE(reason_zh, '')" if "reason_zh" in cols else "''"
        importance = "COALESCE(importance, 0)" if "importance" in cols else "0"
        impact = "COALESCE(impact_zh, '')" if "impact_zh" in cols else "''"
        relevance = "COALESCE(eth_relevance, 0.0)"
        where = "1=1"
        args: list = []
        if since_ms is not None:
            where += f" AND {avail} >= ?"
            args.append(int(since_ms))
        if sources:
            marks = ",".join("?" for _ in sources)
            where += f" AND source IN ({marks})"
            args.extend(sources)
        sql = (f"SELECT id, {avail} AS available_at, ts_published, ts_ingested, source, title, url, "
               f"{relevance}, {direction}, {urgency}, {magnitude}, {event}, {timing}, "
               f"COALESCE(scored_by,''), COALESCE(score_version,''), {summary}, {label}, {content}, "
               f"{reason}, {importance}, {impact} "
               f"FROM news WHERE {where} ORDER BY available_at DESC, id DESC LIMIT ?")
        rows = conn.execute(sql, (*args, int(limit))).fetchall()
    finally:
        conn.close()
    keys = ["id", "available_at", "ts_published", "ts_ingested", "source", "title", "url",
            "relevance", "direction", "urgency", "magnitude", "event_type",
            "timing_quality", "scored_by", "score_version", "summary_zh",
            "sentiment_label", "content_hash", "reason_zh", "importance", "impact_zh"]
    return [dict(zip(keys, row, strict=True)) for row in rows]


def pending_llm_items(db_path: Path, sources: tuple[str, ...], since_ms: int, *,
                      limit: int = 10) -> list[dict]:
    """Panel-source rows still on the rule fallback, oldest gap first within the window.

    GDELT and already-scored model rows are not returned. The caller rate-limits.
    """
    path = Path(db_path)
    if not path.exists() or not sources:
        return []
    marks = ",".join("?" for _ in sources)
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            f"""SELECT id, title, body, url, source, content_hash, ts_published, available_at
                FROM news
                WHERE source IN ({marks})
                  AND COALESCE(ts_published, available_at, ts_ingested) >= ?
                  AND (scored_by IS NULL OR scored_by IN ('', 'pending', 'rule', 'rule_fallback'))
                ORDER BY COALESCE(ts_published, available_at, ts_ingested) DESC
                LIMIT ?""",
            (*sources, int(since_ms), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    keys = ["id", "title", "body", "url", "source", "content_hash", "ts_published", "available_at"]
    return [dict(zip(keys, row, strict=True)) for row in rows]
