"""Anthropic Messages-compatible scoring for the news panel.

The API key is read only through :mod:`supere.secrets`. It is never written to
logs, the database, or API payloads. One request is in flight at a time.
Failed items fall back to the rule scorer with ``scored_by='rule'`` so the
next refresh can try again. Successful scores are cached by content hash.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import httpx

from supere import secrets
from supere.news import summary as news_summary
from supere.news.scoring import (
    Score,
    content_hash,
    ensure_display_fields,
    rule_score,
    stored_score_fields,
)
from supere.storage.db import Database

log = logging.getLogger("supere.news.llm")

VERSION = "llm-v1:p1"
BATCH_SIZE = 10
MAX_ATTEMPTS = 3  # first try plus two retries
BASE_DELAY_S = 0.8
_LLM_LOCK = threading.Lock()
_REDACTORS: dict[str, logging.Filter] = {}

SYSTEM_PROMPT = (
    "你是以太坊消息编辑。只根据给出的标题和 RSS 摘要写中文，"
    "不要编造数字、价格、百分比或摘要里没有的事实，也不要给交易建议。"
    "只返回 JSON，不要其它文字。"
    "对每条给出：id（原样返回）、summary_zh（1 到 2 句中文，说明发生了什么）、"
    "label（只能是利多、利空、中性之一，相对 ETH）、reason_zh（一句为什么）、"
    "importance（整数 1、2 或 3，3 最可能影响 ETH）、"
    "impact_zh（importance 大于等于 2 时用一句说明发生了什么、对 ETH 可能怎么影响；否则空字符串）、"
    "direction（-1 到 1，利空为负）、urgency（0 到 1）、relevance（0 到 1）。"
    "relevance 这样打：直接关于 ETH、以太坊、质押、二层或 Vitalik 用 0.7 以上；"
    "宏观（美联储、通胀、美元、利率）以及大盘加密（比特币价格、ETF 资金流、清算）用 0.5 到 0.8，这些要保留；"
    "和 ETH 价格几乎无关的（其它公链八卦、地方新闻、房价、绑架、税务局接受比特币这类）用 0.2 以下。"
)

DIGEST_SYSTEM = (
    "你是以太坊消息编辑。根据给出的统计和重要条目写两段中文，每段 1 到 2 句。"
    "不要编造统计里没有的数字、价格或百分比，不要给交易建议。"
    "只返回 JSON：{\"prose_1h\":\"...\",\"prose_24h\":\"...\"}。"
)

_LABELS = {
    "利多": "利多", "看多": "利多", "偏多": "利多", "bullish": "利多",
    "利空": "利空", "看空": "利空", "偏空": "利空", "bearish": "利空",
    "中性": "中性", "neutral": "中性",
}


class _KeyRedactor(logging.Filter):
    """Drop configured keys from log records. The key stays in memory only."""

    def __init__(self, secret: str):
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        secret = self._secret
        if not secret:
            return True
        if isinstance(record.msg, str) and secret in record.msg:
            record.msg = record.msg.replace(secret, "***")
        if record.args:
            rendered = None
            try:
                rendered = record.getMessage()
            except Exception:
                rendered = None
            if rendered and secret in rendered:
                record.msg = rendered.replace(secret, "***")
                record.args = ()
                record.message = record.msg
        return True


def install_key_redaction(secret: str | None) -> None:
    if not secret:
        return
    if secret in _REDACTORS:
        return
    redactor = _KeyRedactor(secret)
    _REDACTORS[secret] = redactor
    for name in ("supere.news", "supere.news.llm", "supere.news.refresh", "httpx", "httpcore"):
        logging.getLogger(name).addFilter(redactor)


def messages_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/v1/messages") or base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/v1/messages"


def _headers(key: str) -> dict[str, str]:
    return {
        "x-api-key": key,
        "authorization": f"Bearer {key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-opencode-session": "supere-news",
    }


def _message_text(payload: dict) -> str:
    blocks = payload.get("content")
    if isinstance(blocks, list):
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text") or ""))
        if parts:
            return "\n".join(parts)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return str(message.get("content") or "")
    return ""


def _extract_json(text: str) -> dict | list | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _score_from_obj(row: dict) -> Score | None:
    label = _LABELS.get(str(row.get("label") or "").strip().lower(), "")
    if not label:
        label = _LABELS.get(str(row.get("label") or "").strip(), "")
    summary = str(row.get("summary_zh") or "").strip()
    reason = str(row.get("reason_zh") or "").strip()
    if label not in ("利多", "利空", "中性") or not summary or not reason:
        return None
    try:
        importance = int(row.get("importance"))
    except (TypeError, ValueError):
        return None
    if importance not in (1, 2, 3):
        return None
    try:
        direction = float(row["direction"]) if row.get("direction") is not None else {
            "利多": 0.6, "利空": -0.6, "中性": 0.0,
        }[label]
        urgency = float(row["urgency"]) if row.get("urgency") is not None else {1: 0.3, 2: 0.6, 3: 0.9}[importance]
        relevance = float(row.get("relevance", row.get("eth_relevance", 0.6)))
    except (TypeError, ValueError):
        return None
    impact = str(row.get("impact_zh") or "").strip()
    if importance >= 2 and not impact:
        impact = summary if reason in summary else f"{summary.rstrip('。')}。{reason}"
    return ensure_display_fields(Score(
        eth_relevance=_clip(relevance, 0, 1),
        direction=_clip(direction, -1, 1),
        urgency=_clip(urgency, 0, 1),
        magnitude=_clip(urgency, 0, 1),
        event_type="general",
        confidence=0.6,
        scorer="llm",
        summary_zh=summary[:400],
        label=label,
        reason_zh=reason[:400],
        importance=importance,
        impact_zh=impact[:400],
    ))


def _results(payload: dict | list | None) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("results")
        if isinstance(rows, list):
            return rows
    return []


def _rule(title: str, body: str) -> Score:
    score = rule_score(title, body)
    score.scorer = "rule"
    return score


def _post_json(url: str, key: str, body: dict, *, proxy: str | None, timeout: float,
               sleeper, client_factory) -> dict | None:
    """POST with two retries on HTTP 429 and timeouts. Never logs the key."""
    install_key_redaction(key)
    delay = BASE_DELAY_S
    last_status: int | None = None

    def _open():
        if client_factory is not None:
            return client_factory()
        kwargs: dict = {"timeout": httpx.Timeout(timeout)}
        if proxy:
            kwargs["proxy"] = proxy
        return httpx.Client(**kwargs)

    for attempt in range(MAX_ATTEMPTS):
        try:
            with _open() as client:
                response = client.post(url, headers=_headers(key), json=body)
            last_status = int(response.status_code)
            if response.status_code == 400 and "thinking" in body:
                slim = {name: value for name, value in body.items() if name != "thinking"}
                with _open() as client:
                    response = client.post(url, headers=_headers(key), json=slim)
                last_status = int(response.status_code)
            if response.status_code == 200:
                try:
                    return response.json()
                except Exception:
                    log.warning("news llm response was not json status=%s", last_status)
                    return None
            if response.status_code == 429 or response.status_code == 408:
                if attempt < MAX_ATTEMPTS - 1:
                    sleeper(delay)
                    delay *= 2
                    continue
                log.warning("news llm gave up status=%s", response.status_code)
                return None
            log.warning("news llm request failed status=%s", response.status_code)
            return None
        except httpx.TimeoutException:
            if attempt < MAX_ATTEMPTS - 1:
                sleeper(delay)
                delay *= 2
                continue
            log.warning("news llm timed out")
            return None
        except Exception as exc:
            log.warning("news llm request failed error=%s", type(exc).__name__)
            return None
    log.warning("news llm gave up status=%s", last_status)
    return None


def _chat(system: str, user: str, *, model: str, key: str, base_url: str, proxy: str | None,
          timeout: float, sleeper, client_factory) -> dict | None:
    body = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    with _LLM_LOCK:
        payload = _post_json(
            messages_url(base_url), key, body,
            proxy=proxy, timeout=timeout, sleeper=sleeper, client_factory=client_factory,
        )
    if not isinstance(payload, dict):
        return None
    return payload


def _cached_score(row) -> Score | None:
    try:
        summary = str(row["summary_zh"] or "")
        label = str(row["sentiment_label"] or "")
    except (KeyError, IndexError, TypeError):
        return None
    if not summary or label not in ("利多", "利空", "中性"):
        return None
    try:
        direction = float(row["direction_score"] if row["direction_score"] is not None else 0.0)
        urgency = float(row["urgency"] or 0.0)
        relevance = float(row["eth_relevance"] or 0.0)
        importance = int(row["importance"] or 0)
    except (KeyError, TypeError, ValueError):
        direction, urgency, relevance, importance = 0.0, 0.0, 0.0, 0
    reason = ""
    impact = ""
    try:
        reason = str(row["reason_zh"] or "")
        impact = str(row["impact_zh"] or "")
    except (KeyError, IndexError):
        reason, impact = "", ""
    return ensure_display_fields(Score(
        eth_relevance=relevance,
        direction=direction,
        urgency=urgency,
        magnitude=urgency,
        event_type=str(row["event_type"] or "general") if "event_type" in row.keys() else "general",
        confidence=float(row["confidence"] or 0.6) if "confidence" in row.keys() else 0.6,
        scorer=str(row["scored_by"] or "llm"),
        summary_zh=summary,
        label=label,
        reason_zh=reason,
        importance=importance,
        impact_zh=impact,
    ))


def _apply(db: Database | None, item: dict, score: Score, digest: str) -> None:
    item["content_hash"] = digest
    item["relevance"] = score.eth_relevance
    item["direction"] = score.direction
    item["urgency"] = score.urgency
    item["magnitude"] = score.magnitude
    item["event_type"] = score.event_type
    item["confidence"] = score.confidence
    item["scored_by"] = score.scorer
    item["score_version"] = VERSION
    item["summary_zh"] = score.summary_zh
    item["sentiment_label"] = score.label
    item["reason_zh"] = score.reason_zh
    item["importance"] = int(score.importance or 1)
    item["impact_zh"] = score.impact_zh
    news_id = item.get("id")
    if db is not None and news_id is not None:
        db.update_news_score(int(news_id), stored_score_fields(score, digest, VERSION))


def score_panel_items(db_path: Path, items: list[dict], *, proxy: str | None = None,
                      timeout: float = 40.0, sleeper=None, client_factory=None,
                      batch_size: int = BATCH_SIZE) -> dict:
    """Score items once per content hash. Returns the same shape as ``score_items``."""
    import time

    sleeper = sleeper or time.sleep
    version = VERSION
    result: dict[str, Score] = {}
    counts = {"items": len(items), "cached": 0, "llm": 0, "rule": 0, "failed": 0,
              "deepseek": 0, "deepseek_configured": False, "llm_configured": secrets.llm_configured()}
    if not items:
        return {"scores": result, "counts": counts, "version": version}
    key = secrets.llm_api_key()
    base = secrets.llm_base_url()
    model = secrets.llm_model()
    install_key_redaction(key)
    db = Database(Path(db_path))
    try:
        pending: list[tuple[dict, str]] = []
        seen: set[str] = set()
        for item in items:
            title = str(item.get("title") or "")
            body = str(item.get("body") or "")
            digest = str(item.get("content_hash") or content_hash(title, body))
            if digest in seen:
                continue
            seen.add(digest)
            cached = db.news_llm_hit(digest, version)
            if cached is not None:
                score = _cached_score(cached)
                if score is not None:
                    result[digest] = score
                    _apply(None, item, score, digest)
                    counts["cached"] += 1
                    continue
            pending.append((item, digest))

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            parsed: dict[int, Score] = {}
            if key and base:
                user = json.dumps([
                    {
                        "id": index,
                        "title": str(item.get("title") or "")[:400],
                        "rss_summary": str(item.get("body") or "")[:600],
                        "source": str(item.get("source") or ""),
                    }
                    for index, (item, _digest) in enumerate(batch)
                ], ensure_ascii=False)
                payload = _chat(
                    SYSTEM_PROMPT,
                    "条目：\n" + user + "\n返回 {\"results\":[...]}。",
                    model=model, key=key, base_url=base, proxy=proxy, timeout=timeout,
                    sleeper=sleeper, client_factory=client_factory,
                )
                text = _message_text(payload or {})
                for row in _results(_extract_json(text)):
                    if not isinstance(row, dict):
                        continue
                    try:
                        index = int(row.get("id"))
                    except (TypeError, ValueError):
                        continue
                    score = _score_from_obj(row)
                    if score is not None and 0 <= index < len(batch):
                        parsed[index] = score
            for index, (item, digest) in enumerate(batch):
                score = parsed.get(index)
                if score is None:
                    score = _rule(str(item.get("title") or ""), str(item.get("body") or ""))
                    counts["rule"] += 1
                    if key:
                        counts["failed"] += 1
                else:
                    counts["llm"] += 1
                result[digest] = score
                _apply(db, item, score, digest)
        log.info("news llm batch cached=%s llm=%s rule=%s", counts["cached"], counts["llm"], counts["rule"])
        return {"scores": result, "counts": counts, "version": version}
    finally:
        db.close()


def digest_prose(items: list[dict], now_ms: int, *, proxy: str | None = None, timeout: float = 40.0,
                 sleeper=None, client_factory=None) -> dict | None:
    """Optional 1h/24h prose. None means the caller should use the rule summary."""
    import time

    if not secrets.llm_configured() or not items:
        return None
    key = secrets.llm_api_key()
    base = secrets.llm_base_url()
    if not key or not base:
        return None
    install_key_redaction(key)
    sleeper = sleeper or time.sleep
    facts = news_summary.digest_payload(items, now_ms)
    payload = _chat(
        DIGEST_SYSTEM,
        "统计：\n" + json.dumps(facts, ensure_ascii=False),
        model=secrets.llm_model(), key=key, base_url=base, proxy=proxy, timeout=timeout,
        sleeper=sleeper, client_factory=client_factory,
    )
    parsed = _extract_json(_message_text(payload or {}))
    if not isinstance(parsed, dict):
        return None
    prose_1h = str(parsed.get("prose_1h") or "").strip()
    prose_24h = str(parsed.get("prose_24h") or "").strip()
    if not prose_1h and not prose_24h:
        return None
    if key in prose_1h or key in prose_24h:
        return None
    return {"prose_1h": prose_1h, "prose_24h": prose_24h}
