"""URL hash and 64-bit SimHash. Hamming distance <= 3 means the same item."""

from __future__ import annotations

import hashlib
import re


def url_hash(url: str) -> str:
    normalized = url.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def simhash64(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())
    if not tokens:
        return 0
    vector = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        hashed = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (hashed >> bit) & 1 else -1
    value = 0
    for bit, weight in enumerate(vector):
        if weight > 0:
            value |= 1 << bit
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def same_text(left: int, right: int, max_distance: int = 3) -> bool:
    return hamming(left, right) <= max_distance
