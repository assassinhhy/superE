"""Score pending news rows (DeepSeek when configured, rule fallback otherwise).

The key is read inside :mod:`supere.news.scoring` from ``supere.secrets``.  The
script prints only counts; it never echoes the key or a scored payload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from supere.news.scoring import PROXY, score_pending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_news")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--proxy", default=PROXY)
    parser.add_argument("--no-deepseek", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args(argv)
    counts = score_pending(Path(args.data_dir) / "superE.sqlite", limit=args.limit,
                           batch_size=args.batch_size, use_deepseek=not args.no_deepseek,
                           proxy=args.proxy, rescore=args.rescore)
    print(f"news scoring: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
