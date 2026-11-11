"""1a acceptance entry point.

From the repository root:

    .venv\\Scripts\\python.exe scripts\\accept_1a.py
    .venv\\Scripts\\python.exe scripts\\accept_1a.py --data-dir data
    .venv\\Scripts\\python.exe scripts\\accept_1a.py --live-outage --pause-seconds 300

``--live-outage`` pauses the local forwarder on 127.0.0.1:7891. Point
``proxy.url`` at that forwarder before starting the collector. The forwarder
itself sends bytes to 127.0.0.1:7890. This does not disable the system proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from supere.acceptance import evaluate_data_dir, offline_checks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="superE 1a 验收")
    parser.add_argument("--data-dir", type=Path, help="采集器写出的 data 目录")
    parser.add_argument("--live-outage", action="store_true", help="暂停本地转发代理，模拟断网")
    parser.add_argument("--pause-seconds", type=float, default=300.0)
    parser.add_argument("--listen-port", type=int, default=7891)
    parser.add_argument("--upstream", default="127.0.0.1:7890")
    args = parser.parse_args()

    if args.live_outage:
        return asyncio.run(_outage(args))

    work = Path.cwd() / ".accept-tmp"
    work.mkdir(exist_ok=True)
    checks = offline_checks(work)
    if args.data_dir:
        checks.extend(evaluate_data_dir(args.data_dir))
    failed = 0
    for item in checks:
        print(item.line())
        if item.status == "FAIL":
            failed += 1
    return 1 if failed else 0


async def _outage(args) -> int:
    from supere.forwarder import PauseProxy

    host, port_text = args.upstream.rsplit(":", 1)
    proxy = PauseProxy(host, int(port_text), listen_port=args.listen_port)
    await proxy.start()
    print(f"转发代理已监听 127.0.0.1:{proxy.listen_port} -> {args.upstream}")
    print("请确认采集器的 proxy.url 指向这个端口，然后本脚本会暂停转发。")
    print(f"暂停 {args.pause_seconds:.0f} 秒……")
    await proxy.pause()
    try:
        await asyncio.sleep(args.pause_seconds)
    finally:
        await proxy.resume()
    print("已恢复转发。60 秒内采集器应重连并写 GAP_FILL。随后用 --data-dir 看 A4。")
    await proxy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
