"""Command line. Run from the repository root so ``config.toml`` is found."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from supere import __version__
from supere.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="supere", description="superE 数据层。只读公开行情，不下单。")
    parser.add_argument("--version", action="version", version=f"superE {__version__}")
    parser.add_argument("--config", help="config.toml 路径")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingestor", help="采集器：唯一的 SQLite 写入进程")
    ing.add_argument("--with-backfill", action="store_true", help="先做存档回填，再进入实时采集")

    sub.add_parser("backfill", help="从 data.binance.vision 导入历史，不启动实时连接")
    sub.add_parser("api", help="只读 API，默认 127.0.0.1:8766")
    desk = sub.add_parser("desktop", help="等本机 API 就绪后打开桌面窗口")
    desk.add_argument("--no-window", action="store_true", help="只检查 API，不创建窗口")
    prev = sub.add_parser("preview", help="用合成数据启动界面，不连接币安")
    prev.add_argument("--no-window", action="store_true", help="只启动 API，不创建窗口")
    prev.add_argument("--port", type=int, help="覆盖 API 端口，默认仍是 8766")
    prev.add_argument("--days", type=int, default=10, help="合成 K 线的天数，从 2026-10-26 起")
    detach = sub.add_parser("detach", help="脱离调用方的管道后启动 ingestor 或 api，然后立刻退出")
    detach.add_argument("role", choices=("ingestor", "api", "board", "news"))
    status_cmd = sub.add_parser("status", help="查看各服务的 PID 与端口监听状态")
    status_cmd.add_argument("--role", choices=("ingestor", "api", "board", "news"),
                            help="只看一个服务，默认全部")
    status_cmd.add_argument("--json", action="store_true", help="输出机器可读的 JSON")
    stop_cmd = sub.add_parser("stop", help="停止服务，默认全部")
    stop_cmd.add_argument("--role", choices=("ingestor", "api", "board", "news"),
                          help="只停一个服务，默认全部")
    news_cmd = sub.add_parser("news", help="消息面服务，127.0.0.1:8769，不加载预测模型")
    news_cmd.add_argument("--port", type=int, default=8769)
    news_cmd.add_argument("--data-dir", help="数据目录，默认用配置里的目录")
    board_cmd = sub.add_parser("board", help="盘面分析，127.0.0.1:8770，只读主库，不下单")
    board_cmd.add_argument("--port", type=int)
    board_cmd.add_argument("--data-dir", help="数据目录，默认用配置里的目录")
    board_cmd.add_argument("--analysis", help="analysis.toml 路径")
    board_cmd.add_argument("--fixture", help="回放 CSV 目录。和 --at 一起用时不访问外网")
    board_cmd.add_argument("--at", help="回放时刻，UTC+8，格式 YYYY-MM-DD HH:MM")
    sub.add_parser("backup", help="立刻做一次 SQLite 在线备份和 Parquet 增量复制")
    sub.add_parser("restore-drill", help="把最近一份备份复制出来做完整性检查，不替换正在用的库")

    args = parser.parse_args(argv)
    if args.cmd in {"detach", "status", "stop"}:
        # These three must work even when config.toml is broken: the stop
        # button is most useful exactly when the services misbehave.
        from supere.spawn import service_status, spawn_service, stop_services

        root = Path(args.config).resolve().parent if args.config else Path.cwd()
        data = _data_root(args.config, root)
        if args.cmd == "detach":
            print(spawn_service(args.role, root=root, data_root=data))
            return
        if args.cmd == "status":
            roles = [args.role] if args.role else None
            rows = service_status(data, roles=roles)
            _print_status(rows, as_json=args.json)
            return
        roles = [args.role] if args.role else None
        outcomes = stop_services(data, roles=roles)
        failed = _report_stop(outcomes)
        if failed:
            raise SystemExit(1)
        return
    try:
        cfg, path = load_config(args.config)
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc

    if args.cmd == "ingestor":
        from supere.ingest.live import Ingestor

        asyncio.run(Ingestor(cfg, path).run(backfill=args.with_backfill, live=True))
    elif args.cmd == "backfill":
        from supere.ingest.live import Ingestor

        asyncio.run(Ingestor(cfg, path).run(backfill=True, live=False))
    elif args.cmd == "api":
        from supere.api.app import serve

        asyncio.run(serve(cfg, path))
    elif args.cmd == "desktop":
        from supere.desktop import serve_desktop

        serve_desktop(cfg, path, open_window=not args.no_window)
    elif args.cmd == "preview":
        from supere.preview import serve_preview

        serve_preview(
            cfg,
            path,
            open_window=not args.no_window,
            port=args.port,
            days=args.days,
        )
    elif args.cmd == "news":
        from supere.config import data_dir
        from supere.news.serve import main as news_main

        root = Path(args.data_dir).resolve() if args.data_dir else data_dir(cfg, path)
        raise SystemExit(news_main(["--data-dir", str(root), "--port", str(args.port)]))
    elif args.cmd == "board":
        from supere.config import data_dir
        from supere.board.serve import main as board_main

        root = Path(args.data_dir).resolve() if args.data_dir else data_dir(cfg, path)
        argv = ["--data-dir", str(root), "--config", str(path)]
        if args.port:
            argv += ["--port", str(args.port)]
        if args.analysis:
            argv += ["--analysis", args.analysis]
        if args.fixture:
            argv += ["--fixture", args.fixture]
        if args.at:
            argv += ["--at", args.at]
        raise SystemExit(board_main(argv))
    elif args.cmd == "backup":
        _backup(cfg, path)
    elif args.cmd == "restore-drill":
        raise SystemExit(_restore_drill(cfg, path))
    else:
        raise SystemExit(2)


def _data_root(config: str | None, root: Path) -> Path:
    """Resolve ``data.dir`` from config; fall back to ``<root>/data``.

    ``status``/``stop`` must not fail just because ``config.toml`` is missing
    or invalid, so every config error degrades to the default location.
    """
    from supere.config import data_dir

    try:
        cfg, path = load_config(config)
    except Exception:
        return Path(root) / "data"
    return data_dir(cfg, path)


def _print_status(rows, *, as_json: bool = False) -> None:
    if as_json:
        payload = [
            {
                "role": row.role,
                "port": row.port,
                "pid": row.pid,
                "alive": row.alive,
                "port_open": row.port_open,
                "running": row.running,
                "stale_pid": row.stale_pid,
            }
            for row in rows
        ]
        print(json.dumps(payload, ensure_ascii=False))
        return
    for row in rows:
        pid = str(row.pid) if row.pid is not None else "-"
        alive = "是" if row.alive else "否"
        listening = "是" if row.port_open else "否"
        print(f"{row.role:<9} pid={pid:<8} 存活={alive:<2} 端口={row.port} 监听={listening}")


def _report_stop(outcomes) -> bool:
    """Print one line per service; return True when any stop failed."""
    failed = False
    for outcome in outcomes:
        if outcome.status == "stopped":
            print(f"已停止 {outcome.role} (pid {outcome.pid})")
        elif outcome.status == "not-running":
            print(f"未运行 {outcome.role}")
        else:
            failed = True
            detail = f": {outcome.detail}" if outcome.detail else ""
            print(f"停止失败 {outcome.role}{detail}", file=sys.stderr)
    return failed


def _backup(cfg, path) -> None:
    from supere.config import data_dir
    from supere.storage.backup import backup_sqlite, copy_new_parquet
    from supere.timeutil import UTC8
    from datetime import datetime

    root = data_dir(cfg, path)
    src = root / "superE.sqlite"
    if not src.exists():
        raise SystemExit("还没有数据库，先启动采集器或运行 backfill")
    now = datetime.now(UTC8)
    dest = root / "backups" / f"superE-{now:%Y%m%d}.sqlite"
    backup_sqlite(src, dest)
    copied = copy_new_parquet(root / "raw", root / "backups" / "parquet", root / "backups" / "parquet-manifest.json")
    print(f"备份完成: {dest} ，Parquet 新文件 {copied}")


def _restore_drill(cfg, path) -> int:
    from supere.config import data_dir
    from supere.storage.backup import latest_sqlite_backup, restore_drill

    root = data_dir(cfg, path)
    backup = latest_sqlite_backup(root / "backups")
    if backup is None:
        print("没有找到备份", file=sys.stderr)
        return 1
    result = restore_drill(backup, root / "backups" / "drill")
    print(result)
    return 0 if result["integrity"] == "ok" else 1


if __name__ == "__main__":
    main()
