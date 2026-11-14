"""pywebview shell. The page is the local API, never a remote site."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from supere.config import AppConfig, data_dir


def origin_of(cfg: AppConfig) -> str:
    return f"http://{cfg.server.api_host}:{cfg.server.api_port}"


def wait_for_api(origin: str, timeout: float = 45) -> bool:
    deadline = time.monotonic() + timeout
    url = origin.rstrip("/") + "/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def _plain(text: str, limit: int = 140) -> str:
    cleaned = "".join(ch for ch in str(text) if ch.isprintable() and ch not in "\r\n\t")
    return cleaned[:limit]


class WindowApi:
    """JS bridge. ``window.pywebview.api.notify(title, body)`` shows a Windows balloon.

    WebView2 does not reliably deliver the browser Notification API, so the
    desktop process asks the system shell instead. The call is a no-op off Windows.
    Titles and bodies are clipped and never written to the log.
    """

    def notify(self, title: str, body: str) -> dict:
        if sys.platform != "win32":
            return {"ok": False, "reason": "not_windows"}
        script = (
            "Add-Type -AssemblyName System.Windows.Forms\n"
            "Add-Type -AssemblyName System.Drawing\n"
            "$n = New-Object System.Windows.Forms.NotifyIcon\n"
            "$n.Icon = [System.Drawing.SystemIcons]::Information\n"
            "$n.Visible = $true\n"
            "$n.BalloonTipTitle = $env:SUPERE_NOTIFY_TITLE\n"
            "$n.BalloonTipText = $env:SUPERE_NOTIFY_BODY\n"
            "$n.ShowBalloonTip(6000)\n"
            "Start-Sleep -Seconds 6\n"
            "$n.Dispose()\n"
        )
        env = os.environ.copy()
        env["SUPERE_NOTIFY_TITLE"] = _plain(title)
        env["SUPERE_NOTIFY_BODY"] = _plain(body, 180)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            env=env,
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"ok": True}


def run_window(url: str) -> None:
    import webview

    webview.create_window(
        "superE",
        url,
        width=1440,
        height=900,
        min_size=(1100, 720),
        text_select=True,
        js_api=WindowApi(),
    )
    webview.start()


def serve_desktop(cfg: AppConfig, config_path: Path, *, open_window: bool = True) -> None:
    root = data_dir(cfg, config_path)
    pid = root / "run" / "desktop.pid"
    pid.parent.mkdir(parents=True, exist_ok=True)
    pid.write_text(str(os.getpid()), encoding="utf-8")
    try:
        origin = origin_of(cfg)
        if not wait_for_api(origin):
            raise SystemExit(
                f"API 没有在 {origin} 就绪。先启动采集器和 API，或运行 python -m supere preview。"
            )
        if not open_window:
            print(origin + "/")
            return
        try:
            run_window(origin + "/")
        except Exception as exc:
            print(f"桌面窗口没有打开: {exc}", file=sys.stderr)
            print(
                "Windows 需要已安装的 WebView2 Runtime。浏览器打开 "
                f"{origin}/ 也能看同一份界面。",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
    finally:
        pid.unlink(missing_ok=True)
