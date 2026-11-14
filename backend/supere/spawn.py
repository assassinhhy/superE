"""Start the ingestor or API without inheriting the caller's pipes.

PowerShell waits until every process that inherited its stdout pipe has
exited. ``Start-Process -RedirectStandard*`` still sets ``bInheritHandles``,
so the service keeps that pipe open and ``cmd /c 启动superE.bat`` never
returns. ``close_fds=True`` on Windows passes only stdin/stdout/stderr
(``PROC_THREAD_ATTRIBUTE_HANDLE_LIST``), and ``DETACHED_PROCESS`` drops the
console.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Win32 CreateProcess flags. DETACHED_PROCESS must not be combined with
# CREATE_NEW_CONSOLE; CREATE_NO_WINDOW is ignored when DETACHED_PROCESS is set.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def spawn_detached(
    argv: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(stdout_path, "ab", buffering=0)
    err = open(stderr_path, "ab", buffering=0)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        kwargs: dict = {
            "args": argv,
            "cwd": str(cwd),
            "stdin": subprocess.DEVNULL,
            "stdout": out,
            "stderr": err,
            "close_fds": True,
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(**kwargs)
    finally:
        out.close()
        err.close()
    return int(proc.pid)


def spawn_service(role: str, *, root: Path, python: str | None = None) -> int:
    if role not in {"ingestor", "api", "board", "news"}:
        raise ValueError(f"未知进程 {role}")
    root = Path(root)
    logs = root / "data" / "logs"
    executable = python or sys.executable
    return spawn_detached(
        [executable, "-m", "supere", role],
        cwd=root,
        stdout_path=logs / f"{role}.stdout.log",
        stderr_path=logs / f"{role}.stderr.log",
    )
