"""Start, inspect and stop the ingestor or API without inheriting pipes.

PowerShell waits until every process that inherited its stdout pipe has
exited. ``Start-Process -RedirectStandard*`` still sets ``bInheritHandles``,
so the service keeps that pipe open and ``cmd /c 启动superE.bat`` never
returns. ``close_fds=True`` on Windows passes only stdin/stdout/stderr
(``PROC_THREAD_ATTRIBUTE_HANDLE_LIST``), and ``DETACHED_PROCESS`` drops the
console.

Every detached service writes ``data/run/<role>.pid``.  ``status`` combines
that file with an actual process check and a loopback port probe, so a stale
pid file cannot make a dead service look alive.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Win32 CreateProcess flags. DETACHED_PROCESS must not be combined with
# CREATE_NEW_CONSOLE; CREATE_NO_WINDOW is ignored when DETACHED_PROCESS is set.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200

# Roles in start-up order, plus the loopback port each one listens on.
ROLE_ORDER = ("ingestor", "api", "board", "news")
ROLE_PORTS = {
    "ingestor": 8767,  # IPC / collector control port
    "api": 8766,  # read-only chart API
    "board": 8770,
    "news": 8769,
}


def _data_root(root: Path, data_root: Path | None) -> Path:
    return Path(data_root) if data_root is not None else Path(root) / "data"


def run_dir(data_root: Path) -> Path:
    return Path(data_root) / "run"


def pid_path(data_root: Path, role: str) -> Path:
    _check_role(role)
    return run_dir(data_root) / f"{role}.pid"


def _check_role(role: str) -> None:
    if role not in ROLE_PORTS:
        raise ValueError(f"未知进程 {role}")


def write_pid(data_root: Path, role: str, pid: int) -> Path:
    """Write ``<role>.pid`` atomically and return its path."""
    path = pid_path(data_root, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(f"{int(pid)}\n", encoding="ascii")
    os.replace(temp, path)
    return path


def read_pid(data_root: Path, role: str) -> int | None:
    """Return the recorded pid, or ``None`` when missing or malformed."""
    try:
        text = pid_path(data_root, role).read_text(encoding="ascii").strip()
    except OSError:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid > 0 else None


def remove_pid(data_root: Path, role: str) -> None:
    pid_path(data_root, role).unlink(missing_ok=True)


# -- process / port probes -------------------------------------------------


def process_alive(pid: int) -> bool:
    """True when ``pid`` exists.  Never signals the process on Windows."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_alive(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """True when something accepts a TCP connection on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pid_on_port(port: int, host: str = "127.0.0.1") -> int | None:
    """Best-effort lookup of the listener pid.  ``None`` when unknown."""
    if sys.platform == "win32":
        return _pid_on_port_windows(port)
    return _pid_on_port_posix(port)


def _pid_on_port_windows(port: int) -> int | None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    suffix = f":{port}"
    for line in (completed.stdout or "").splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "TCP":
            continue
        if fields[3].upper() != "LISTENING":
            continue
        if not fields[1].endswith(suffix):
            continue
        try:
            return int(fields[4])
        except ValueError:
            continue
    return None


def _pid_on_port_posix(port: int) -> int | None:
    try:
        completed = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def terminate_pid(pid: int, *, force: bool = False) -> bool:
    """Ask ``pid`` to exit; return whether the command was accepted."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        args = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            args.append("/F")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# -- status ----------------------------------------------------------------


@dataclass(frozen=True)
class RoleStatus:
    role: str
    port: int
    pid: int | None
    alive: bool
    port_open: bool
    stale_pid: bool = False

    @property
    def running(self) -> bool:
        return self.alive or self.port_open


def service_status(
    data_root: Path,
    *,
    roles: tuple[str, ...] | list[str] | None = None,
    alive=process_alive,
    probe=port_open,
    pid_for_port=pid_on_port,
) -> list[RoleStatus]:
    """One status row per role.  All probes are injectable for tests."""
    selected = tuple(roles) if roles else ROLE_ORDER
    rows: list[RoleStatus] = []
    for role in selected:
        _check_role(role)
        port = ROLE_PORTS[role]
        pid = read_pid(data_root, role)
        is_alive = pid is not None and alive(pid)
        opened = probe(port)
        if not is_alive and opened:
            discovered = pid_for_port(port)
            if discovered:
                pid = discovered
                is_alive = alive(discovered)
        rows.append(
            RoleStatus(
                role=role,
                port=port,
                pid=pid,
                alive=is_alive,
                port_open=opened,
                stale_pid=pid is not None and not is_alive,
            )
        )
    return rows


# -- stop ------------------------------------------------------------------


@dataclass(frozen=True)
class StopOutcome:
    role: str
    pid: int | None
    status: str  # "stopped" | "not-running" | "failed"
    detail: str = ""


def _wait_gone(pid: int, alive, sleep, timeout: float, interval: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(interval)


def stop_role(
    data_root: Path,
    role: str,
    *,
    alive=process_alive,
    probe=port_open,
    pid_for_port=pid_on_port,
    terminate=terminate_pid,
    sleep=time.sleep,
    timeout: float = 2.0,
    interval: float = 0.2,
) -> StopOutcome:
    """Stop one service.  A stale pid file is removed without killing anything."""
    _check_role(role)
    port = ROLE_PORTS[role]
    pid = read_pid(data_root, role)

    if pid is not None and not alive(pid):
        # The file outlived its process: drop it, then check for an orphan
        # that still owns the port.
        remove_pid(data_root, role)
        pid = None

    if pid is None and probe(port):
        pid = pid_for_port(port)

    if pid is None:
        return StopOutcome(role, None, "not-running")
    if not alive(pid):
        return StopOutcome(role, pid, "not-running")

    # Try a graceful request first, then force.  The command's exit code is
    # not authoritative: the process may have died between the alive check
    # and the call, so the wait decides.
    terminate(pid, force=False)
    if _wait_gone(pid, alive, sleep, timeout, interval):
        remove_pid(data_root, role)
        return StopOutcome(role, pid, "stopped")

    terminate(pid, force=True)
    if _wait_gone(pid, alive, sleep, timeout, interval):
        remove_pid(data_root, role)
        return StopOutcome(role, pid, "stopped")

    detail = "进程仍在运行" if alive(pid) else "终止命令失败"
    return StopOutcome(role, pid, "failed", detail)


def stop_services(
    data_root: Path,
    *,
    roles: tuple[str, ...] | list[str] | None = None,
    **kwargs,
) -> list[StopOutcome]:
    """Stop the selected roles (default: all) and return one outcome each."""
    selected = tuple(roles) if roles else ROLE_ORDER
    return [stop_role(data_root, role, **kwargs) for role in selected]


# -- spawn -----------------------------------------------------------------


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


def spawn_service(
    role: str,
    *,
    root: Path,
    data_root: Path | None = None,
    python: str | None = None,
) -> int:
    if role not in ROLE_PORTS:
        raise ValueError(f"未知进程 {role}")
    root = Path(root)
    data = _data_root(root, data_root)
    logs = data / "logs"
    executable = python or sys.executable
    pid = spawn_detached(
        [executable, "-m", "supere", role],
        cwd=root,
        stdout_path=logs / f"{role}.stdout.log",
        stderr_path=logs / f"{role}.stderr.log",
    )
    write_pid(data, role, pid)
    return pid
