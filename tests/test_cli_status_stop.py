"""Unit tests for ``supere status`` / ``supere stop``.

Every process and port probe is injected, so this file never enumerates,
signals or starts a real process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from supere import cli, spawn
from supere.spawn import RoleStatus, StopOutcome


def test_role_ports_match_the_documented_layout():
    assert spawn.ROLE_PORTS == {
        "ingestor": 8767,
        "api": 8766,
        "board": 8770,
        "news": 8769,
    }
    assert spawn.ROLE_ORDER == ("ingestor", "api", "board", "news")


def test_pid_file_roundtrip(tmp_path: Path):
    path = spawn.write_pid(tmp_path, "api", 1234)
    assert path == tmp_path / "run" / "api.pid"
    assert path.read_text(encoding="ascii").strip() == "1234"
    assert not path.with_name("api.pid.tmp").exists()
    assert spawn.read_pid(tmp_path, "api") == 1234

    spawn.remove_pid(tmp_path, "api")
    assert spawn.read_pid(tmp_path, "api") is None
    spawn.remove_pid(tmp_path, "api")  # missing file is fine


@pytest.mark.parametrize("text", ["", "not-a-pid", "0", "-5"])
def test_read_pid_rejects_malformed_files(tmp_path: Path, text: str):
    spawn.pid_path(tmp_path, "news").parent.mkdir(parents=True)
    spawn.pid_path(tmp_path, "news").write_text(text, encoding="ascii")
    assert spawn.read_pid(tmp_path, "news") is None


def test_unknown_role_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        spawn.write_pid(tmp_path, "desktop", 1)
    with pytest.raises(ValueError):
        spawn.service_status(tmp_path, roles=["desktop"])


def test_service_status_combines_pid_file_and_port(tmp_path: Path):
    spawn.write_pid(tmp_path, "api", 1234)
    spawn.write_pid(tmp_path, "ingestor", 9999)
    ports = {8766: True, 8767: False, 8769: True, 8770: False}

    rows = spawn.service_status(
        tmp_path,
        alive=lambda pid: pid in {1234, 4321},
        probe=lambda port: ports[port],
        pid_for_port=lambda port: 4321 if port == 8769 else None,
    )
    by_role = {row.role: row for row in rows}

    assert [row.role for row in rows] == list(spawn.ROLE_ORDER)
    assert by_role["api"].pid == 1234
    assert by_role["api"].alive and by_role["api"].port_open
    assert by_role["api"].running and not by_role["api"].stale_pid
    # A pid file whose process is gone is stale, even though the port probe
    # still says closed.
    assert by_role["ingestor"].pid == 9999
    assert by_role["ingestor"].stale_pid
    assert not by_role["ingestor"].running
    # No pid file but the port is held: discover the owner.
    assert by_role["news"].pid == 4321
    assert by_role["news"].alive and by_role["news"].running
    assert not by_role["board"].running


def test_status_cli_prints_human_readable_rows(monkeypatch, capsys, tmp_path: Path):
    rows = [
        RoleStatus("api", 8766, 1234, alive=True, port_open=True),
        RoleStatus("board", 8770, None, alive=False, port_open=False),
    ]
    monkeypatch.setattr(cli, "_data_root", lambda config, root: tmp_path)
    monkeypatch.setattr(spawn, "service_status", lambda data, roles=None: rows)

    cli.main(["status"])

    out = capsys.readouterr().out
    assert "api" in out and "pid=1234" in out and "8766" in out
    assert "board" in out and "port" not in out  # Chinese label, not the field name
    assert "8770" in out


def test_status_cli_json(monkeypatch, capsys, tmp_path: Path):
    rows = [RoleStatus("news", 8769, 77, alive=True, port_open=False)]
    monkeypatch.setattr(cli, "_data_root", lambda config, root: tmp_path)
    monkeypatch.setattr(spawn, "service_status", lambda data, roles=None: rows)

    cli.main(["status", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "role": "news",
            "port": 8769,
            "pid": 77,
            "alive": True,
            "port_open": False,
            "running": True,
            "stale_pid": False,
        }
    ]


def test_status_cli_passes_role(monkeypatch, capsys, tmp_path: Path):
    seen = {}

    def fake_status(data, roles=None):
        seen["roles"] = roles
        return []

    monkeypatch.setattr(cli, "_data_root", lambda config, root: tmp_path)
    monkeypatch.setattr(spawn, "service_status", fake_status)

    cli.main(["status", "--role", "api"])
    assert seen["roles"] == ["api"]
    assert capsys.readouterr().out == ""


def test_stop_role_graceful(tmp_path: Path):
    spawn.write_pid(tmp_path, "api", 1234)
    calls: list[bool] = []
    state = {"alive": True}

    def alive(_pid):
        return state["alive"]

    def terminate(pid, *, force):
        calls.append(force)
        state["alive"] = False
        return True

    outcome = spawn.stop_role(
        tmp_path,
        "api",
        alive=alive,
        terminate=terminate,
        probe=lambda port: True,
        sleep=lambda _s: None,
        timeout=0.01,
    )

    assert outcome == StopOutcome("api", 1234, "stopped")
    assert calls == [False]
    assert spawn.read_pid(tmp_path, "api") is None


def test_stop_role_forces_when_graceful_does_not_work(tmp_path: Path):
    spawn.write_pid(tmp_path, "board", 4321)
    calls: list[bool] = []
    state = {"alive": True}

    def alive(_pid):
        return state["alive"]

    def terminate(pid, *, force):
        calls.append(force)
        if force:
            state["alive"] = False
        return True

    outcome = spawn.stop_role(
        tmp_path,
        "board",
        alive=alive,
        terminate=terminate,
        probe=lambda port: True,
        sleep=lambda _s: None,
        timeout=0.01,
    )

    assert outcome.status == "stopped"
    assert calls == [False, True]


def test_stop_role_removes_stale_pid_without_killing(tmp_path: Path):
    spawn.write_pid(tmp_path, "ingestor", 9999)

    def terminate(pid, *, force):
        raise AssertionError("a stale pid must not be signalled")

    outcome = spawn.stop_role(
        tmp_path,
        "ingestor",
        alive=lambda pid: False,
        terminate=terminate,
        probe=lambda port: False,
        sleep=lambda _s: None,
        timeout=0.01,
    )

    assert outcome.status == "not-running"
    assert spawn.read_pid(tmp_path, "ingestor") is None


def test_stop_role_finds_owner_by_port_when_pid_file_is_missing(tmp_path: Path):
    state = {"alive": True}

    def alive(_pid):
        return state["alive"]

    def terminate(pid, *, force):
        state["alive"] = False
        return True

    outcome = spawn.stop_role(
        tmp_path,
        "news",
        alive=alive,
        terminate=terminate,
        probe=lambda port: port == 8769,
        pid_for_port=lambda port: 777,
        sleep=lambda _s: None,
        timeout=0.01,
    )

    assert outcome == StopOutcome("news", 777, "stopped")


def test_stop_role_reports_failure_and_keeps_pid_file(tmp_path: Path):
    spawn.write_pid(tmp_path, "api", 1234)

    outcome = spawn.stop_role(
        tmp_path,
        "api",
        alive=lambda pid: True,
        terminate=lambda pid, *, force: False,
        probe=lambda port: True,
        sleep=lambda _s: None,
        timeout=0.01,
    )

    assert outcome.status == "failed"
    assert outcome.pid == 1234
    assert spawn.read_pid(tmp_path, "api") == 1234


def test_stop_cli_forwards_role(monkeypatch, capsys, tmp_path: Path):
    seen = {}

    def fake_stop(data, roles=None):
        seen["roles"] = roles
        return [StopOutcome("api", 1234, "stopped")]

    monkeypatch.setattr(cli, "_data_root", lambda config, root: tmp_path)
    monkeypatch.setattr(spawn, "stop_services", fake_stop)

    cli.main(["stop", "--role", "api"])

    assert seen["roles"] == ["api"]
    assert "已停止 api (pid 1234)" in capsys.readouterr().out


def test_stop_cli_failure_exits_nonzero(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setattr(cli, "_data_root", lambda config, root: tmp_path)
    monkeypatch.setattr(
        spawn,
        "stop_services",
        lambda data, roles=None: [StopOutcome("api", 1, "failed", "进程仍在运行")],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["stop"])

    assert exc.value.code == 1
    assert "停止失败 api" in capsys.readouterr().err


def test_spawn_service_records_pid(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(spawn, "spawn_detached", lambda *args, **kwargs: 2468)

    pid = spawn.spawn_service("board", root=tmp_path, data_root=tmp_path / "data")

    assert pid == 2468
    assert spawn.read_pid(tmp_path / "data", "board") == 2468
