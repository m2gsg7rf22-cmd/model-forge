"""Integration preparation and ownership boundaries; no host settings touched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "modelforge_bootstrap", Path(__file__).parents[1] / "packaging/bootstrap.py"
)
assert spec and spec.loader
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def state_for(prefix, client="claude"):
    return {
        "prefix": str(prefix),
        "client": client,
        "server_name": "model-forge-isolated-123",
        "scope": "local",
        "applied": False,
        "engines": [],
        "desktop": False,
    }


def test_claude_preparation_absolute_real_module(tmp_path):
    prefix = tmp_path / "env"
    prefix.mkdir()
    state = state_for(prefix)
    target = bootstrap.prepare(prefix, state)
    mcp = json.loads((target / ".mcp.json").read_text())
    server = mcp["mcpServers"][state["server_name"]]
    assert Path(server["command"]).is_absolute()
    assert server["args"] == ["-m", "modelforge.mcp_server"]
    assert (target / "skills/model-forge/SKILL.md").is_file()
    assert (
        json.loads((target / ".claude-plugin/marketplace.json").read_text())["name"]
        == state["server_name"] + "-marketplace"
    )
    commands = bootstrap.registration_commands(state, target, "claude")
    assert commands[0][-2:] == ["--scope", "local"]
    assert commands[1][-2:] == ["--scope", "local"]


def test_codex_preparation_and_commands(tmp_path):
    state = state_for(tmp_path, "codex")
    target = bootstrap.prepare(tmp_path, state)
    assert "args = " in (target / "config.toml").read_text()
    command = bootstrap.registration_commands(state, target, "codex")[0]
    assert command[:4] == ["codex", "mcp", "add", state["server_name"]]
    assert command[-2:] == ["-m", "modelforge.mcp_server"]


def test_reject_unmarked_or_mismatched_installs(tmp_path):
    with pytest.raises(ValueError):
        bootstrap._read_state(tmp_path)
    (tmp_path / ".modelforge-install").touch()
    with pytest.raises(ValueError):
        bootstrap._read_state(tmp_path)
    bootstrap._save(tmp_path, state_for(tmp_path / "elsewhere"))
    with pytest.raises(ValueError):
        bootstrap._read_state(tmp_path)


def test_existing_prefix_never_overwritten(tmp_path):
    preserved = tmp_path / "unrelated"
    preserved.write_text("keep")
    with pytest.raises(ValueError):
        bootstrap.install(tmp_path, client="claude", engines=[])
    assert preserved.read_text() == "keep"


def test_uninstall_connected_requires_explicit_detach(tmp_path, monkeypatch):
    (tmp_path / ".modelforge-install").touch()
    state = state_for(tmp_path)
    state["applied"] = True
    bootstrap._save(tmp_path, state)

    def forbidden(*args, **kwargs):
        raise AssertionError("No CLI change allowed")

    monkeypatch.setattr(bootstrap, "_run", forbidden)
    with pytest.raises(ValueError):
        bootstrap.uninstall(tmp_path)


def test_apply_executes_only_owned_names(tmp_path, monkeypatch):
    state = state_for(tmp_path, "codex")
    state["prepared"] = str(bootstrap.prepare(tmp_path, state))
    calls = []

    def record(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(bootstrap, "_run", record)
    bootstrap.apply(tmp_path, state, executable="codex", project=tmp_path)
    assert state["applied"]
    assert calls[0][3] == state["server_name"]
    with pytest.raises(ValueError):
        bootstrap.apply(tmp_path, state, executable="codex")


def test_install_verifies_before_optional_host_changes(tmp_path, monkeypatch):
    prefix = tmp_path / "new"
    calls = []

    def simulate(command, **kwargs):
        calls.append(command)
        if "install.py" in command[1]:
            prefix.mkdir()
            (prefix / ".modelforge-install").touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(bootstrap, "_run", simulate)
    monkeypatch.setattr(
        bootstrap,
        "verify",
        lambda p: {"tool_count": 17, "verified_calls": ["generate"]},
    )
    state = bootstrap.install(prefix, client="claude", engines=[])
    assert not state["applied"]
    assert len(calls) == 1
    assert state["verification"]["tool_count"] == 17


def test_claude_partial_registration_rolls_back_own_marketplace(tmp_path, monkeypatch):
    state = state_for(tmp_path)
    state["prepared"] = str(bootstrap.prepare(tmp_path, state))
    calls = []

    def simulate(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["plugin", "install"]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(bootstrap, "_run", simulate)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.apply(tmp_path, state, executable="claude", project=tmp_path)
    assert not state["applied"]
    assert calls[-1][1:4] == ["plugin", "marketplace", "remove"]
    assert calls[-1][4] == state["server_name"] + "-marketplace"
    assert calls[-1][-2:] == ["--scope", "local"]
