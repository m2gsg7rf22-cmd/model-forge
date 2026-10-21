#!/usr/bin/env python3
"""Dedicated GitHub-checkout install, real MCP verification, scoped host connection.

Run using Python 3.12+. Default prepares integration without editing host settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
STATE = ".modelforge-bootstrap.json"

# Executed by the installed environment, not the checkout interpreter.
VERIFY_SCRIPT = r"""
import asyncio,json,os,sys
from mcp import StdioServerParameters
from modelforge.mcp_client import safe_mcp_session
async def check():
    env=dict(os.environ)
    env.pop("PYTHONPATH",None)
    env["MODELFORGE_HOME"]=sys.argv[1]
    async with safe_mcp_session(StdioServerParameters(command=sys.executable,args=["-m","modelforge.mcp_server"],env=env)) as session:
        await session.initialize()
        tools=await session.list_tools()
        names=sorted(t.name for t in tools.tools)
        required={"create_project","build_model","generate","hardware_status"}
        if not required.issubset(names):
            raise RuntimeError("Required ModelForge engine tools are missing")
        hardware=await session.call_tool("hardware_status",{})
        if hardware.isError:
            raise RuntimeError("Real hardware tool invocation failed")
        project=await session.call_tool("create_project",{"name":"Bootstrap isolated engine verification"})
        if project.isError:
            raise RuntimeError("Real project tool invocation failed")
        result=project.structuredContent
        if not result:
            result=json.loads(next(c.text for c in project.content if c.type=="text"))
        if "result" in result:
            result=result["result"]
        project_id=result["id"]
        built=await session.call_tool("build_model",{"project_id":project_id})
        generated=await session.call_tool("generate",{"project_id":project_id,"text":"hello"})
        if built.isError or generated.isError:
            raise RuntimeError("Real build/inference invocation failed")
        print(json.dumps({"transport":"stdio","tool_count":len(names),"tools":names,"verified_calls":["hardware_status","create_project","build_model","generate"],"engines":"project-owned byte model; optional heavyweight runtimes are listed, not certified"}))
asyncio.run(asyncio.wait_for(check(),timeout=90))
"""


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    timeout: int = 1200,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        timeout=timeout,
        capture_output=capture,
    )


def _python(prefix: Path) -> Path:
    return prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _read_state(prefix: Path) -> dict[str, Any]:
    if prefix.is_symlink() or not (prefix / ".modelforge-install").is_file():
        raise ValueError("Expected a dedicated marked ModelForge installation")
    state_path = prefix / STATE
    if not state_path.is_file() or state_path.is_symlink():
        raise ValueError("This installation was not created by bootstrap")
    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("prefix") != str(prefix):
        raise ValueError("Installation identity does not match prefix")
    return state


def _save(prefix: Path, state: dict[str, Any]) -> None:
    tmp = prefix / (STATE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, prefix / STATE)


def verify(prefix: Path) -> dict[str, Any]:
    """MCP SDK initialize/list/tools/calls against the installed package in isolated data."""
    with tempfile.TemporaryDirectory(prefix="modelforge-mcp-verification-") as data:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        completed = _run(
            [str(_python(prefix)), "-c", VERIFY_SCRIPT, data],
            cwd=Path(data),
            capture=True,
            timeout=120,
            env=env,
        )
        result: dict[str, Any] = json.loads(completed.stdout)
        return result


def prepare(prefix: Path, state: dict[str, Any]) -> Path:
    """New integration generation; absolute MCP command, no host configuration writes."""
    name = state["server_name"]
    root = prefix / "integrations"
    root.mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="prepared-", dir=root))
    command = str(_python(prefix))
    if state["client"] == "claude":
        for folder in [".claude-plugin", "skills", "agents", "commands"]:
            shutil.copytree(SOURCE / folder, target / folder)
        marketplace_path = target / ".claude-plugin/marketplace.json"
        marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
        marketplace["name"] = name + "-marketplace"
        marketplace_path.write_text(json.dumps(marketplace, indent=2), encoding="utf-8")
        plugin_path = target / ".claude-plugin/plugin.json"
        plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
        # Inline manifests override defaults in some supported host versions.
        plugin.pop("mcpServers", None)
        plugin_path.write_text(json.dumps(plugin, indent=2), encoding="utf-8")
    else:
        shutil.copytree(SOURCE / "skills/model-forge", target / "model-forge")
        (target / "config.toml").write_text(
            "[mcp_servers."
            + name
            + "]\ncommand = "
            + json.dumps(command)
            + '\nargs = ["-m", "modelforge.mcp_server"]\n',
            encoding="utf-8",
        )
    (target / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    name: {"command": command, "args": ["-m", "modelforge.mcp_server"]}
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def registration_commands(
    state: dict[str, Any], prepared: Path, executable: str
) -> list[list[str]]:
    if state["client"] == "claude":
        marketplace = state["server_name"] + "-marketplace"
        return [
            [
                executable,
                "plugin",
                "marketplace",
                "add",
                str(prepared),
                "--scope",
                state["scope"],
            ],
            [
                executable,
                "plugin",
                "install",
                "model-forge@" + marketplace,
                "--scope",
                state["scope"],
            ],
        ]
    return [
        [
            executable,
            "mcp",
            "add",
            state["server_name"],
            "--",
            str(_python(Path(state["prefix"]))),
            "-m",
            "modelforge.mcp_server",
        ]
    ]


def install(
    prefix: Path, *, client: str, engines: list[str], desktop: bool = False
) -> dict[str, Any]:
    if sys.version_info < (3, 12):
        raise ValueError("Python 3.12+ is required")
    if client not in {"claude", "codex"} or any(
        e not in {"llm", "mlx"} for e in engines
    ):
        raise ValueError("Unsupported client or engine")
    if "mlx" in engines and not (
        sys.platform == "darwin" and __import__("platform").machine() == "arm64"
    ):
        raise ValueError("MLX requires macOS Apple Silicon")
    if prefix.exists() or prefix.is_symlink():
        raise ValueError("Use a new dedicated installation prefix")
    if any(parent.is_symlink() for parent in prefix.parents):
        raise ValueError("Installation prefix ancestors must not be symlinks")
    command = [
        sys.executable,
        str(SOURCE / "packaging/install.py"),
        "--prefix",
        str(prefix),
    ]
    if not desktop:
        command.append("--core-only")
    _run(command)
    name = "model-forge-" + hashlib.sha256(str(prefix).encode()).hexdigest()[:10]
    state: dict[str, Any] = {
        "prefix": str(prefix),
        "client": client,
        "engines": sorted(set(engines)),
        "desktop": desktop,
        "server_name": name,
        "scope": "local",
        "applied": False,
        "source": str(SOURCE),
    }
    _save(prefix, state)
    if engines:
        extras = ",".join(sorted(set(engines)))
        _run(
            [
                str(_python(prefix)),
                "-m",
                "pip",
                "install",
                str(SOURCE) + "[" + extras + "]",
            ]
        )
    state["verification"] = verify(prefix)
    state["prepared"] = str(prepare(prefix, state))
    _save(prefix, state)
    return state


def apply(
    prefix: Path,
    state: dict[str, Any],
    *,
    executable: str | None = None,
    scope: str = "local",
    project: Path | None = None,
) -> None:
    if scope not in {"local", "project", "user"}:
        raise ValueError("Unsupported scope")
    program = executable or shutil.which(state["client"])
    if not program:
        raise ValueError("Install the chosen host CLI, or supply --client-executable")
    if state["applied"]:
        raise ValueError("Installation is already connected; update instead")
    # Prefix-derived names avoid replacing any pre-existing standard ModelForge connection.
    state["scope"] = scope
    state["project"] = str((project or Path.cwd()).resolve())
    completed = 0
    try:
        for command in registration_commands(state, Path(state["prepared"]), program):
            _run(command, cwd=Path(state["project"]), timeout=120)
            completed += 1
    except subprocess.SubprocessError:
        if state["client"] == "claude" and completed == 1:
            # Remove only the unique marketplace added immediately above.
            _run(
                [
                    program,
                    "plugin",
                    "marketplace",
                    "remove",
                    state["server_name"] + "-marketplace",
                    "--scope",
                    state["scope"],
                ],
                cwd=Path(state["project"]),
                timeout=120,
            )
        raise
    state["applied"] = True
    state["client_executable"] = program
    _save(prefix, state)


def update(prefix: Path) -> dict[str, Any]:
    state = _read_state(prefix)
    extras = state["engines"] + (["desktop"] if state["desktop"] else [])
    specification = str(SOURCE) + ("[" + ",".join(extras) + "]" if extras else "")
    _run([str(_python(prefix)), "-m", "pip", "install", "--upgrade", specification])
    state["verification"] = verify(prefix)
    # Registered command remains the same; engine updates load from that environment.
    # Plugin skills/agents remain pinned until explicitly re-prepared/reconnected.
    state["source"] = str(SOURCE)
    _save(prefix, state)
    return state


def uninstall(prefix: Path, *, detach: bool = False) -> None:
    state = _read_state(prefix)
    if state["applied"] and not detach:
        raise ValueError(
            "Connected install requires --apply to detach before uninstall"
        )
    if state["applied"]:
        program = state["client_executable"]
        if state["client"] == "claude":
            marketplace = state["server_name"] + "-marketplace"
            _run(
                [
                    program,
                    "plugin",
                    "uninstall",
                    "model-forge@" + marketplace,
                    "--scope",
                    state["scope"],
                ],
                cwd=Path(state["project"]),
                timeout=120,
            )
            _run(
                [
                    program,
                    "plugin",
                    "marketplace",
                    "remove",
                    marketplace,
                    "--scope",
                    state["scope"],
                ],
                cwd=Path(state["project"]),
                timeout=120,
            )
        else:
            _run([program, "mcp", "remove", state["server_name"]], timeout=120)
    _run(
        [
            sys.executable,
            str(SOURCE / "packaging/uninstall.py"),
            "--prefix",
            str(prefix),
            "--yes",
        ],
        timeout=120,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["install", "verify", "connect", "update", "uninstall"]
    )
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--client", choices=["claude", "codex"], default="claude")
    parser.add_argument("--engines", nargs="*", choices=["llm", "mlx"], default=[])
    parser.add_argument("--desktop", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Authorize host CLI configuration changes; required for connect/detach",
    )
    parser.add_argument(
        "--scope",
        choices=["local", "project", "user"],
        default="local",
        help="Claude plugin scope; Codex MCP registrations are user-level",
    )
    parser.add_argument("--project", type=Path)
    parser.add_argument("--client-executable")
    args = parser.parse_args()
    # Preserve symlink identity for rejection before canonicalization.
    prefix = args.prefix.expanduser().absolute()
    try:
        if args.action == "install":
            state = install(
                prefix, client=args.client, engines=args.engines, desktop=args.desktop
            )
            if args.apply:
                apply(
                    prefix,
                    state,
                    executable=args.client_executable,
                    scope=args.scope,
                    project=args.project,
                )
        elif args.action == "verify":
            _read_state(prefix)
            state = verify(prefix)
        elif args.action == "update":
            state = update(prefix)
        elif args.action == "connect":
            state = _read_state(prefix)
            if not args.apply:
                parser.error(
                    "connect requires --apply because it modifies host configuration"
                )
            apply(
                prefix,
                state,
                executable=args.client_executable,
                scope=args.scope,
                project=args.project,
            )
        else:
            uninstall(prefix, detach=args.apply)
            state = {"removed": str(prefix), "research_data_retained": True}
        print(json.dumps(state, indent=2))
    except (ValueError, subprocess.SubprocessError, OSError) as error:
        parser.exit(1, f"ModelForge bootstrap failed: {error}\n")


if __name__ == "__main__":
    main()
