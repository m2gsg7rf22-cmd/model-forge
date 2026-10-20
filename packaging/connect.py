#!/usr/bin/env python3
"""Prepare isolated integrations; host registration requires explicit --apply."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("client", choices=["claude", "codex"])
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument(
        "--apply",
        action="store_true",
        help="Approve invoking host CLI to update its registration",
    )
    a = p.parse_args()
    command = shutil.which("modelforge-mcp")
    if command is None:
        p.error("Activate the ModelForge installation environment first")
    source = Path(__file__).resolve().parents[1]
    dest = a.destination.expanduser().resolve()
    if dest.exists():
        p.error("Destination already exists; existing settings will not be overwritten")
    dest.mkdir(parents=True)
    if a.client == "claude":
        for name in [".claude-plugin", "skills", "agents", "commands"]:
            shutil.copytree(source / name, dest / name)
        (dest / ".mcp.json").write_text(
            json.dumps(
                {"mcpServers": {"model-forge": {"command": command, "args": []}}},
                indent=2,
            )
        )
        registration = ["claude", "plugin", "marketplace", "add", str(dest)]
        install = ["claude", "plugin", "install", "model-forge@model-forge-marketplace"]
    else:
        shutil.copytree(source / "skills" / "model-forge", dest / "model-forge")
        (dest / "config.toml").write_text(
            "[mcp_servers.model-forge]\ncommand = " + json.dumps(command) + "\n"
        )
        registration = ["codex", "mcp", "add", "model-forge", "--", command]
        install = []
    print("Prepared:", dest)
    print("Registration:", registration)
    if a.client == "codex":
        print(
            "Copy model-forge skill folder into your project .agents/skills/ to enable discovery."
        )
    if a.apply:
        subprocess.run(registration, check=True)
        if install:
            subprocess.run(install, check=True)
    else:
        print(
            "Host settings unchanged. Re-run with a NEW destination and --apply to approve host registration."
        )


if __name__ == "__main__":
    main()
