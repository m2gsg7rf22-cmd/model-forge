"""Real CLI dispatch and in-process MCP tool service acceptance."""

import asyncio
import json
import sys
from modelforge.cli import main
from modelforge.mcp_server import create_server


def test_cli_commands(tmp_path, monkeypatch, capsys):
    home = tmp_path / "data"

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["modelforge", "--home", str(home), *args])
        main()
        return json.loads(capsys.readouterr().out)

    pid = run("create", "CLI test")["id"]
    assert run("projects")[0]["id"] == pid
    assert run("hardware")["memory_available"] > 0
    run("build", pid)
    assert run("infer", pid, "hello")["output"] is not None
    run("train", pid, "hello world " * 5, "--steps", "1")
    model = home / pid / "model"
    run("inspect", pid, str(model))
    assert run("findings", pid)
    run("enqueue", pid, "infer", '{"text":"hi"}')
    assert run("tasks")[0]["status"] == "queued"
    assert run("run-tasks")["tasks"][0]["status"] == "completed"
    assert run("export", pid, str(tmp_path / "cli.zip"))["sha256"]
    example = run("example", "--destination", str(tmp_path / "example.zip"))
    assert example["export"]["generated_model_files"] > 0


def test_mcp_tools_in_process(tmp_path):
    async def run():
        server = create_server(str(tmp_path))

        async def call(name, arguments):
            result = await server.call_tool(name, arguments)
            # SDK structured tool conversion can return (content, structured).
            if isinstance(result, tuple):
                return result[1]
            if isinstance(result, dict):
                return result
            return json.loads(result[0].text)

        created = await call("create_project", {"name": "MCP direct"})
        pid = created["id"]
        assert await call("list_projects", {})
        assert await call("hardware_status", {})
        await call("build_model", {"project_id": pid})
        await call("generate", {"project_id": pid, "text": "hello"})
        await call("train_model", {"project_id": pid, "text": "hello " * 8, "steps": 1})
        await call(
            "inspect_model", {"project_id": pid, "path": str(tmp_path / pid / "model")}
        )
        assert await call("read_findings", {"project_id": pid})
        await call(
            "enqueue_task",
            {"project_id": pid, "kind": "infer", "payload": {"text": "hi"}},
        )
        await call("run_tasks", {})
        assert await call("task_status", {})
        await call(
            "export_project",
            {"project_id": pid, "destination": str(tmp_path / "mcp.zip")},
        )

    asyncio.run(run())
