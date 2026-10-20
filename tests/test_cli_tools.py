"""Real CLI dispatch and in-process MCP tool service acceptance."""

import asyncio
import json
import sys
from modelforge.cli import main
from modelforge.mcp_server import create_server
from modelforge.service import Service


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
    run("quantize", pid)
    rebuilt = run("create", "CLI rebuild")["id"]
    run("reconstruct", rebuilt, str(home / pid / "model"))
    model = home / pid / "model"
    run("inspect", pid, str(model))
    assert run("findings", pid)
    run("enqueue", pid, "infer", '{"text":"hi"}')
    assert run("tasks")[0]["status"] == "queued"
    assert run("run-tasks")["tasks"][0]["status"] == "completed"
    assert run("export", pid, str(tmp_path / "cli.zip"))["sha256"]
    fixture_service = Service(home)
    fixture_service._record(
        pid, "behavior", {"results": [{"prompt": "x", "output": "yes"}]}
    )
    finding_id = fixture_service.findings(pid)[0]["id"]
    assert run("benchmark", pid, finding_id, '{"x":"yes"}')["score"]["rate"] == 1
    assert (
        run("compare-experiments", pid, finding_id, finding_id)["output_change"]["rate"]
        == 0
    )
    fixture_service.close()
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
        await call("quantize_model", {"project_id": pid})
        rebuilt = await call("create_project", {"name": "MCP rebuild"})
        await call(
            "reconstruct_model",
            {"project_id": rebuilt["id"], "source_path": str(tmp_path / pid / "model")},
        )
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
