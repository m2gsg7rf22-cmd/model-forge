"""Actual agent transports using only public known calibration; no secret labels."""

import asyncio
import json
import os
import subprocess
import sys
import zipfile
import numpy as np

pytest_plugins = ["test_glu32"]


def test_actual_cli_reconstruct_predict_export_reload(public_case, tmp_path):
    from modelforge.glu32_io import load_candidate
    from modelforge.glu32 import predict

    root, _ = public_case
    env = dict(os.environ, MODELFORGE_PRIVATE_HOME=str(tmp_path / "unused-private"))

    def call(*args):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "modelforge.cli",
                "--home",
                str(tmp_path / "cli"),
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env=env,
            cwd=tmp_path,
        )
        return json.loads(r.stdout)

    project = call("create", "GLU32 CLI")["id"]
    report = call(
        "engine",
        "reconstruct_glu32",
        json.dumps(
            {
                "project_id": project,
                "observations_path": str(root / "public/observations.json"),
            }
        ),
    )
    result = call(
        "engine",
        "predict_glu32",
        json.dumps(
            {
                "project_id": project,
                "artifact_id": report["artifact_id"],
                "inputs": [[0.2] * 32],
            }
        ),
    )
    assert report["candidate_built"] and not report["heldout_verified"]
    assert report["process_monitor_scope"] == "process_and_children"
    assert report["memory_limit_mb"] == 512
    assert any(f["kind"] == "glu32-reconstruction" for f in call("findings", project))
    archive = tmp_path / "cli.zip"
    call("export", project, str(archive))
    with zipfile.ZipFile(archive) as z:
        assert not any("hessians.npz" in n for n in z.namelist())
        z.extractall(tmp_path / "reload")
    g, c = load_candidate(
        tmp_path / "reload/glu32-reconstruction" / report["artifact_id"]
    )
    np.testing.assert_allclose(
        predict(g, c, [[0.2] * 32]), result["outputs"], rtol=1e-14
    )


def test_actual_mcp_schema_reconstruct_predict_export(public_case, tmp_path):
    from mcp import StdioServerParameters
    from modelforge.mcp_client import safe_mcp_session

    root, _ = public_case

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "modelforge.mcp_server"],
            env=dict(
                os.environ,
                MODELFORGE_HOME=str(tmp_path / "mcp"),
                MODELFORGE_PRIVATE_HOME=str(tmp_path / "unused-private"),
            ),
        )
        async with safe_mcp_session(params) as client:
            await client.initialize()
            schemas = {t.name: t.inputSchema for t in (await client.list_tools()).tools}
            assert set(schemas["reconstruct_glu32"]["required"]) == {
                "project_id",
                "observations_path",
            }

            async def call(name, args):
                result = await client.call_tool(name, args)
                assert not result.isError, result.content
                if result.structuredContent is not None:
                    data = result.structuredContent
                    return data["result"] if set(data) == {"result"} else data
                return json.loads(result.content[0].text)

            project = (await call("create_project", {"name": "GLU32 MCP"}))["id"]
            result = await call(
                "reconstruct_glu32",
                {
                    "project_id": project,
                    "observations_path": str(root / "public/observations.json"),
                },
            )
            predicted = await call(
                "predict_glu32",
                {
                    "project_id": project,
                    "artifact_id": result["artifact_id"],
                    "inputs": [[0.1] * 32],
                },
            )
            assert (
                len(predicted["outputs"][0]) == 32
                and not predicted["reference_verified"]
            )
            assert (
                not result["original_weights_recovered"]
                and not result["heldout_verified"]
            )
            findings = await call("read_findings", {"project_id": project})
            assert any(f["kind"] == "glu32-reconstruction" for f in findings)
            await call(
                "export_project",
                {"project_id": project, "destination": str(tmp_path / "mcp.zip")},
            )
            with zipfile.ZipFile(tmp_path / "mcp.zip") as z:
                assert any(n.endswith("student.npz") for n in z.namelist())
                assert not any(
                    "teacher.npz" in n or "multiscale-observations.npz" in n
                    for n in z.namelist()
                )

    asyncio.run(asyncio.wait_for(run(), timeout=90))
