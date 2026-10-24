"""Exercise actual observation artifacts, failures, export and MCP dispatch."""

from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest
from modelforge.antipodal import canonical_probe_points
from modelforge.antipodal_io import reconstruct_observations
from modelforge.service import Service


def observation(path: Path) -> dict:
    rng = np.random.default_rng(14)
    g, u, down = (
        rng.normal(size=(5, 3)),
        rng.normal(size=(5, 3)),
        rng.normal(size=(2, 5)),
    )

    def f(x):
        z = x @ g.T
        return (z / (1 + np.exp(-z)) * (x @ u.T)) @ down.T

    x = canonical_probe_points(3)
    data = {
        "schema_version": 1,
        "observation_kind": "isolated_glu_antipodal",
        "family": "bias_free_swiglu",
        "authorized": True,
        "dimension": 3,
        "positive_outputs": f(x).tolist(),
        "negative_outputs": f(-x).tolist(),
    }
    path.write_text(json.dumps(data))
    return data


def test_actual_service_reconstruction_and_export(tmp_path: Path) -> None:
    source = tmp_path / "observations.json"
    observation(source)
    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Paired reconstruction")["id"]
        report = service.reconstruct_antipodal(project, str(source))
        assert not report["full_glu_recovered"] and not report["numeric_certificate"]
        assert not report["heldout_verified"]
        saved = service.root / project / "antipodal-reconstruction"
        artifact = next(saved.glob("*/quadratic-even.json"))
        q = np.asarray(json.loads(artifact.read_text())["quadratic_even"])
        assert q.shape == (2, 3, 3)
        assert np.linalg.norm(q) > 0
        assert service.findings(project)[0]["kind"] == "antipodal-reconstruction"
        out = tmp_path / "export.zip"
        service.export(project, str(out))
        with zipfile.ZipFile(out) as z:
            assert any(name.endswith("quadratic-even.json") for name in z.namelist())
            assert not any(
                "teacher" in name or name.endswith(".npz") for name in z.namelist()
            )
    finally:
        service.close()


@pytest.mark.parametrize(
    "change",
    [
        {"authorized": False},
        {"dimension": True},
        {"weights": [1]},
        {"family": "unknown"},
    ],
)
def test_invalid_or_unauthorized_observations_create_no_artifact(
    tmp_path: Path, change: dict
) -> None:
    source = tmp_path / "observations.json"
    data = observation(source)
    data.update(change)
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        reconstruct_observations(str(source), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_refuse_overwrite_and_record_failed_service_call(tmp_path: Path) -> None:
    source = tmp_path / "observations.json"
    observation(source)
    out = tmp_path / "output"
    reconstruct_observations(str(source), str(out))
    original = (out / "quadratic-even.json").read_bytes()
    with pytest.raises(FileExistsError):
        reconstruct_observations(str(source), str(out))
    assert (out / "quadratic-even.json").read_bytes() == original
    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Failure")["id"]
        with pytest.raises(RuntimeError):
            service.reconstruct_antipodal(project, str(tmp_path / "missing.json"))
        assert service.findings(project)[0]["kind"] == "antipodal-reconstruction-failed"
    finally:
        service.close()


def test_real_mcp_antipodal_tool(tmp_path: Path) -> None:
    from mcp import StdioServerParameters
    from modelforge.mcp_client import safe_mcp_session

    source = tmp_path / "observations.json"
    observation(source)

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "modelforge.mcp_server"],
            env=dict(os.environ, MODELFORGE_HOME=str(tmp_path / "mcp")),
        )
        async with safe_mcp_session(params) as client:
            await client.initialize()
            assert "reconstruct_antipodal" in {
                t.name for t in (await client.list_tools()).tools
            }
            created = await client.call_tool("create_project", {"name": "Antipodal"})
            assert not created.isError
            project = json.loads(created.content[0].text)["id"]
            result = await client.call_tool(
                "reconstruct_antipodal",
                {"project_id": project, "observations_path": str(source)},
            )
            assert not result.isError, result.content
            report = json.loads(result.content[0].text)
            assert report["dimension"] == 3 and not report["original_weights_recovered"]

    asyncio.run(asyncio.wait_for(run(), timeout=30))
