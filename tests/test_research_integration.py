"""Actual research service persistence and UI controls, separate from GPU claims."""

from __future__ import annotations
from pathlib import Path
import json
import pytest
from modelforge.service import Service


def test_failed_discovery_is_preserved_for_resume(tmp_path: Path) -> None:
    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Failure evidence")["id"]
        with pytest.raises(RuntimeError, match="seed"):
            service.run_discovery(project, seed=-1)
        finding = service.findings(project)[0]
        assert finding["kind"] == "discovery-failed"
        assert finding["data"]["seed"] == -1
        assert finding["data"]["completed"] is False
        assert finding["data"]["error_type"] == "RuntimeError"
    finally:
        service.close()


def test_actual_discovery_persists_public_evidence_and_excludes_private_export(
    tmp_path: Path,
) -> None:
    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Owned blind reconstruction")["id"]
        report = service.run_discovery(project, seed=17)
        assert report["verification"]["process_separated"]
        assert report["verification"]["sealed_test_used_for_selection"] is False
        assert service.findings(project)[0]["kind"] == "discovery"
        assert list((service.root / "private-research").glob("teacher-*/seal.json"))
        destination = tmp_path / "export.zip"
        service.export(project, str(destination))
        import zipfile

        with zipfile.ZipFile(destination) as archive:
            assert any(
                name.startswith("discovery/")
                and name.endswith("/verification-core.json")
                for name in archive.namelist()
            )
            assert any(
                name.startswith("discovery/") and name.endswith("/observations.json")
                for name in archive.namelist()
            )
            assert not any(
                "truth" in name or "seal.json" in name for name in archive.namelist()
            )
            assert not any(
                "teacher-" in name or "private-research" in name
                for name in archive.namelist()
            )
    finally:
        service.close()


def test_attention_service_exports_observations_and_no_private_truth(
    tmp_path: Path,
) -> None:
    import zipfile

    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Attention observation evidence")["id"]
        result = service.run_attention_research(project)
        assert result["engine"]
        assert service.findings(project)[0]["kind"] == "attention-research"
        archive_path = tmp_path / "attention.zip"
        service.export(project, str(archive_path))
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert any(
                name.startswith("attention-research/") and name.endswith(".json")
                for name in names
            )
            assert not any(
                "private-attention" in name or "teacher-" in name for name in names
            )
    finally:
        service.close()


def test_actual_mcp_research_and_recovery_tools(tmp_path: Path) -> None:
    import asyncio
    import os
    import sys
    from mcp import StdioServerParameters
    from modelforge.mcp_client import safe_mcp_session

    async def run() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-c",
                "import faulthandler;faulthandler.dump_traceback_later(15,repeat=True);"
                "from modelforge.mcp_server import main;main()",
            ],
            env=dict(os.environ, MODELFORGE_HOME=str(tmp_path / "mcp")),
        )
        async with safe_mcp_session(parameters) as session:
            print("MCP phase: initialize", flush=True)
            await session.initialize()
            print("MCP phase: list tools", flush=True)
            names = {t.name for t in (await session.list_tools()).tools}
            assert {
                "run_discovery",
                "run_attention_research",
                "run_representation_research",
                "resume_research",
                "benchmark_mlx_cache",
            } <= names
            print("MCP phase: create project", flush=True)
            created = await session.call_tool(
                "create_project", {"name": "MCP attention"}
            )
            assert not created.isError
            project = json.loads(created.content[0].text)["id"]
            print("MCP phase: attention", flush=True)
            result = await session.call_tool(
                "run_attention_research",
                {"project_id": project},
            )
            assert not result.isError, result.content
            print("MCP phase: resume", flush=True)
            resume = await session.call_tool(
                "resume_research",
                {"repository_path": str(Path(__file__).resolve().parents[1])},
            )
            print(f"MCP phase: resume returned; isError={resume.isError}", flush=True)
            if resume.isError:
                print(f"MCP resume error: {resume.content}", flush=True)
            assert not resume.isError
            body = json.loads(resume.content[0].text)
            assert body["execution"] == "report_only"
            assert body["private_model_gate"] == "closed"

    asyncio.run(asyncio.wait_for(run(), timeout=45))


def test_representation_service_exports_only_public_observations(
    tmp_path: Path,
) -> None:
    """Exercise fresh owned decoder workers through the actual service boundary."""
    import importlib.util
    import zipfile

    if importlib.util.find_spec("torch") is None:
        pytest.skip("Requires optional owned decoder CPU engine")
    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Owned decoder head evidence")["id"]
        result = service.run_representation_research(project)
        assert result["full_rank_recovery"]["criteria_met"]
        assert service.findings(project)[0]["kind"] == "representation-research"
        assert list((service.root / "private-representation").glob("*/seal.json"))
        archive_path = tmp_path / "representation.zip"
        service.export(project, str(archive_path))
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert any(
                name.startswith("representation-research/")
                and name.endswith("/observations.json")
                for name in names
            )
            assert not any(
                "private-representation" in name or name.endswith(".safetensors")
                for name in names
            )
    finally:
        service.close()
