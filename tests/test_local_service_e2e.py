"""Actual local engines through persistent Service and the stdio MCP transport."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from modelforge.service import Service

HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "peft")
)
HAS_MLX = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_lm") is not None
)


@pytest.fixture
def service(tmp_path: Path) -> Any:
    value = Service(tmp_path / "data")
    value.save_settings({"runtime_python": sys.executable})
    try:
        yield value
    finally:
        value.close()


@pytest.mark.skipif(not HAS_HF, reason="Optional HF/PEFT local engines are required")
def test_owned_llama_train_resume_merge_export_reload(
    service: Service, tmp_path: Path
) -> None:
    from modelforge.runtime_hf import generate

    project = service.create_project("Actual local training workflow")["id"]
    built = service.create_local_fixture(project, "llama")
    assert built["project_owned"]
    assert built["preflight"]["complete_checkpoint"]
    initial = service.local_generate(project, "hello world", 3)
    assert initial["new_tokens"] == 3
    text = "hello world hello world hello world"
    options = {
        "text": text,
        "method": "lora",
        "steps": 2,
        "learning_rate": 0.1,
        "authorized": True,
        "rights": "Project-owned random weights and authored text; MIT",
    }
    trained = service.train_local_model(project, **options)
    assert trained["completed_steps"] == 2
    resumed = service.train_local_model(project, **options, resume=True)
    assert resumed["completed_steps"] == 4
    assert Path(resumed["artifact_path"]).is_dir()
    adapted = service.local_generate(project, "hello world", 3)
    assert adapted["new_tokens"] == 3
    with pytest.raises(RuntimeError, match="redistribution rights"):
        service.reconstruct_local(project, ["hello world"])
    assert not (service.root / project / "hf-reconstruction").exists()
    before = tmp_path / "before.zip"
    service.export(project, str(before))
    with zipfile.ZipFile(before) as archive:
        assert not any(name.endswith(".safetensors") for name in archive.namelist())
    merged = service.reconstruct_local(
        project, ["hello world"], redistributable=True, rights=options["rights"]
    )
    destination = Path(merged["destination"])
    assert (destination / "model.safetensors").is_file()
    assert (destination / "tokenizer.json").is_file()
    parity = generate(destination, "hello world", 3, device="cpu")
    assert parity["token_ids"] == adapted["token_ids"]
    exported = tmp_path / "with-authorized-weights.zip"
    service.export(project, str(exported))
    with zipfile.ZipFile(exported) as archive:
        assert "hf-reconstruction/model.safetensors" in archive.namelist()
        assert not any(
            name.startswith(("local-model/", "local-training/"))
            for name in archive.namelist()
        )
        extracted = tmp_path / "reload"
        archive.extractall(extracted)
    assert (
        generate(extracted / "hf-reconstruction", "hello world", 3, device="cpu")[
            "token_ids"
        ]
        == adapted["token_ids"]
    )
    reopened = Service(service.root)
    try:
        assert (
            reopened.local_generate(project, "hello world", 3)["token_ids"]
            == adapted["token_ids"]
        )
        assert len(reopened.findings(project)) >= 6
    finally:
        reopened.close()


@pytest.mark.skipif(
    not (HAS_HF and HAS_MLX),
    reason="Real MLX requires the optional Apple Silicon engine",
)
def test_service_quantized_mlx_training_and_adapter_reload(service: Service) -> None:
    project = service.create_project("Actual MLX workflow")["id"]
    service.create_local_fixture(project, "llama")
    cpu = service.local_generate(project, "hello world", 1, backend="mlx", device="cpu")
    assert cpu["device"] == "cpu"
    trained = service.train_mlx_model(
        project,
        "hello world hello world hello world",
        steps=3,
        learning_rate=0.1,
        authorized=True,
        rights="Project-owned random weights and authored text; MIT",
        quantize_bits=4,
    )
    assert trained["quantize_bits"] == 4
    assert trained["final_loss"] < trained["initial_loss"]
    result = service.local_generate(project, "hello world", 3, backend="mlx")
    assert result["adapter_verified"]
    assert result["device"] == "gpu"
    assert result["new_tokens"] == 3
    with pytest.raises(ValueError, match="GPU"):
        service.local_generate(project, "hello world", 1, backend="mlx", device="cpu")
    with pytest.raises(ValueError, match="MLX backend"):
        service.local_generate(project, "hello world", backend="hf")
    with pytest.raises(ValueError, match="unsupported"):
        service.reconstruct_local(project, ["hello"])


@pytest.mark.skipif(
    not HAS_HF, reason="Optional local engines are required for stdio workflow"
)
def test_real_new_mcp_tools_with_local_preflight_and_research(tmp_path: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run() -> None:
        environment = dict(
            os.environ,
            MODELFORGE_HOME=str(tmp_path / "mcp-data"),
            MODELFORGE_ENGINE_PYTHON=sys.executable,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "modelforge.mcp_server"],
            env=environment,
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                names = {item.name for item in (await client.list_tools()).tools}
                assert {
                    "create_local_fixture",
                    "attach_local_model",
                    "local_generate",
                    "run_umrre",
                    "train_mlx_model",
                } <= names

                async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    result = await client.call_tool(name, arguments)
                    assert not result.isError, result.content
                    content = result.content[0]
                    assert content.type == "text"
                    value: dict[str, Any] = json.loads(content.text)
                    return value

                project = (await call("create_project", {"name": "Real local MCP"}))[
                    "id"
                ]
                built = await call(
                    "create_local_fixture",
                    {"project_id": project, "model_type": "llama"},
                )
                checked = await call(
                    "attach_local_model",
                    {
                        "project_id": project,
                        "source_path": built["path"],
                        "authorized": True,
                    },
                )
                assert checked["complete_checkpoint"]
                inference = await call(
                    "local_generate",
                    {"project_id": project, "text": "hello world", "max_new_tokens": 2},
                )
                assert inference["new_tokens"] == 2
                research = await call("run_umrre", {"project_id": project})
                assert research

    asyncio.run(asyncio.wait_for(run(), timeout=180))
