"""Real artifact and subprocess-tested opt-in plugin adapter paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelforge.adapters import (
    AdapterRegistry,
    AudioAdapter,
    ImageAdapter,
    LanguageAdapter,
    ModelAdapter,
    MultimodalAdapter,
    UnsupportedModelOperation,
    VideoAdapter,
    detect_adapter,
)
from modelforge.builder import create_model


@pytest.mark.parametrize(
    "adapter",
    [
        ModelAdapter(),
        MultimodalAdapter(),
        ImageAdapter(),
        VideoAdapter(),
        AudioAdapter(),
    ],
)
def test_metadata_only_adapters(tmp_path: Path, adapter: ModelAdapter) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"model_type":"example"}')
    report = adapter.inspect(str(path))
    assert report["status"] == "inspected"
    assert report["adapter"]["modality"] == adapter.modality
    assert report["adapter"]["inference"] is False
    with pytest.raises(UnsupportedModelOperation, match="no compatible runtime"):
        adapter.infer(str(path), "hello")


def test_language_real_tiny_inference(tmp_path: Path) -> None:
    path = tmp_path / "tiny"
    create_model(path, d_model=4, context_length=8)
    adapter = detect_adapter(adapter_report := LanguageAdapter().inspect(str(path)))
    assert isinstance(adapter, LanguageAdapter)
    assert adapter_report["adapter"]["inference"] is True
    report = adapter.infer(str(path), "hello")
    assert isinstance(report["output"], str)
    assert report["scope"] == "project-owned tiny causal model"
    with pytest.raises(ValueError, match="Prompt"):
        adapter.infer(str(path), "x" * 16001)


def test_language_refuses_arbitrary_imported_models(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"model_type":"llama"}')
    adapter = LanguageAdapter()
    with pytest.raises(UnsupportedModelOperation, match="directory"):
        adapter.infer(str(path), "hello")
    with pytest.raises(
        UnsupportedModelOperation, match="Imported language architecture"
    ):
        adapter.infer(str(tmp_path), "hello")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "llama"}, "language"),
        ({"model_type": "qwen2_vl"}, "multimodal"),
        ({"vision_config": {}, "text_config": {}}, "multimodal"),
        ({"model_type": "whisper"}, "audio"),
        ({"_class_name": "UNet2DConditionModel"}, "image"),
        ({"model_type": "vit"}, "image"),
        ({"model_type": "hunyuanvideo"}, "video"),
        ({"model_type": "unknown-custom"}, "unknown"),
    ],
)
def test_declared_routing(config: dict[str, object], expected: str) -> None:
    assert detect_adapter({"metadata": config}).modality == expected


def test_registry_unknown_and_untrusted_plugin() -> None:
    registry = AdapterRegistry()
    assert registry.get("language").modality == "language"
    with pytest.raises(ValueError, match="Unknown modality"):
        registry.get("not-registered")
    with pytest.raises(PermissionError, match="explicitly trust"):
        registry.load_plugin("anything")
    with pytest.raises(ValueError, match="exactly one"):
        registry.load_plugin("model-forge-test-not-installed", trusted=True)
    assert isinstance(registry.list_plugins(), list)
    assert detect_adapter({"metadata": None}).modality == "unknown"


def test_real_entry_point_discovery_does_not_load_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Install a real distribution declaration on an isolated import path. No
    # importlib mocks: plugin code must stay unloaded until trusted loading.
    dist = tmp_path / "modelforge_fixture-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: modelforge-fixture\nVersion: 1.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[modelforge.adapters]\nfixture = modelforge_fixture_plugin:Fixture\n"
    )
    marker = tmp_path / "imported.json"
    (tmp_path / "modelforge_fixture_plugin.py").write_text(
        "from pathlib import Path\nfrom modelforge.adapters import ModelAdapter\n"
        f"Path({str(marker)!r}).write_text('true')\n"
        "class Fixture(ModelAdapter):\n    modality = 'fixture-test'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = AdapterRegistry()
    declarations = registry.list_plugins()
    assert any(x["name"] == "fixture" for x in declarations)
    assert not marker.exists()
    with pytest.raises(PermissionError):
        registry.load_plugin("fixture")
    assert not marker.exists()
    loaded = registry.load_plugin("fixture", trusted=True)
    assert loaded.modality == "fixture-test"
    assert registry.get("fixture-test") is loaded
    assert json.loads(marker.read_text()) is True
