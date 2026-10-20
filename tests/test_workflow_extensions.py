"""Actual service reconstruction, storage quantization, and fixture-report evaluation."""

import zipfile
import pytest
from modelforge.service import Service


def test_reconstruct_quantize_export_and_report_scoring(tmp_path):
    s = Service(tmp_path / "data")
    original = s.create_project("Original fixture")["id"]
    s.build(original)
    rebuilt = s.create_project("Reconstructed fixture")["id"]
    parity = s.reconstruct(rebuilt, str(s.root / original / "model"))
    assert parity["parity"]["exact_on_all_tested_prompts"]
    with pytest.raises(ValueError):
        s.reconstruct(rebuilt, str(s.root / original / "model"))
    s.quantize(rebuilt)
    archive = tmp_path / "export.zip"
    s.export(rebuilt, str(archive))
    with zipfile.ZipFile(archive) as z:
        assert "quantized/model.safetensors" in z.namelist()
    # Synthetic stored report fixtures, not claims of third-party API observations.
    s._record(
        original,
        "behavior",
        {"results": [{"prompt": "x", "output": "yes", "latency_seconds": 1}]},
    )
    left = s.findings(original)[0]["id"]
    s._record(
        original,
        "behavior",
        {"results": [{"prompt": "x", "output": "no", "latency_seconds": 2}]},
    )
    right = s.findings(original)[0]["id"]
    result = s.compare_experiments(original, left, right)
    assert result["output_change"]["rate"] == 1
    assert s.benchmark(original, left, {"x": "yes"})["score"]["rate"] == 1
    with pytest.raises(ValueError):
        s.compare_experiments(original, "missing", right)
    with pytest.raises(ValueError):
        s.benchmark(original, "missing", {})
    s.close()


def test_resource_setting_validation(tmp_path):
    s = Service(tmp_path)
    for value in [0, 3, True, "2"]:
        with pytest.raises(ValueError):
            s.save_settings({"max_workers": value})
    with pytest.raises(ValueError):
        s.save_settings({"api_endpoint": 9})
    s.save_settings({"max_workers": 1})
    pid = s.create_project("one worker")["id"]
    for i in range(2):
        s.enqueue(pid, "infer", {"text": str(i)})
    assert s.run_pending()["started"] == 1
    assert sum(x["status"] == "queued" for x in s.tasks()) == 1
    s.close()
