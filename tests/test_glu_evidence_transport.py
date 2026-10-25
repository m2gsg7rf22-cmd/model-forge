import importlib.util
from pathlib import Path
import json
import sys
import numpy as np
import pytest
from modelforge.execution import run_json


def path():
    p = Path(__file__).parents[1] / "examples/isolated_glu/evidence_transport.py"
    assert p.exists(), "Compact evidence transport is not implemented"
    return p


def load():
    spec = importlib.util.spec_from_file_location("evidence_transport", path())
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_large_numeric_evidence_crosses_small_json_engine_bound(tmp_path):
    p = path()
    source = f"""import importlib.util,json,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location('transport',{str(p)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.save_evidence(Path('evidence'),{{'large':np.ones(600000)}},{{'synthetic':True}})))
"""
    result = run_json(
        [sys.executable, "-c", source],
        {},
        cwd=tmp_path,
        memory_limit_mb=128,
        timeout_seconds=10,
        output_limit_bytes=2048,
    )
    assert result["evidence_bytes"] > 2_000_000
    assert result["metadata"]["synthetic"] is True
    assert result["process_monitor_scope"] == "process_and_children"
    assert len(json.dumps(result)) < 2048
    with np.load(
        tmp_path / "evidence/numeric-evidence.npz", allow_pickle=False
    ) as data:
        np.testing.assert_array_equal(data["large"], np.ones(600000))
    assert (
        json.loads((tmp_path / "evidence/summary.json").read_text())["evidence_sha256"]
        == result["evidence_sha256"]
    )


def test_exclusive_attempt_finite_bounds_names_and_compact_metadata(tmp_path):
    m = load()
    out = tmp_path / "failed"
    with pytest.raises(ValueError):
        m.save_evidence(out, {"../bad": np.ones(2)}, {})
    assert (out / "attempt.json").exists()
    with pytest.raises(FileExistsError):
        m.save_evidence(out, {"ok": np.ones(2)}, {})
    for i, arrays in enumerate(
        [
            {"bad": np.array([np.nan])},
            {"bad": np.ones(2, dtype=np.float32)},
            {"big": np.ones(4_200_000)},
        ]
    ):
        with pytest.raises(ValueError):
            m.save_evidence(tmp_path / str(i), arrays, {})
    with pytest.raises(ValueError):
        m.save_evidence(tmp_path / "meta", {"ok": np.ones(2)}, {"too_big": "x" * 20000})
    with pytest.raises(ValueError):
        m.save_evidence(tmp_path / "nan", {"ok": np.ones(2)}, {"bad": float("nan")})


def test_symlink_ancestor_rejected_before_writing(tmp_path):
    m = load()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        m.save_evidence(link / "new", {"ok": np.ones(2)}, {})
    assert not (real / "new").exists()
