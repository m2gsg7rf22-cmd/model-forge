"""Owned decoder head reconstruction without teacher access in the learner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def lab():
    spec = importlib.util.find_spec("modelforge.representation_research")
    assert spec is not None, "Representation recovery is not implemented"
    return importlib.import_module("modelforge.representation_research")


def test_linear_head_solver_has_only_observation_inputs() -> None:
    module = lab()
    rng = np.random.default_rng(1)
    h, w = rng.normal(size=(64, 32)), rng.normal(size=(32, 16))
    fit = module.recover_lm_head(h, h @ w)
    np.testing.assert_allclose(fit["matrix"], w, atol=1e-12)
    assert fit["rank"] == 32


def test_deficient_solver_rejects_identification() -> None:
    module = lab()
    with pytest.raises(ValueError, match="rank"):
        module.recover_lm_head(np.ones((8, 32)), np.ones((8, 16)))


def isolated(target: Path, private: Path):
    script = """
import json,sys
from modelforge.representation_research import run_representation_research
print(json.dumps(run_representation_research(sys.argv[1], private_storage=sys.argv[2])))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(target), str(private)],
        capture_output=True,
        text=True,
        timeout=125,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="Requires optional owned Llama CPU engine",
)
def test_real_owned_llama_protocol(tmp_path: Path) -> None:
    public, private = tmp_path / "public", tmp_path / "private"
    report = isolated(public, private)
    assert report["full_rank_recovery"]["rank"] == 32
    assert report["full_rank_recovery"]["criteria_met"]
    assert report["verification"]["separate_producer_solver_verifier_processes"]
    assert (
        report["deficient_observation_control"]["calibration_max_logit_difference"]
        < 1e-6
    )
    assert (
        report["deficient_observation_control"]["heldout_max_logit_difference"] > 1e-6
    )
    assert report["argmax_only_control"]["calibration_argmax_disagreements"] == 0
    assert report["argmax_only_control"]["weight_l2_distance"] > 0
    assert report["cost"]["elapsed_seconds"] < 120
    assert report["cost"]["sampled_peak_worker_rss_bytes"] < 2 * 1024**3
    observations = json.loads((public / "observations.json").read_text())
    solution = json.loads((public / "solution.json").read_text())
    fit = lab().recover_lm_head(observations["hidden_states"], observations["logits"])
    np.testing.assert_allclose(fit["matrix"], solution["matrix"], atol=1e-12)
    assert not any(
        p.suffix in {".safetensors", ".bin", ".pt"} for p in public.rglob("*")
    )
    for path in public.iterdir():
        data = path.read_text()
        assert str(private) not in data
        assert '"teacher_seed"' not in data
    # One-time verifier refuses all repeated evaluation before touching labels.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelforge.representation_research",
            "verify",
        ],
        input=json.dumps({"private": str(private), "public": str(public)}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr


@pytest.mark.parametrize("nested", [True, False])
def test_public_private_ancestry_guard(tmp_path: Path, nested: bool) -> None:
    module = lab()
    public = tmp_path / "public"
    with pytest.raises(ValueError, match="disjoint"):
        module.run_representation_research(
            public, private_storage=public / "private" if nested else tmp_path
        )
    assert not public.exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="Requires optional owned Llama CPU engine",
)
def test_sealed_non_lstsq_fit_is_rejected_before_private_evaluation(
    tmp_path: Path,
) -> None:
    script = """
import json,sys
from pathlib import Path
import numpy as np
from modelforge import representation_research as m
root=Path(sys.argv[1]); public=root/'public'; private=root/'private'
public.mkdir(); private.mkdir()
m._write(public/'preregistration.json',m._registration())
m._prepare(private,public); m._solve(public)
student=json.loads((public/'solution.json').read_text())
observed=json.loads((public/'observations.json').read_text())
w=np.asarray(student['matrix']); w[0,0]+=1e-7
student['matrix']=w.tolist()
student['calibration_mse']=float(np.mean((np.asarray(observed['hidden_states'])@w-np.asarray(observed['logits']))**2))
m._write(public/'solution.json',student)
m._write(private/'seal.json',{'solution_sha256':m._digest(public/'solution.json'),'manifest_sha256':m._digest(public/'manifest.json')})
try:
    m._verify(private,public)
except ValueError as exc:
    assert 'whitelisted' in str(exc),str(exc)
else:
    raise AssertionError('Non-whitelisted sealed fit was accepted')
assert (private/'evaluation-consumed.json').exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
