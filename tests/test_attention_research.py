"""Known-family attention recovery and identifiability acceptance tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def lab():
    # Explicit assertion gives a real missing-feature RED before implementation.
    spec = importlib.util.find_spec("modelforge.attention_research")
    assert spec is not None, "Attention recovery implementation is missing"
    return importlib.import_module("modelforge.attention_research")


def isolated_experiment(target: Path, private: Path | None = None):
    # Full-suite collection imports Torch. Run the complete experiment in a fresh
    # interpreter so its real 512 MiB verifier guard measures its own runtime.
    script = """
import json
from pathlib import Path
import sys
from modelforge import attention_research as module
target = Path(sys.argv[1])
original = module._one
def checked(*args, **kwargs):
    assert (target / "preregistration.json").exists()
    return original(*args, **kwargs)
module._one = checked
report = module.run_attention_research(target, private_storage=sys.argv[2] if len(sys.argv) > 2 else None)
print(json.dumps(report))
"""
    args = [sys.executable, "-c", script, str(target)]
    if private is not None:
        args.append(str(private))
    completed = subprocess.run(args, capture_output=True, text=True, timeout=65)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def softmax(score):
    exp = np.exp(score - score.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_score_recovery_only_from_authorized_probability_observations() -> None:
    module = lab()
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    truth = np.asarray([[0.2, -0.3], [0.4, 0.1]])
    recovered = module.recover_score_matrix(
        x, softmax(x @ truth @ x.T / np.sqrt(2)), head_dimension=2
    )
    np.testing.assert_allclose(recovered["matrix"], truth, atol=1e-12)
    assert recovered["design_rank"] == 4
    assert recovered["parameter_count"] == 4


def test_rank_deficiency_rejects_identification() -> None:
    module = lab()
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="rank"):
        module.recover_score_matrix(x, softmax(x @ x.T), head_dimension=2)


@pytest.mark.parametrize("bad", [0.0, 1e-13, -0.1, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_probabilities_rejected(bad: float) -> None:
    module = lab()
    p = np.ones((3, 3)) / 3
    p[0, 0] = bad
    with pytest.raises(ValueError, match="probabilit"):
        module.recover_score_matrix(np.eye(3), p, head_dimension=3)


def test_three_seed_independent_process_evidence(tmp_path: Path) -> None:
    report = isolated_experiment(tmp_path / "attention")
    assert len(report["experiments"]) == 3
    for row in report["experiments"]:
        assert row["full_probability_recovery"]["criteria_met"]
        assert row["gauge_symmetry"]["max_output_difference"] < 1e-10
        assert row["gauge_symmetry"]["q_weight_l2_distance"] > 0.1
        assert row["row_constant_invariance"]["max_probability_difference"] < 1e-12
        negative = row["rank_deficient_control"]
        assert negative["training_max_probability_difference"] < 1e-12
        assert negative["unseen_max_probability_difference"] > 1e-5
        assert row["outputs_only_control"]["same_output_max_difference"] < 1e-12
        assert (
            row["outputs_only_control"]["different_probability_max_difference"] > 1e-5
        )
        assert row["verification"]["learner_pid"] != row["verification"]["verifier_pid"]
    assert report["cost"]["elapsed_seconds"] < 60
    assert report["cost"]["sampled_peak_learner_rss_bytes"] < 512 * 1024**2
    assert report["cost"]["remote_requests"] == 0


def test_preregistration_persisted_before_first_teacher_run(tmp_path: Path) -> None:
    target = tmp_path / "attention"
    report = isolated_experiment(target)
    import hashlib

    assert (
        report["preregistration_sha256"]
        == hashlib.sha256((target / "preregistration.json").read_bytes()).hexdigest()
    )
    assert not any("truth" in p.name for p in target.iterdir())


def test_explicit_private_artifacts_are_outside_public_output(tmp_path: Path) -> None:
    public, private = tmp_path / "public", tmp_path / "secret"
    report = isolated_experiment(public, private)
    assert (private / "truth.json").exists()
    assert len(list(private.glob("evaluation-[0-9]*.json"))) == 3
    assert set(p.name for p in public.iterdir()) == {
        "preregistration.json",
        "attention-results.json",
    }
    assert str(private) not in (public / "attention-results.json").read_text()
    assert report["experiments"][0]["verification"][
        "heldout_generated_after_solution_sealed"
    ]


@pytest.mark.parametrize("nested", [True, False])
def test_public_private_directories_must_be_disjoint(
    tmp_path: Path, nested: bool
) -> None:
    module = lab()
    public = tmp_path / "public"
    private = public / "private" if nested else tmp_path
    with pytest.raises(ValueError, match="disjoint"):
        module.run_attention_research(public, private_storage=private)
    assert not public.exists()


def test_repeated_seed_run_does_not_overwrite_private_teacher(tmp_path: Path) -> None:
    module = lab()
    module._one(11, np.random.default_rng(123), module.time.monotonic() + 60, tmp_path)
    teacher = (tmp_path / "teacher-11.json").read_bytes()
    with pytest.raises(FileExistsError):
        module._one(
            11, np.random.default_rng(456), module.time.monotonic() + 60, tmp_path
        )
    assert (tmp_path / "teacher-11.json").read_bytes() == teacher


def test_public_observations_allow_independent_fit_reproduction(tmp_path: Path) -> None:
    module = lab()
    report = isolated_experiment(tmp_path / "public")
    for row in report["experiments"]:
        observed = row["authorized_observations"]
        fit = module.recover_score_matrix(
            observed["inputs"],
            observed["probabilities"],
            head_dimension=observed["head_dimension"],
        )
        np.testing.assert_allclose(
            fit["matrix"], row["student_solution"]["matrix"], atol=1e-12
        )
