"""Scientific file protocol, leakage, and one-time evaluation acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modelforge.discovery import (
    _digest,
    _write,
    features,
    preregistration,
    run_discovery,
    solve,
)
from modelforge.discovery_oracle import prepare, verify


def fixture_protocol(tmp_path: Path) -> tuple[Path, Path]:
    private, public = tmp_path / "private", tmp_path / "public"
    private.mkdir()
    public.mkdir()
    _write(private / "config.json", preregistration(17))
    prepare(private, public)
    solve(public)
    _write(
        private / "seal.json",
        {
            "solution_sha256": _digest(public / "solution.json"),
            "manifest_sha256": _digest(public / "manifest.json"),
        },
    )
    return private, public


def test_canonical_feature_basis() -> None:
    np.testing.assert_array_equal(
        features(np.asarray([[2.0, 3.0]])), [[1, 2, 3, 4, 6, 9]]
    )


def test_full_separate_process_recovery_and_negative_controls(tmp_path: Path) -> None:
    report = run_discovery(tmp_path / "run")
    rows = {row["condition"]: row for row in report["results"]}
    assert rows["real_noiseless"]["exact_recovery_criteria_met"]
    assert rows["real_noiseless"]["selected_method"] == "quadratic_lstsq"
    for name in ("real_noise_0.0001", "real_noise_0.1", "sign_only"):
        assert not rows[name]["exact_recovery_criteria_met"]
    witness = rows["sign_only"]["nonidentifiability_witness"]
    assert witness["observed_sign_disagreements"] == 0
    assert witness["coefficient_l2_distance"] > 0
    assert witness["heldout_real_output_mse_between_witnesses"] > 0
    assert report["verification"]["process_separated"]
    assert not report["verification"]["independent_external_scientist"]
    assert report["cost"]["sampled_peak_worker_rss_bytes"] < 512 * 1024**2
    assert report["cost"]["elapsed_seconds"] < 60
    assert report["cost"]["remote_requests"] == 0


def test_public_files_exclude_private_truth_and_test_labels(tmp_path: Path) -> None:
    private, public = fixture_protocol(tmp_path)
    assert sorted(p.name for p in public.iterdir()) == [
        "manifest.json",
        "observations.json",
        "solution.json",
    ]
    observations = json.loads((public / "observations.json").read_text())
    assert all(
        set(row) == {"train_x", "train_y", "validation_x", "validation_y"}
        for row in observations.values()
    )
    for path in public.iterdir():
        value = path.read_text()
        assert '"private_teacher_seed"' not in value
        assert '"tests"' not in value
    # Teacher entropy is not deterministically exposed by public sampling seed.
    truth = json.loads((private / "truth.json").read_text())
    assert truth["private_teacher_seed"] != 17
    teachers = truth["teachers"]
    assert len({json.dumps(value) for value in teachers.values()}) == 4


def test_repeated_evaluation_is_rejected(tmp_path: Path) -> None:
    private, public = fixture_protocol(tmp_path)
    verify(private, public)
    with pytest.raises(FileExistsError):
        verify(private, public)


@pytest.mark.parametrize(
    "filename", ["solution.json", "manifest.json", "observations.json"]
)
def test_tampering_consumes_evaluation_and_is_rejected(
    tmp_path: Path, filename: str
) -> None:
    private, public = fixture_protocol(tmp_path)
    with (public / filename).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="tamper|commitment"):
        verify(private, public)
    assert (private / "evaluation-consumed.json").exists()
    with pytest.raises(FileExistsError):
        verify(private, public)


def test_observation_tampering_rejected_before_solver(tmp_path: Path) -> None:
    _, public = fixture_protocol(tmp_path)
    with (public / "observations.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="Observation commitment"):
        solve(public)


def test_destination_and_seed_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        run_discovery(tmp_path / "run", seed=True)
    with pytest.raises(ValueError, match="new"):
        run_discovery(tmp_path)


def test_verifier_rejects_private_truth_tampering(tmp_path: Path) -> None:
    private, public = fixture_protocol(tmp_path)
    with (private / "truth.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="truth commitment"):
        verify(private, public)


def test_final_envelope_preserves_immutable_verifier_commitment(tmp_path: Path) -> None:
    report = run_discovery(tmp_path / "run")
    core = tmp_path / "run" / "verification-core.json"
    assert report["verification_core_sha256"] == _digest(core)
    verified = json.loads(core.read_text())
    assert "stages" not in verified["cost"]
    assert verified["results"] == report["results"]


def test_private_evaluation_marker_matches_immutable_core(tmp_path: Path) -> None:
    private, public = fixture_protocol(tmp_path)
    verify(private, public)
    marker = json.loads((private / "evaluation-consumed.json").read_text())
    assert marker["verified_core_sha256"] == _digest(public / "verification-core.json")


@pytest.mark.parametrize(
    "corruption", ["score", "selection", "retention", "coefficients"]
)
def test_verifier_recomputes_sealed_learner_selection(
    tmp_path: Path, corruption: str
) -> None:
    private, public = fixture_protocol(tmp_path)
    path = public / "solution.json"
    solution = json.loads(path.read_text())
    row = solution["conditions"]["real_noiseless"]
    if corruption == "score":
        row["candidates"][0]["validation_mse"] = 1e9
    elif corruption == "selection":
        row["selected_method"] = "quadratic_ridge_0.001"
    elif corruption == "retention":
        row["validation_retained_as_exact_candidate"] = False
    else:
        row["coefficients"][0][0] += 0.1
    _write(path, solution)
    _write(
        private / "seal.json",
        {
            "solution_sha256": _digest(path),
            "manifest_sha256": _digest(public / "manifest.json"),
        },
    )
    with pytest.raises(ValueError, match="[Vv]alidation|coefficients"):
        verify(private, public)


def test_private_truth_has_managed_persistent_location(tmp_path: Path) -> None:
    storage = tmp_path / "retained-private"
    report = run_discovery(tmp_path / "published", seed=17, private_storage=storage)
    retained = list(storage.iterdir())
    assert len(retained) == 1 and retained[0].is_dir()
    assert (retained[0] / "seal.json").is_file()
    assert "persistent" in report["private_truth_retention"]
    assert str(storage) not in json.dumps(report)
