"""Regressions for experimental access/rank claims, not model equivalence."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_zero_hessians_do_not_create_claimed_gate_subspace(tmp_path: Path) -> None:
    np.savez(tmp_path / "hessians.npz", hessian=np.zeros((64, 32, 32, 32)))
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/isolated_glu/hessian_pilot.py"),
            "solve",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert p.returncode != 0, (
        "Uninformative observations must not be treated as rank128"
    )
    assert "subspace" in p.stderr.lower()
    assert not (tmp_path / "solution.npz").exists()


def test_noise_without_spectral_gap_is_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(240361)
    h = rng.normal(size=(64, 32, 32, 32))
    h = 0.5 * (h + h.swapaxes(-1, -2))
    np.savez(tmp_path / "hessians.npz", hessian=h)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/isolated_glu/hessian_pilot.py"),
            "solve",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert p.returncode != 0
    assert "subspace" in p.stderr.lower()
    assert not (tmp_path / "solution.npz").exists()


def test_public_pipeline_and_consistent_orientation_metric(tmp_path: Path) -> None:
    import json
    from modelforge.antipodal import recover_even_quadratic
    from modelforge.execution import run_json

    data = tmp_path / "public"
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    results = []
    for name, args in [
        ("hessian_pilot.py", ["prepare", str(data)]),
        ("refine.py", [str(data)]),
        ("multiscale.py", ["prepare", str(data)]),
    ]:
        results.append(
            run_json(
                [sys.executable, str(ROOT / "examples/isolated_glu" / name), *args],
                {},
                cwd=tmp_path / "monitor",
                env=env,
                memory_limit_mb=512,
                timeout_seconds=180,
            )
        )
    report = json.loads((data / "multiscale-report.json").read_text())
    assert report["full_glu_validation_relative_error"] < 1e-3
    assert report["sealed"] is False
    assert report["original_weights_recovered"] is False
    obs = np.load(data / "multiscale-observations.npz", allow_pickle=False)
    solution = np.load(data / "multiscale-solution.npz", allow_pickle=False)
    q = recover_even_quadratic(obs["even_pos"], obs["even_neg"], 32)
    g, c = solution["gates"], solution["couplings"]
    predicted = 0.25 * (np.einsum("wi,woj->oij", g, c) + np.einsum("wj,woi->oij", g, c))
    i, j = np.triu_indices(32)
    expected = np.linalg.norm((predicted - q)[:, i, j]) / np.linalg.norm(q[:, i, j])
    assert abs(report["orientation_rounded_relative_residual"] - expected) < 1e-12
