"""Scientific invariants, oracle separation and reproducibility, not only execution."""

import json

import numpy as np
import pytest

from modelforge.umrre import recover_affine, run_research


def test_hidden_affine_recovery_independent_target():
    rng = np.random.default_rng(993)
    w, b = rng.normal(size=(5, 4)), rng.normal(size=4)
    seen = []

    def oracle(x):
        seen.extend(x.tolist())
        return x @ w + b

    recovered, bias, cost = recover_affine(oracle, 5)
    assert cost == len(seen) == 6
    np.testing.assert_allclose(recovered, w, rtol=0, atol=1e-14)
    np.testing.assert_allclose(bias, b, rtol=0, atol=1e-14)


def test_experiments_scientific_invariants_and_saved_result(tmp_path):
    report = run_research(tmp_path / "experiment")
    assert (
        json.loads((tmp_path / "experiment/umrre-results.json").read_text()) == report
    )
    experiments = report["experiments"]
    assert experiments["affine_recovery"]["max_parameter_error"] < 1e-13
    ambiguous = experiments["rank_deficient_nonidentifiability"]
    assert ambiguous["observed_max_output_difference"] < 1e-12
    assert ambiguous["parameter_l2_distance"] > 1
    assert ambiguous["witness_max_output_difference"] > 1
    symmetry = experiments["neural_symmetry"]
    assert symmetry["weight_l2_distance"] > 1
    assert symmetry["heldout_max_output_difference"] < 1e-11
    approximate = experiments["functional_approximation"]
    assert approximate["heldout_mse"] < approximate["zero_predictor_heldout_mse"]
    assert not approximate["selection_uses_test_set"]
    assert approximate["heldout_mse"] > 0
    assert len(approximate["candidates"]) == 3
    assert all(
        row["output_mse"] > 0 and not row["lossless"]
        for row in experiments["compression"]["methods"]
    )
    kv = experiments["compression"]["kv_cache"]
    assert 0 < kv["int8_representation_bytes"] < kv["dense_bytes"]
    assert kv["attention_output_mse"] > 0
    assert len(experiments["multimodal_adapters"]["experiments"]) == 3
    assert report["cost"]["remote_requests"] == 0
    assert report["cost"]["sampled_peak_process_rss_bytes"] > 0


def test_reproducible_numerics_and_new_seeds():
    a, b, c = run_research(seed=7), run_research(seed=7), run_research(seed=8)
    assert a["numerical_sha256"] == b["numerical_sha256"]
    assert a["numerical_sha256"] != c["numerical_sha256"]


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "7"])
def test_seed_bounds(seed):
    with pytest.raises(ValueError):
        run_research(seed=seed)


def test_no_overwrite_or_symlink_write(tmp_path):
    occupied = tmp_path / "existing"
    occupied.mkdir()
    with pytest.raises(ValueError):
        run_research(occupied)
    link = tmp_path / "link"
    try:
        link.symlink_to(occupied, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    with pytest.raises(ValueError):
        run_research(link / "child")


def test_malformed_oracle_and_query_bound():
    with pytest.raises(ValueError):
        recover_affine(lambda x: x, 65)
    with pytest.raises(ValueError):
        recover_affine(lambda x: np.full((3, 1), np.nan), 2)
    with pytest.raises(ValueError):
        recover_affine(lambda x: np.zeros((1, 2)), 2)
