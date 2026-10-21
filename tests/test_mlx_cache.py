"""Actual per-call MLX KV cache compared against preserved full-prefix decoding."""

from __future__ import annotations
import importlib.util
import platform
from pathlib import Path
import pytest
from modelforge.runtime_hf import create_owned_fixture
from modelforge.runtime_mlx import benchmark_cache

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or importlib.util.find_spec("mlx_lm") is None,
    reason="Optional Apple MLX backend unavailable",
)


def test_fresh_cache_matches_reference_and_is_released(tmp_path: Path) -> None:
    import mlx.core as mx

    path = tmp_path / "owned"
    create_owned_fixture(path, model_type="llama")
    prior = mx.get_memory_limit()
    result = benchmark_cache(path, max_new_tokens=4, repeats=2)
    assert result["configuration"]["prompt"] == "hello world"
    assert result["configuration"]["max_new_tokens"] == 4
    assert result["tested_token_parity"]
    assert len(result["measurements"]) == 4
    assert [row["cached"] for row in result["measurements"]] == [
        False,
        True,
        True,
        False,
    ]
    for row in result["measurements"]:
        assert row["new_tokens"] == 4
        assert (
            row["cache_logical_bytes"] > 0
            if row["cached"]
            else row["cache_logical_bytes"] == 0
        )
    assert mx.get_memory_limit() == prior
    again = benchmark_cache(path, max_new_tokens=4, repeats=1)
    assert (
        again["measurements"][0]["token_ids"] == result["measurements"][0]["token_ids"]
    )


def test_cache_rejects_unbounded_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        benchmark_cache("/unused", repeats=100)


def test_float32_control_recomputes_logits_on_reference_prefix(tmp_path: Path) -> None:
    path = tmp_path / "owned-control"
    create_owned_fixture(path, model_type="llama")
    report = benchmark_cache(
        path, max_new_tokens=4, repeats=1, precision="float32", compare_logits=True
    )
    assert report["precision"] == "float32"
    diagnostic = report["reference_prefix_logit_comparison"]
    assert diagnostic["steps"] == 4
    assert diagnostic["argmax_disagreement_steps"] == []
    assert diagnostic["max_absolute_logit_error"] < 1e-5
