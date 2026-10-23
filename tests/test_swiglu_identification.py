import pytest

torch = pytest.importorskip("torch")

from modelforge import swiglu_identification as si  # noqa: E402


def _teacher(width: int = 6, dim: int = 4, rows: int = 400, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    gate = torch.randn((width, dim), generator=g, dtype=torch.float64)
    up = torch.randn((width, dim), generator=g, dtype=torch.float64)
    down = torch.randn((3, width), generator=g, dtype=torch.float64)
    z = torch.randn((rows, dim), generator=g, dtype=torch.float64)
    return z, si._features(z, gate, up) @ down.T, gate


def test_consensus_and_dedup() -> None:
    a = torch.tensor([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0], [5.0, 5.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.01]], dtype=torch.float64)
    c = si.consensus_gates(a, b)
    assert len(c) == 2


def test_frozen_true_gates_give_exact_fit() -> None:
    z, y, gate = _teacher()
    fit = si.fit_swiglu(z, y, 6, seed=1, seconds=5, frozen_gate=gate)
    assert fit.relative_loss < 1e-8
    assert si.match_gates(fit.gate, gate)["within_1pct"] == 6


def test_pipeline_on_small_teacher() -> None:
    z, y, gate = _teacher()
    out = si.identify(z, y, 6, over_width=18, seconds=4)
    report = si.match_gates(out["result"].gate, gate)
    # mechanics only; identification quality is measured by the longer pilots
    assert 0 <= out["consensus_count"] <= 6 and out["result"].gate.shape == (6, 4)
    assert out["result"].relative_loss < 1.0 and report["true_units"] == 6
    with pytest.raises(ValueError):
        si.fit_swiglu(z, y, 2, seed=0, seconds=1, frozen_gate=gate)


def test_round_budget_is_load_independent() -> None:
    z, y, gate = _teacher()
    fit = si.fit_swiglu(z, y, 6, seed=1, seconds=1e9, max_rounds=3)
    assert fit.rounds == 3
