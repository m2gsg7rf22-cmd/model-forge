import importlib
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_protocol")


def test_policy_is_fixed_all_six_with_joint_optimizer_and_closed_gate(m):
    p = m.POLICY
    assert p["jobs"] == 6 and p["adam_steps"] == 4000 and p["batch_size"] == 256
    assert p["checkpoint_steps"] == [0, *range(256, 4000, 256), 4000]
    assert p["timeout_seconds"] == 180 and p["memory_limit_mb"] == 512
    assert (
        p["output_limit_bytes"] == 2048 and p["per_engine_output_limit_bytes"] == 1024
    )
    assert p["private_gate"] == "closed" and p["certificate"] is False
    assert m.MARKER == "joint-response-development-v1-attempt.json"


def test_full_source_approval_rejected_before_config_and_raw_access(
    m, tmp_path, monkeypatch
):
    approval = tmp_path / "review.json"
    approval.write_text("{}")

    def forbidden(*args):
        raise AssertionError("config/source accessed before approval")

    monkeypatch.setattr(m, "config_template", forbidden)
    with pytest.raises(ValueError):
        m.verify_source(tmp_path, tmp_path / "missing-config.json", approval)


def test_pair_bindings_hash_all_initial_arrays_schedule_and_states_without_rng(m):
    from joint_response_kernel import Parameters
    from joint_response_streams import OptimizerInputs
    from dataclasses import replace
    import numpy as np

    p = Parameters(np.zeros((1, 32)), np.zeros((1, 32)), np.zeros((32, 1)))
    opt = OptimizerInputs(p, np.zeros((4000, 256), dtype=np.int64), "a", "b", "c", "d")
    first = m.optimizer_binding(opt)
    assert first == m.optimizer_binding(opt)
    changed = opt.schedule.copy()
    changed[0, 0] = 1
    assert first != m.optimizer_binding(replace(opt, schedule=changed))
    assert first != m.optimizer_binding(replace(opt, batch_state_after="e"))
    assert set(first) == {"g", "u", "d", "schedule", "states"}


def test_typed_pcg_wrapper_routes_exact_draws_to_injected_fake_backend(m, monkeypatch):
    import numpy as np

    calls = []

    class Generator:
        def __init__(self, bitgen):
            self.bit_generator = bitgen

        def standard_normal(self, shape, dtype):
            calls.append(("normal", shape, dtype))
            return np.ones(shape)

        def integers(self, low, high, size, dtype):
            calls.append(("indices", low, high, size, dtype))
            return np.zeros(size, dtype=dtype)

    class Bit:
        def __init__(self, seed):
            self.state = {"literal_seed": seed}

    monkeypatch.setattr(m.np.random, "PCG64", Bit)
    monkeypatch.setattr(m.np.random, "Generator", Generator)
    stream = m.PCG64Stream(7)
    assert stream.state() == {"literal_seed": 7}
    assert stream.normal((1, 2)).shape == (1, 2)
    assert stream.indices(4, (2, 2)).dtype == np.int64
    assert calls[0] == ("normal", (1, 2), np.float64)
    assert calls[1] == ("indices", 0, 4, (2, 2), np.int64)


def test_small_source_component_evidence_paths_are_exact_and_closed(m):
    assert len(m.COMPONENT_PINS) == 7
    assert all(
        path.startswith("experiments/evidence/joint-response-")
        for path in m.COMPONENT_PINS
    )
    assert all(len(h) == 64 for h in m.COMPONENT_PINS.values())
    assert "examples/isolated_glu/joint_response_driver.py" in m.DEPENDENCIES
    assert "examples/isolated_glu/joint_response_controller.py" in m.DEPENDENCIES
    assert "src/modelforge/execution.py" in m.DEPENDENCIES


def test_positive_source_contract_checks_every_pin_and_rejects_changed_source(
    m, tmp_path, monkeypatch
):
    import json
    import shutil

    original = Path(__file__).parents[1]
    repo = tmp_path / "synthetic-source-repository"
    repo.mkdir()
    for relative in m.REQUIRED_REVIEW:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, target)
    cfg = repo / "experiments/preregistrations/literal-source-test.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(m.config_template(repo)))
    signed = {
        relative: m.digest(repo / relative, 2 * 1024**2)
        for relative in m.REQUIRED_REVIEW
    }
    signed[str(cfg.relative_to(repo))] = m.digest(cfg)
    approval = repo / "experiments/evidence/literal-full-review.json"
    approval.write_text(
        json.dumps(
            dict(
                full_joint_response_development_protocol_approved=True,
                registered_execution_after_gates_approved=True,
                private_gate="closed",
                certificate=False,
                source_sha256=signed,
            )
        )
    )

    def git(command, **kwargs):
        assert kwargs["cwd"] == repo
        if command[1] == "status":
            return ""
        if command[1] == "branch":
            return "discovery-lab\n"
        return "a" * 40 + "\n"

    monkeypatch.setattr(m.subprocess, "check_output", git)
    for key, value in m.THREAD_ENV.items():
        monkeypatch.setenv(key, value)
    assert m.verify_source(repo, cfg, approval)[1] == "a" * 40
    source = repo / "examples/isolated_glu/joint_response_controller.py"
    source.write_text(source.read_text() + "\n# literal changed source\n")
    with pytest.raises(ValueError):
        m.verify_source(repo, cfg, approval)
