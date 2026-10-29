import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_driver")


def literals():
    from joint_response_case import CaseArrays
    from joint_response_join import JointJob
    from joint_response_kernel import Parameters
    from joint_response_streams import OptimizerInputs
    from joint_response_trace import TraceStore

    indices = np.arange(71808, dtype=np.int64)
    guard = np.isin((indices // 2112) % 17, (3, 7, 11, 15))
    ones = np.ones((71808, 32))
    case = CaseArrays(
        ones,
        ones,
        indices[~guard],
        indices[guard],
        ones[~guard],
        ones[~guard],
        ones[guard],
        ones[guard],
        1.0,
        1.0,
    )
    p = Parameters(np.zeros((1, 32)), np.zeros((1, 32)), np.zeros((32, 1)))
    state = json.dumps(
        dict(
            bit_generator="PCG64",
            state={"state": 0, "inc": 1},
            has_uint32=0,
            uinteger=0,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    opt = OptimizerInputs(
        p, np.zeros((4000, 256), dtype=np.int64), state, state, state, state
    )
    job = JointJob("job00", 32, 32, 1, 34, iter(()))
    return job, case, opt, TraceStore


def test_complete_literal_driver_precommits_before_first_optimizer_update(
    m, tmp_path, monkeypatch
):
    job, case, opt, Store = literals()
    store = Store(tmp_path / "case", 1)
    actual = m.fit
    calls = []

    def inspected_fit(*args):
        assert {"rng.json", "case.json"} <= store.metadata_names
        assert {"initial", "schedule", "split-indices"} <= store.names
        assert store.completed_updates == 0
        for name in (
            "rng.json",
            "case.json",
            "initial.npz",
            "schedule.npz",
            "split-indices.npz",
        ):
            assert (store.root / name).is_file()
        calls.append(1)
        return actual(*args)

    monkeypatch.setattr(m, "fit", inspected_fit)
    result = m.run_case(job, case, opt, store)
    assert calls == [1] and result["status"] == "completed"
    assert result["updates_completed"] == 4000 and result["selected_step"] == 0
    assert result["selected_guard_mse"] == pytest.approx(1.0)
    metrics = json.loads((store.root / "metrics.json").read_text())
    assert len(metrics["checkpoints"]) == 17
    assert (
        metrics["selected_step"] == 0
        and metrics["guard_is_consumed_calibration"] is True
    )
    assert store.finished and not store.failed
    for i in range(9):
        with np.load(
            store.root / f"prediction-{i:02d}.npz", allow_pickle=False
        ) as values:
            assert not np.any(values["predictions"])


def test_numeric_failure_preserves_state_and_never_selected_prefix(
    m, tmp_path, monkeypatch
):
    from joint_response_kernel import Checkpoint, Step, NumericalFailure

    job, case, opt, Store = literals()
    store = Store(tmp_path / "failure", 1)

    def fail(initial, tx, ty, gx, gy, schedule, observer):
        observer(Checkpoint(0, initial, initial, initial, 0.5, 1.0, 1.0))
        observer(Step(1, 0.5, initial))
        raise NumericalFailure(1, initial, initial, initial)

    monkeypatch.setattr(m, "fit", fail)
    result = m.run_case(job, case, opt, store)
    assert result["status"] == "numerical_failure" and result["selected_step"] is None
    assert result["updates_completed"] == 1 and store.finished
    assert "failure-state" in store.names and "selected-physical" not in store.names
    assert not any(name.startswith("prediction-") for name in store.names)


def test_io_failure_before_fit_is_global_and_has_no_success_receipt(
    m, tmp_path, monkeypatch
):
    job, case, opt, Store = literals()
    store = Store(tmp_path / "io", 1)

    def forbidden(*args):
        raise AssertionError("optimizer started after IO failure")

    def io(*args):
        raise OSError("literal disk failure")

    monkeypatch.setattr(m, "fit", forbidden)
    monkeypatch.setattr(store, "_exclusive", io)
    with pytest.raises(OSError):
        m.run_case(job, case, opt, store)
    assert store.failed and not store.finished


def test_rng_metadata_must_be_canonical_exact_state_before_fit(
    m, tmp_path, monkeypatch
):
    from dataclasses import replace

    job, case, opt, Store = literals()
    store = Store(tmp_path / "bad", 1)

    def forbidden(*args):
        raise AssertionError("fit before metadata validation")

    monkeypatch.setattr(m, "fit", forbidden)
    with pytest.raises(ValueError):
        m.run_case(job, case, replace(opt, initial_state_before='{"teacher":1}'), store)
    assert not store.names and not store.metadata_names


def test_metric_journal_failure_propagates_without_partial_success(
    m, tmp_path, monkeypatch
):
    from joint_response_kernel import Checkpoint

    job, case, opt, Store = literals()
    store = Store(tmp_path / "journal", 1)

    def stop(*args):
        raise OSError("literal journal failure")

    monkeypatch.setattr(m._Journal, "append", stop)

    def checkpoint_only(initial, tx, ty, gx, gy, schedule, observer):
        observer(Checkpoint(0, initial, initial, initial, 0.5, 1.0, 1.0))
        raise AssertionError("continued after failed metric persistence")

    monkeypatch.setattr(m, "fit", checkpoint_only)
    with pytest.raises(OSError):
        m.run_case(job, case, opt, store)
    assert store.failed and not store.finished and store.completed_updates == 0
    assert (
        "checkpoint-0000" in store.names and not (store.root / "terminal.json").exists()
    )


def test_physical_export_failure_is_numeric_failure_even_after_completed_fit(
    m, tmp_path, monkeypatch
):
    from joint_response_kernel import Checkpoint, Step, FitResult

    job, case, opt, Store = literals()
    store = Store(tmp_path / "export", 1)

    def literal_kernel(initial, tx, ty, gx, gy, schedule, observer):
        records = []
        for step in range(4001):
            if step:
                observer(Step(step, 0.5, initial))
            if step in m.CHECKPOINTS:
                c = Checkpoint(step, initial, initial, initial, 0.5, 1.0, 1.0)
                records.append(c)
                observer(c)
        return FitResult(initial, initial, 0, 4000, tuple(records))

    def fail(*args):
        raise FloatingPointError("literal export failure")

    monkeypatch.setattr(m, "fit", literal_kernel)
    monkeypatch.setattr(m, "physical_parameters", fail)
    result = m.run_case(job, case, opt, store)
    assert (
        result["status"] == "numerical_failure" and result["updates_completed"] == 4000
    )
    assert result["selected_step"] is None and "failure-state" in store.names
    assert result["failure_phase"] == "physical_export_or_candidate_prediction"
