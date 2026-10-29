import importlib
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_controller")


def tree():
    return dict(
        status="passed",
        shards=2,
        manifest_sha256="1" * 64,
        process_monitor_scope="process_and_children",
        memory_samples=3,
        peak_sampled_rss_bytes=12345,
        memory_limit_mb=512,
    )


def paths(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    cfg = repo / "config.json"
    cfg.write_text("{}")
    review = repo / "review.json"
    review.write_text("{}")
    return repo, tmp_path / "experiments/attempt", cfg, review


def test_missing_full_approval_stops_before_output_or_engine(m, tmp_path):
    repo, out, cfg, review = paths(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("engine before source approval")

    with pytest.raises(ValueError):
        m.run_protocol(repo, out, cfg, review, engine=forbidden)
    assert not out.exists()


def test_failed_preflight_prevents_marker_sources_and_rng(m, tmp_path, monkeypatch):
    repo, out, cfg, review = paths(tmp_path)
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "a" * 40))
    monkeypatch.setattr(m, "prior_markers", lambda *a: None)

    def forbidden(*a, **k):
        raise AssertionError("source/RNG before preflight")

    monkeypatch.setattr(m, "load_join", forbidden)
    monkeypatch.setattr(m, "PCG64Stream", forbidden)
    with pytest.raises(ValueError):
        m.run_protocol(
            repo, out, cfg, review, engine=lambda *a, **k: dict(tree(), status="failed")
        )
    assert not (out.parent / m.MARKER).exists()


def test_all_degenerate_cases_abstain_without_rng_but_preserve_all_six(
    m, tmp_path, monkeypatch
):
    repo, out, cfg, review = paths(tmp_path)
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "a" * 40))
    monkeypatch.setattr(m, "prior_markers", lambda *a: None)
    monkeypatch.setattr(m, "validate_transport", lambda *a, **k: None)
    monkeypatch.setattr(m, "disk_free", lambda p: 3 * 1024**3)
    events = []
    from joint_response_join import JointJob

    class View:
        jobs = tuple(JointJob(f"job{i:02d}", 32, 32, 1, 34, iter(())) for i in range(6))

        def report(self):
            return {"fixture": True}

        def finish(self):
            events.append("finish")
            return {"status": "completed"}

    def joined(*args):
        assert (out.parent / m.MARKER).is_file()
        events.append("joined")
        return View()

    monkeypatch.setattr(m, "load_join", joined)

    def degenerate(job):
        events.append(job.job_id)
        raise ValueError("degenerate training RMS")

    monkeypatch.setattr(m, "assemble_case", degenerate)

    def forbidden(*args):
        raise AssertionError("RNG for degenerate scale")

    monkeypatch.setattr(m, "prepare_inputs", forbidden)
    result = m.run_protocol(repo, out, cfg, review, engine=lambda *a, **k: tree())
    assert result["status"] == "completed" and len(result["cases"]) == 6
    assert all(
        c["status"] == "abstained" and c["updates_completed"] == 0
        for c in result["cases"]
    )
    assert events == ["joined", *[f"job{i:02d}" for i in range(6)], "finish"]
    assert result["scientific_reconstruction_success"] is False


def test_consumed_marker_prevents_default_engine(m, tmp_path):
    repo, out, cfg, review = paths(tmp_path)
    out.parent.mkdir()
    (out.parent / m.MARKER).write_text("{}")

    def forbidden(*a, **k):
        raise AssertionError("engine after consumed marker")

    with pytest.raises(ValueError):
        m.launch(
            dict(repo=str(repo), output=str(out), config=str(cfg), review=str(review)),
            engine=forbidden,
        )


def test_default_launcher_enforces_shared_tree_time_blas_and_output_limits(m, tmp_path):
    repo, out, cfg, review = paths(tmp_path)

    def engine(command, payload, **kwargs):
        assert payload["operation"] == "controlled_run"
        assert kwargs["memory_limit_mb"] == 512 and kwargs["timeout_seconds"] == 180
        assert kwargs["output_limit_bytes"] == 1024
        assert all(kwargs["env"][k] == "1" for k in m.THREAD_ENV)
        assert command[-1].endswith("joint_response_controller.py")
        raise TimeoutError("literal shared timeout")

    with pytest.raises(TimeoutError):
        m.launch(
            dict(repo=str(repo), output=str(out), config=str(cfg), review=str(review)),
            engine=engine,
        )


def test_saved_initial_numeric_failure_is_bound_without_rng_or_optimization_replay(
    m, tmp_path, monkeypatch
):
    import hashlib
    import json
    import numpy as np
    import joint_response_driver as driver
    from joint_response_case import CaseArrays
    from joint_response_join import JointJob
    from joint_response_kernel import Parameters, NumericalFailure
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

    def no_fit(initial, *args):
        raise NumericalFailure(0, initial, initial, initial)

    monkeypatch.setattr(driver, "fit", no_fit)
    root = tmp_path / "trace"
    result = driver.run_case(job, case, opt, TraceStore(root, 1))
    h = hashlib.sha256(memoryview(ones).cast("B")).hexdigest()
    pins = m.validate_case(
        root,
        result,
        1,
        m.optimizer_binding(opt),
        dict(input_array_sha256=h, response_array_sha256=h),
    )
    assert "failure-state.npz" in pins and "initial.npz" in pins
    with (root / "schedule.npz").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError):
        m.validate_case(
            root,
            result,
            1,
            m.optimizer_binding(opt),
            dict(input_array_sha256=h, response_array_sha256=h),
        )


def test_second_case_can_run_when_first_pair_member_abstains(m, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    repo, out, cfg, review = paths(tmp_path)
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "a" * 40))
    monkeypatch.setattr(m, "prior_markers", lambda *a: None)
    monkeypatch.setattr(m, "validate_transport", lambda *a, **k: None)
    monkeypatch.setattr(m, "disk_free", lambda p: 3 * 1024**3)
    from joint_response_join import JointJob

    class View:
        jobs = tuple(JointJob(f"job{i:02d}", 32, 32, 1, 34, iter(())) for i in range(6))

        def report(self):
            return {"fixture": True}

        def finish(self):
            return {"status": "completed"}

    monkeypatch.setattr(m, "load_join", lambda *a: View())

    def assemble(job):
        if job.job_id != "job01":
            raise ValueError("degenerate training RMS")
        return SimpleNamespace(inputs=np.ones((1, 1)), responses=np.ones((1, 1)))

    monkeypatch.setattr(m, "assemble_case", assemble)
    prepared = []

    def prepare(job, *a):
        prepared.append(job.job_id)
        return object()

    monkeypatch.setattr(m, "prepare_inputs", prepare)
    monkeypatch.setattr(m, "optimizer_binding", lambda *a: {"fixture": "no RNG"})
    monkeypatch.setattr(m, "validate_case", lambda *a, **k: {})
    monkeypatch.setattr(
        m,
        "run_case",
        lambda job, *a: dict(
            job_id=job.job_id,
            status="numerical_failure",
            updates_completed=0,
            selected_step=None,
        ),
    )
    result = m.run_protocol(repo, out, cfg, review, engine=lambda *a, **k: tree())
    assert prepared == ["job01"] and result["case_driver_invocations"] == 1
    assert result["paired_binding_checks"]["0"] == "unavailable_first_case_abstained"
    assert len(result["cases"]) == 6


def test_real_default_cli_rejects_unapproved_source_before_data_or_marker(m, tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    from joint_response_protocol import DEPENDENCIES, environment

    original = Path(__file__).parents[1]
    repo, out, cfg, review = paths(tmp_path)
    for relative in DEPENDENCIES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, target)
    request = dict(repo=str(repo), output=str(out), config=str(cfg), review=str(review))
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "examples/isolated_glu/joint_response_controller.py"),
        ],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=15,
        env=environment(repo),
    )
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "failed"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 2048
    assert not out.exists() and not (out.parent / m.MARKER).exists()
