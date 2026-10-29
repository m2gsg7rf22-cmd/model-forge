import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_controller")


def setup(m, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = tmp_path / "experiments"
    parent.mkdir()
    output = parent / "joint-response-numeric-audit-v1"
    config = repo / "config.json"
    review = repo / "review.json"
    config.write_text("{}")
    review.write_text("{}")
    cfg = {
        "policy": m.POLICY,
        "input_bindings": {"root_files": {}, "case_files": {}},
        "inventory": {},
    }
    monkeypatch.setattr(m, "verify_source", lambda *a: (cfg, "a" * 40))
    monkeypatch.setattr(
        m, "prior_outputs", lambda *a: {"widths_hypothesized": [96, 96, 48, 48, 16, 16]}
    )
    monkeypatch.setattr(m, "layout", lambda *a: {"retained_bytes": 0})
    monkeypatch.setattr(m, "validate_transport", lambda *a: None)
    monkeypatch.setattr(m, "disk_free", lambda *a: 3 * 1024**3)

    def forbidden(*a, **k):
        raise AssertionError("observations touched before gates")

    monkeypatch.setattr(m, "load_join", forbidden)
    return repo, output, config, review, cfg


def test_failed_actual_TREE_gate_saves_preflight_before_marker_or_observations(
    m, tmp_path, monkeypatch
):
    repo, output, config, review, _ = setup(m, tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        m.run_protocol(
            repo,
            output,
            config,
            review,
            engine=lambda *a, **k: {
                "status": "passed",
                "process_monitor_scope": "process_only",
            },
        )
    assert (output / "preflight-receipt.json").exists()
    assert not (output.parent / m.MARKER).exists()
    assert (
        json.loads((output / "controller-report.json").read_text())["status"]
        == "failed"
    )


def test_disk_rejection_before_large_preflight_or_marker(m, tmp_path, monkeypatch):
    repo, output, config, review, _ = setup(m, tmp_path, monkeypatch)
    monkeypatch.setattr(m, "disk_free", lambda *a: 0)

    def forbidden(*a, **k):
        raise AssertionError("engine before free-disk gate")

    with pytest.raises(ValueError):
        m.run_protocol(repo, output, config, review, engine=forbidden)
    assert not (output.parent / m.MARKER).exists()


def test_default_launcher_always_controls_outer_TREE_and_exact_limits(
    m, tmp_path, monkeypatch
):
    repo, output, config, review, _ = setup(m, tmp_path, monkeypatch)
    calls = []

    def engine(command, payload, **kw):
        calls.append((payload, kw))
        output.mkdir()
        (output / "controller-report.json").write_text(
            json.dumps({"status": "completed"})
        )
        return {
            "status": "completed",
            "memory_limit_mb": 512,
            "process_monitor_scope": "process_and_children",
            "memory_samples": 2,
            "peak_sampled_rss_bytes": 1024,
        }

    result = m.launch(
        dict(
            repo=str(repo), output=str(output), config=str(config), review=str(review)
        ),
        engine=engine,
    )
    assert result["status"] == "completed"
    assert calls[0][0]["operation"] == "controlled_run"
    assert (
        calls[0][1]["memory_limit_mb"] == 512
        and calls[0][1]["timeout_seconds"] == 180
        and calls[0][1]["output_limit_bytes"] == 1024
    )
    assert all(calls[0][1]["env"][k] == "1" for k in m.THREAD_ENV)


def test_existing_output_denies_retry_even_if_no_postpreflight_marker(
    m, tmp_path, monkeypatch
):
    repo, output, config, review, _ = setup(m, tmp_path, monkeypatch)
    output.mkdir()

    def forbidden(*a, **k):
        raise AssertionError("repeat attempt")

    with pytest.raises(ValueError):
        m.launch(
            dict(
                repo=str(repo),
                output=str(output),
                config=str(config),
                review=str(review),
            ),
            engine=forbidden,
        )


def test_missing_approval_cli_does_not_create_output_or_marker(m, tmp_path):
    import os
    import subprocess
    import sys

    repo = tmp_path / "repo"
    repo.mkdir()
    review = repo / "review.json"
    review.write_text("{}")
    output = tmp_path / "experiments/joint-response-numeric-audit-v1"
    script = (
        Path(__file__).parents[1]
        / "examples/isolated_glu/joint_response_audit_controller.py"
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).parents[1] / "src")
        + os.pathsep
        + str(script.parent),
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(
            dict(
                repo=str(repo),
                output=str(output),
                config=str(repo / "missing"),
                review=str(review),
            )
        ),
        text=True,
        capture_output=True,
        timeout=10,
        env=env,
    )
    assert proc.returncode == 1 and json.loads(proc.stdout)["status"] == "failed"
    assert len(proc.stdout.encode()) + len(proc.stderr.encode()) <= 1024
    assert not output.exists() and not (output.parent / m.MARKER).exists()


def test_source_change_after_preflight_stops_before_marker(m, tmp_path, monkeypatch):
    repo, output, config, review, cfg = setup(m, tmp_path, monkeypatch)
    calls = []

    def verify(*args):
        calls.append(True)
        return cfg, ("a" * 40 if len(calls) == 1 else "b" * 40)

    monkeypatch.setattr(m, "verify_source", verify)
    resource = dict(
        status="passed",
        memory_limit_mb=512,
        process_monitor_scope="process_and_children",
        memory_samples=2,
        peak_sampled_rss_bytes=1024,
    )
    with pytest.raises(ValueError, match="freeze changed"):
        m.run_protocol(repo, output, config, review, engine=lambda *a, **kw: resource)
    assert not (output.parent / m.MARKER).exists()


def test_success_control_flow_consumes_marker_before_view_and_commits_six_summaries(
    m, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from test_joint_response_audit_protocol import audit

    repo, output, config, review, cfg = setup(m, tmp_path, monkeypatch)
    protocol = importlib.import_module("joint_response_audit_protocol")
    cfg["input_bindings"] = protocol.input_bindings(
        audit(protocol, tmp_path / "actual"), tmp_path / "actual"
    )
    jobs = tuple(
        SimpleNamespace(
            job_id=f"job{i:02d}", n=32, outputs=32, width_hypothesis=w, batch_count=34
        )
        for i, w in enumerate(m.POLICY["width_hypotheses"])
    )

    class View:
        def __init__(self):
            self.jobs = jobs

        def report(self):
            return {"status": "ready"}

        def finish(self):
            return {"status": "completed"}

    def view(*args):
        assert (output.parent / m.MARKER).exists()
        return View()

    monkeypatch.setattr(m, "load_join", view)

    def audited(jobs, readers, root, *, check, progress):
        root.mkdir()
        cases = []
        for j, r in zip(jobs, readers, strict=True):
            check()
            p = root / j.job_id
            p.mkdir()
            (p / "result.json").write_text("{}")
            cases.append(
                dict(job_id=j.job_id, status="agreed", checkpoints=[{"step": 0}])
            )
            r.done = True
        progress({"completed_case_visits": 6})
        return dict(
            status="completed",
            cases=cases,
            completed_case_visits=6,
            numeric_consistency_agreed=True,
        )

    monkeypatch.setattr(m, "audit_jobs", audited)
    resource = dict(
        status="passed",
        memory_limit_mb=512,
        process_monitor_scope="process_and_children",
        memory_samples=2,
        peak_sampled_rss_bytes=1024,
    )
    a = m.run_protocol(repo, output, config, review, engine=lambda *a, **kw: resource)
    assert a["status"] == "completed" and a["marker_consumed"] and len(a["cases"]) == 6
    assert all("checkpoints" not in c for c in a["cases"])
    assert (output / "terminal-binding.json").exists() and not a[
        "scientific_reconstruction_success"
    ]


def test_late_outer_layout_failure_never_leaves_completed_error_receipt(
    m, tmp_path, monkeypatch
):
    repo, output, config, review, _ = setup(m, tmp_path, monkeypatch)

    def engine(*args, **kwargs):
        output.mkdir()
        (output / "controller-report.json").write_text(
            json.dumps({"status": "completed"})
        )
        return dict(
            status="completed",
            memory_limit_mb=512,
            process_monitor_scope="process_and_children",
            memory_samples=2,
            peak_sampled_rss_bytes=1024,
        )

    def failed(*args):
        raise ValueError("literal final layout failure")

    monkeypatch.setattr(m, "layout", failed)
    with pytest.raises(ValueError):
        m.launch(
            dict(
                repo=str(repo),
                output=str(output),
                config=str(config),
                review=str(review),
            ),
            engine=engine,
        )
    receipt = json.loads((output / "outer-monitor.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"


def test_failure_summary_retains_reader_counter_uncertainty(m):
    partial = {
        "reads": 7,
        "bytes_read": 0,
        "read_bytes_are_lower_bounds_on_io_failure": True,
        "status": "failed",
    }
    a = m._summary({"job_id": "job02", "status": "failed", "reader_partial": partial})
    assert a["reader_partial"] == partial
