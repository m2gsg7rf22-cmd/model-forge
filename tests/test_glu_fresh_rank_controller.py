import importlib.util
import json
from pathlib import Path
import sys
import pytest


def load(monkeypatch):
    b = Path(__file__).parents[1] / "examples/isolated_glu"
    monkeypatch.syspath_prepend(str(b))
    p = b / "fresh_rank_controller.py"
    assert p.exists(), "Full controller absent"
    spec = importlib.util.spec_from_file_location("fresh_rank_controller", p)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fresh_rank_controller", m)
    spec.loader.exec_module(m)
    return m


def test_no_full_review_blocks_before_any_worker(tmp_path, monkeypatch):
    m = load(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    config, review = repo / "config.json", repo / "review.json"
    config.write_text("{}")
    review.write_text("{}")

    def forbidden(*a, **kw):
        pytest.fail("Worker started without full source approval")

    with pytest.raises(m.ProtocolError):
        m.run_protocol(repo, tmp_path / "run", config, review, engine=forbidden)
    report = json.loads((tmp_path / "run/controller-report.json").read_text())
    assert report["status"] == "failed" and not report["collection_started"]


@pytest.mark.parametrize(
    "failure",
    [
        "scope",
        "transport",
        "disk",
        "existing_marker",
        "partial_marker",
        "collection",
        "killed",
    ],
)
def test_predatum_and_terminal_failures_never_retry(tmp_path, monkeypatch, failure):
    m = load(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    ex = tmp_path / "experiments"
    ex.mkdir()
    config, review = repo / "config.json", repo / "review.json"
    config.write_text("{}")
    review.write_text("{}")
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "a" * 40))
    monkeypatch.setattr(m, "validate_transport", lambda *a: None)
    disk = type("Disk", (), {"free": 0 if failure == "disk" else 3 * 1024**3})()
    monkeypatch.setattr(m.shutil, "disk_usage", lambda *a: disk)
    marker = ex / m.MARKER
    if failure in ("existing_marker", "partial_marker"):
        marker.write_text("{}" if failure == "existing_marker" else "{")
    calls = []

    def engine(command, payload, **kw):
        calls.append(payload)
        assert kw["memory_limit_mb"] == 512 and kw["output_limit_bytes"] == 2048
        assert 0 < kw["timeout_seconds"] <= 180
        assert kw["env"]["OPENBLAS_NUM_THREADS"] == "1"
        if len(calls) == 3:
            assert marker.exists()
            if failure == "killed":
                raise TimeoutError("injected killed worker")
            return dict(
                status="failed",
                error="injected",
                process_monitor_scope="process_and_children",
                memory_samples=1,
            )
        return dict(
            status="failed" if failure == "transport" and len(calls) == 2 else "passed",
            process_monitor_scope="process_only"
            if failure == "scope"
            else "process_and_children",
            memory_samples=1,
        )

    with pytest.raises(m.ProtocolError):
        m.run_protocol(repo, ex / "run", config, review, engine=engine)
    expected = (
        1 if failure == "scope" else (3 if failure in ("collection", "killed") else 2)
    )
    assert len(calls) == expected
    report = json.loads((ex / "run/controller-report.json").read_text())
    assert report["status"] == "failed" and not report["evaluation_started"]
    assert report["collection_started"] == (expected == 3)
    if failure == "collection":
        assert report["collection_receipt"]["status"] == "failed"
    if failure == "killed":
        assert report["unreceived_worker_counters_unknown"] is True
    if failure in ("scope", "transport", "disk"):
        assert not marker.exists()


def test_literal_transport_roundtrips_all_seven_families(tmp_path, monkeypatch):
    m = load(monkeypatch)
    result = m.synthetic_transport(tmp_path / "transport")
    assert result["status"] == "passed" and result["shards"] == 7
    assert result["bytes"] <= 17621380
    assert result["new_queries"] == 0


def test_signed_paths_cannot_escape_repository(tmp_path, monkeypatch):
    m = load(monkeypatch)
    for name in ("../outside", "/absolute", "examples/../../outside"):
        with pytest.raises(ValueError):
            m.signed_path(tmp_path, name)


def test_complete_controller_orders_seal_before_truth_and_limits_estimator_payload(
    tmp_path, monkeypatch
):
    m = load(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    ex = tmp_path / "experiments"
    ex.mkdir()
    config, review = repo / "config.json", repo / "review.json"
    config.write_text("{}")
    review.write_text("{}")
    monkeypatch.setattr(m, "verify_source", lambda *a: ({}, "a" * 40))
    monkeypatch.setattr(m, "validate_transport", lambda *a: None)
    monkeypatch.setattr(
        m.shutil, "disk_usage", lambda *a: type("D", (), {"free": 3 * 1024**3})()
    )
    out = ex / "run"
    stages = []
    sealed = []

    def engine(command, payload, **kw):
        index = len(stages)
        stages.append(payload)
        if index == 2:
            assert (ex / m.MARKER).exists()
            c = out / "collection"
            c.mkdir()
            for name in ("truth-commitment.json", "seed-commitment.json"):
                (c / name).write_text("{}")
            report = dict(
                status="completed",
                source_commit="a" * 40,
                certificate=False,
                private_gate="closed",
                calls_reserved=545280,
                calls_attempted=545280,
                calls_returned=545280,
                calls_persisted=545280,
                callback_batches=768,
                truth_inventory=dict(shards=1608),
                public_inventory=dict(shards=144),
                public_cases_sha256="0" * 64,
                public_manifest_sha256="1" * 64,
                truth_commitment_sha256=m.sha(c / "truth-commitment.json"),
                seed_commitment_sha256=m.sha(c / "seed-commitment.json"),
            )
            name = "collection"
        elif index == 3:
            assert set(payload) == {
                "public_root",
                "output",
                "cases_sha256",
                "manifest_sha256",
                "source_commit",
                "store_limits",
            }
            name = "estimator"
            report = dict(
                status="completed", source_commit="a" * 40, seal_sha256="2" * 64
            )
            (out / name).mkdir()
        elif index == 4:
            assert sealed and payload["seed_commitment_sha256"] == m.sha(
                out / "collection/seed-commitment.json"
            )
            name = "evaluation"
            report = dict(
                status="completed",
                source_commit="a" * 40,
                rule_passes=dict(gap=False, stable=False),
            )
            (out / name).mkdir()
        else:
            return dict(
                status="passed",
                process_monitor_scope="process_and_children",
                memory_samples=1,
            )
        m.save_json(out / name / (name + "-report.json"), report)
        return dict(
            status="completed",
            report_sha256=m.sha(out / name / (name + "-report.json")),
            process_monitor_scope="process_and_children",
            memory_samples=1,
        )

    monkeypatch.setattr(m, "public_boundary", lambda *a: sealed.append(True))
    monkeypatch.setattr(m, "validate_final", lambda *a: None)
    monkeypatch.setattr(m, "exact_layout", lambda *a: None)
    result = m.run_protocol(repo, out, config, review, engine=engine)
    assert len(stages) == 5 and result["status"] == "completed"
    assert result["rule_passes"] == dict(gap=False, stable=False)
    assert result["certificate"] is False


def test_collection_count_boolean_is_not_a_measured_count(monkeypatch):
    m = load(monkeypatch)
    with pytest.raises(ValueError):
        m.check_counts(
            dict(
                status="completed",
                source_commit="a" * 40,
                calls_reserved=True,
                calls_attempted=545280,
                calls_returned=545280,
                calls_persisted=545280,
                callback_batches=768,
                certificate=False,
                private_gate="closed",
            ),
            "a" * 40,
        )


@pytest.mark.parametrize("extra", ["uncommitted.npz", "extra.json", "extra-dir/"])
def test_final_layout_rejects_uncommitted_artifacts(tmp_path, monkeypatch, extra):
    m = load(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    for name in m.METADATA_PATHS:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    for name in m.STORE_PATHS.values():
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        (p / "manifest.jsonl").write_text(
            json.dumps(dict(name="literal.npz", bytes=1, sha256="0" * 64)) + "\n"
        )
        (p / "literal.npz").write_bytes(b"0")
    p = root / extra
    if extra.endswith("/"):
        p.mkdir()
    else:
        p.write_text("extra")
    with pytest.raises(ValueError, match="layout"):
        m.exact_layout(root)


@pytest.mark.parametrize("failure", ["none", "scope", "kill"])
def test_outer_launch_monitors_controller_and_all_children(
    tmp_path, monkeypatch, failure
):
    m = load(monkeypatch)
    output = tmp_path / "run"

    def engine(command, payload, **kw):
        assert payload["operation"] == "controlled_run"
        assert kw["timeout_seconds"] == 180 and kw["memory_limit_mb"] == 512
        assert kw["output_limit_bytes"] == 2048
        assert kw["env"]["OPENBLAS_NUM_THREADS"] == "1"
        output.mkdir()
        m.save_json(
            output / "controller-report.json",
            dict(status="completed", private_gate="closed", certificate=False),
        )
        if failure == "kill":
            raise TimeoutError("outer kill")
        return dict(
            status="completed",
            process_monitor_scope="process_only"
            if failure == "scope"
            else "process_and_children",
            memory_samples=1,
            peak_sampled_rss_bytes=100,
            memory_limit_mb=512,
        )

    request = dict(
        repo=str(tmp_path / "repo"),
        output=str(output),
        config="config",
        review="review",
    )
    if failure == "none":
        receipt = m.launch(request, engine=engine)
        assert receipt["status"] == "completed" and receipt[
            "controller_sha256"
        ] == m.sha(output / "controller-report.json")
    else:
        with pytest.raises(m.ProtocolError):
            m.launch(request, engine=engine)
    receipt = json.loads((output / "outer-monitor.json").read_text())
    assert receipt["status"] == ("completed" if failure == "none" else "failed")
    assert receipt["private_gate"] == "closed" and receipt["certificate"] is False


@pytest.mark.parametrize(
    "failure", ["none", "dirty", "source", "runtime", "review", "missing_test"]
)
def test_real_source_gate_binds_clean_git_runtime_docs_and_tests(
    tmp_path, monkeypatch, failure
):
    import subprocess
    import platform
    import numpy as np

    m = load(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in m.REQUIRED_REVIEW:
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("literal signed fixture")
    config = repo / "prereg.json"
    config_data = dict(
        policy=m.POLICY,
        inventory_plan=m.inventory_plan(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        frozen_dependencies={n: m.sha(repo / n) for n in m.DEPENDENCIES},
    )
    if failure == "runtime":
        config_data["runtime"]["numpy"] = "wrong"
    m.save_json(config, config_data)
    signed = {n: m.sha(repo / n) for n in m.REQUIRED_REVIEW | {"prereg.json"}}
    if failure == "missing_test":
        del signed["tests/test_glu_fresh_rank_stages.py"]
    review = tmp_path / "review.json"
    m.save_json(
        review,
        dict(full_fresh_protocol_approved=failure != "review", source_sha256=signed),
    )

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()

    git("init", "-b", "discovery-lab")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    if failure == "dirty":
        (repo / "new.txt").write_text("dirty")
    if failure == "source":
        (repo / "src/modelforge/__init__.py").write_text("changed")
    if failure == "none":
        _, head = m.verify_source(repo, config, review)
        assert head == git("rev-parse", "HEAD")
    else:
        with pytest.raises(ValueError):
            m.verify_source(repo, config, review)


def test_outer_launch_never_overwrites_an_existing_attempt(tmp_path, monkeypatch):
    m = load(monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    saved = output / "outer-monitor.json"
    saved.write_text("original receipt")

    def forbidden(*a, **kw):
        pytest.fail("Existing attempt launched again")

    with pytest.raises(m.ProtocolError):
        m.launch(
            dict(output=str(output), repo=str(tmp_path), config="c", review="r"),
            engine=forbidden,
        )
    assert saved.read_text() == "original receipt"


def test_transport_receipt_rejects_manifest_mutation_before_data(tmp_path, monkeypatch):
    m = load(monkeypatch)
    output = tmp_path / "transport"
    receipt = m.synthetic_transport(output)
    m.validate_transport(output, receipt)
    manifest = output / "arrays/manifest.jsonl"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        m.validate_transport(output, receipt)
