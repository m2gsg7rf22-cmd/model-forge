import importlib
from pathlib import Path
import pytest


@pytest.fixture
def inv(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("variable_gate_inventory")


def test_complete_worst_inventory_includes_progress_and_overshoot(inv):
    names = inv.trace_names("job00")
    assert len(names) == len(set(names)) and len(names) > 900
    assert "job00-r3-refine-chunk063" in names
    assert "job00-r3-eigen-chunk031" in names
    assert "job00-initial-refine-intent031" in names
    assert inv.upper_fields("job00-final")["cosines"] == (607, 607)
    assert inv.upper_fields("job00-r0-geometry")["residual"] == (192, 528)
    p = inv.inventory()
    assert p["trace_shards"] == 6 * len(names)
    assert p["bytes_upper_bound"] < 1024**3
    assert p["manifest_bytes_upper_bound"] < 1024**2
    assert p["minimum_free_disk_bytes"] == 2 * 1024**3
    assert p["parent_retained_bytes_upper_bound"] == 534980568


def test_exact_name_specific_manifest_bound(inv):
    p = inv.inventory()
    assert p["trace_shards"] < 8192
    assert p["largest_shard_upper_bound"] < 8 * 1024**2
    assert p["preflight_shards"] == len(inv.transport_schemas())
    assert p["bytes_upper_bound"] == sum(
        p[k]
        for k in [
            "trace_bytes_upper_bound",
            "preflight_bytes_upper_bound",
            "metadata_reserve_bytes",
            "parent_retained_bytes_upper_bound",
        ]
    )


@pytest.mark.parametrize(
    "name",
    [
        "job06-final",
        "job00-r4-input",
        "job00-r0-refine-chunk064",
        "job00-r0-eigen-chunk032",
        "job00-initial-refine-intent032",
        "../bad",
        "job00-garbage",
    ],
)
def test_inventory_rejects_unregistered_names(inv, name):
    with pytest.raises(ValueError):
        inv.upper_fields(name)


def common():
    return importlib.import_module("variable_gate_common")


def test_seed_domains_are_fixed_and_distinct(inv):
    c = common()
    seeds = c.seed_map()
    assert (
        len(seeds) == 6
        and len({v for row in seeds.values() for v in row.values()}) == 12
    )
    import hashlib

    assert seeds["job00"]["initial"] == int.from_bytes(
        hashlib.sha256(
            b"modelforge-variable-gate-development-v1/job00/initial"
        ).digest(),
        "big",
    )
    with pytest.raises(ValueError):
        c.seed("job00", "teacher")


def test_trace_shape_and_counter_guards(inv, tmp_path):
    import numpy as np

    c = common()
    store = c.TraceStore(
        tmp_path / "trace", tmp_path / "progress.json", {"job00": (32, 192)}
    )
    with pytest.raises(ValueError):
        store.write(
            "job00-final",
            dict(
                directions=np.zeros((608, 32)),
                losses=np.zeros(608),
                cosines=np.zeros((608, 608)),
                count=np.array([608.0]),
            ),
        )
    assert not store.store.names
    store = c.TraceStore(
        tmp_path / "next", tmp_path / "next-progress.json", {"job00": (32, 192)}
    )
    with pytest.raises(ValueError):
        store.write(
            "job00-initial-refine-intent000",
            dict(index=np.array([8.0]), requested=np.array([8.0])),
        )
    assert not store.store.names


def test_trace_failure_is_terminal(inv, tmp_path, monkeypatch):
    import numpy as np

    c = common()
    store = c.TraceStore(
        tmp_path / "trace", tmp_path / "progress.json", {"job00": (32, 192)}
    )
    original = c.save_json

    def fail(*args):
        raise OSError("fixture progress write")

    monkeypatch.setattr(c, "save_json", fail)
    with pytest.raises(OSError):
        store.write("job00-basis", {"basis": np.eye(528)[:192]})
    assert store.store.terminal
    monkeypatch.setattr(c, "save_json", original)
    with pytest.raises(ValueError):
        store.write("job00-basis", {"basis": np.eye(528)[:192]})


def test_trace_read_header_shape_and_extra_fields(inv, tmp_path):
    import numpy as np

    c = common()
    store = c.TraceStore(
        tmp_path / "trace", tmp_path / "progress.json", {"job00": (32, 192)}
    )
    store.write("job00-basis", {"basis": np.eye(528)[:192]})
    a = c.read_trace(store.store.root, "job00-basis", 32, 192)
    assert a["basis"].shape == (192, 528)
    with pytest.raises(ValueError):
        store.write("job00-r0-input", {"known": np.ones((96, 32))})


def test_controller_requires_full_approval_before_any_engine(inv, tmp_path):
    c = importlib.import_module("variable_gate_controller")
    root = tmp_path / "repo"
    root.mkdir()
    from fresh_rank_common import save_json

    save_json(root / "config.json", {})
    save_json(
        root / "review.json", {"component_approved": True, "execution_approved": False}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("engine must not run")

    with pytest.raises(ValueError, match="Full"):
        c.run_protocol(
            root,
            tmp_path / "output",
            root / "config.json",
            root / "review.json",
            engine=forbidden,
        )
    assert not (tmp_path / "experiments" / c.MARKER).exists()


def test_outer_rejects_existing_output(inv, tmp_path):
    c = importlib.import_module("variable_gate_controller")
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError):
        c.launch(dict(output=str(out), repo=str(tmp_path)))


def test_outer_failed_monitor_cannot_claim_completion(inv, tmp_path):
    c = importlib.import_module("variable_gate_controller")
    out = tmp_path / "result"

    def fake(*args, **kwargs):
        out.mkdir()
        from fresh_rank_common import save_json

        save_json(out / "controller-report.json", {"status": "completed"})
        return dict(
            status="completed",
            process_monitor_scope="process_only_children_unavailable",
            memory_samples=1,
            peak_sampled_rss_bytes=1024,
            memory_limit_mb=512,
        )

    with pytest.raises(ValueError):
        c.launch(dict(output=str(out), repo=str(tmp_path)), engine=fake)
    from fresh_rank_common import read_json

    assert read_json(out / "outer-monitor.json")["status"] == "failed"


def test_rng_trace_rejects_non_json_state(inv, tmp_path):
    import numpy as np

    c = common()
    store = c.TraceStore(
        tmp_path / "trace", tmp_path / "progress.json", {"job00": (32, 192)}
    )
    with pytest.raises(ValueError):
        store.write(
            "job00-rng-before",
            dict(
                initial=np.zeros(512),
                rescue=np.zeros(512),
                initial_length=np.array([1.0]),
                rescue_length=np.array([1.0]),
            ),
        )


def test_all_maximum_literal_transport_families_roundtrip(inv, tmp_path):
    c = importlib.import_module("variable_gate_controller")
    receipt = c.synthetic_transport(tmp_path / "transport")
    c.validate_transport(tmp_path / "transport", receipt)
    assert receipt["shards"] == 20


def test_read_trace_explicit_descriptor_cannot_redirect(inv, tmp_path):
    c = common()
    with pytest.raises(ValueError, match="descriptor"):
        c.read_trace(
            tmp_path,
            "job00-basis",
            32,
            192,
            descriptor={"name": "../secret.npz", "bytes": 1, "sha256": "0" * 64},
        )


def test_completed_case_rejects_partial_chunk(inv, tmp_path, monkeypatch):
    import numpy as np

    c = importlib.import_module("variable_gate_controller")
    from variable_gate_view import GateJob
    from variable_gate_search import GateResult

    j = GateJob("job00", np.eye(3)[:2])
    result = GateResult(
        "zero_candidates",
        np.empty((0, 2)),
        dict(
            rounds=[],
            base_count=0,
            target_hypothesis=1,
            final_count=0,
            certificate=False,
            coverage_certified=False,
            private_gate="closed",
        ),
    )
    names = c.expected_case_names(j.job_id, result)
    monkeypatch.setattr(c, "records", lambda root: {n + ".npz": {} for n in names})

    def read(root, name, *args, **kwargs):
        if "refine-chunk" in name:
            return dict(
                complete=np.array([0.0]),
                count=np.array([7.0]),
                attempted=np.array([8.0]),
            )
        return {}

    monkeypatch.setattr(c, "read_trace", read)

    class Store:
        class store:
            root = tmp_path

    with pytest.raises(ValueError, match="chunk"):
        c.validate_case(j, result, Store())


@pytest.fixture
def controlled_fixture(inv, tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from fresh_rank_common import save_json
    from variable_gate_view import GateJob, GateBundle
    import variable_gate_search as kernel

    c = importlib.import_module("variable_gate_controller")
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "experiments" / "run"
    config = dict(
        parent_directory=c.PARENT_NAME,
        input_commitments={
            k: "0" * 64
            for k in ("cases_sha256", "manifest_sha256", "seal_sha256", "report_sha256")
        },
    )
    configpath = repo / "config.json"
    reviewpath = repo / "review.json"
    save_json(configpath, config)
    save_json(reviewpath, {})
    monkeypatch.setattr(c, "verify_source", lambda *args: (config, "a" * 40))
    monkeypatch.setattr(
        c.shutil, "disk_usage", lambda *args: SimpleNamespace(free=4 * 1024**3)
    )
    jobs = tuple(GateJob(f"job{i:02d}", np.eye(528)[:2]) for i in range(6))
    monkeypatch.setattr(
        c,
        "load_bundle",
        lambda *args: GateBundle(
            jobs, dict(mapping={j.job_id: "fixture" for j in jobs})
        ),
    )
    # Fixed literal unit fixtures; never instantiate registered seed domains or
    # call the registered initializer, teacher, oracle or public input loaders.
    monkeypatch.setattr(
        c,
        "streams",
        lambda *args: (np.random.default_rng(11), np.random.default_rng(12)),
    )

    def kernel_fixture(j, initial, rescue, sink):
        v = np.zeros((256, 32))
        v[:, -1] = 1

        def init(j, rng, s):
            s.write(j.job_id + "-initial-raw", {"raw": v})
            s.write(j.job_id + "-initial-starts", {"starts": v})
            return v.copy(), v.copy()

        return kernel._search_fixture(
            j,
            initial,
            rescue,
            sink,
            initializer=init,
            refiner=lambda x, b, steps: x.copy(),
        )

    monkeypatch.setattr(c, "search_job", kernel_fixture)

    def views(j, result, store, *args):
        store.write(
            j.job_id + "-view-residuals",
            dict(losses=np.zeros((3, 1)), count=np.array([0.0])),
        )

    monkeypatch.setattr(c, "view_diagnostics", views)

    def engine(command, payload, **kwargs):
        receipt = (
            c.synthetic_transport(Path(payload["output"]))
            if payload.get("operation") == "transport"
            else {"status": "passed"}
        )
        return dict(
            receipt,
            process_monitor_scope="process_and_children",
            memory_samples=1,
            peak_sampled_rss_bytes=16 * 1024**2,
            memory_limit_mb=512,
        )

    return c, repo, output, configpath, reviewpath, engine


def test_controlled_six_scientific_failures_complete_without_fake_success(
    controlled_fixture,
):
    c, repo, out, config, review, engine = controlled_fixture
    report = c.run_protocol(repo, out, config, review, engine=engine)
    assert report["status"] == "completed" and report["candidate_target_jobs"] == 0
    assert len(report["cases"]) == 6 and all(
        x["candidate_status"] == "zero_candidates" for x in report["cases"]
    )
    assert report["method_promotion"] is False and report["new_queries"] == 0
    assert (out.parent / c.MARKER).exists()
    assert (out / "trace-progress.json").exists()


def test_failed_preflight_keeps_scope_receipt_without_marker(
    controlled_fixture, monkeypatch
):
    c, repo, out, config, review, engine = controlled_fixture
    from types import SimpleNamespace

    monkeypatch.setattr(c.shutil, "disk_usage", lambda *args: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="2GiB"):
        c.run_protocol(repo, out, config, review, engine=engine)
    from fresh_rank_common import read_json

    saved = read_json(out / "controller-report.json")
    assert (
        saved["scope_receipt"]["status"] == "passed"
        and saved["transport_receipt"]["status"] == "passed"
    )
    assert not saved["registered_rng_started"] and not (out.parent / c.MARKER).exists()


def test_consumed_marker_blocks_all_engines(controlled_fixture):
    c, repo, out, config, review, engine = controlled_fixture
    out.parent.mkdir()
    (out.parent / c.MARKER).write_text("already consumed")

    def forbidden(*args, **kwargs):
        raise AssertionError("must not invoke engine")

    with pytest.raises(ValueError, match="Consumed"):
        c.run_protocol(repo, out, config, review, engine=forbidden)


def test_real_clean_git_source_gate_and_mutation(inv, tmp_path):
    import subprocess
    import platform
    import numpy as np

    c = importlib.import_module("variable_gate_controller")
    from fresh_rank_common import save_json, sha

    repo = tmp_path / "source"
    repo.mkdir()
    config_path = repo / "experiments/preregistrations/fixture.json"
    config_path.parent.mkdir(parents=True)
    for name in c.REQUIRED_REVIEW:
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("literal source gate fixture\n")
    config = dict(
        policy=c.POLICY,
        inventory=inv.inventory(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        seeds=c.seed_map(),
        parent_directory=c.PARENT_NAME,
        parent_source_commit=c.PARENT_SOURCE,
        parent_outcome_audit_sha256=c.PARENT_AUDIT,
        input_commitments=c.INPUT_PINS,
        frozen_dependencies={n: sha(repo / n) for n in c.DEPENDENCIES},
    )
    save_json(config_path, config)
    signed = {n: sha(repo / n) for n in c.REQUIRED_REVIEW}
    signed[str(config_path.relative_to(repo))] = sha(config_path)
    review = repo / "review.json"
    save_json(
        review, dict(full_variable_gate_protocol_approved=True, source_sha256=signed)
    )
    subprocess.run(["git", "init", "-q", "-b", "discovery-lab"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "source gate literal fixture",
        ],
        cwd=repo,
        check=True,
    )
    data, head = c.verify_source(repo, config_path, review)
    assert len(head) == 40 and data["seeds"] == c.seed_map()
    target = repo / next(iter(c.DEPENDENCIES))
    target.write_text("changed fixture")
    with pytest.raises(ValueError, match="bytes changed"):
        c.verify_source(repo, config_path, review)


def test_controller_binds_saved_basis_to_actual_job(controlled_fixture, monkeypatch):
    import numpy as np

    c, repo, out, config, review, engine = controlled_fixture
    original = c.read_trace

    def wrong(root, name, *args, **kwargs):
        value = original(root, name, *args, **kwargs)
        if name == "job00-basis":
            return {"basis": np.zeros_like(value["basis"])}
        return value

    monkeypatch.setattr(c, "read_trace", wrong)
    with pytest.raises(ValueError, match="basis"):
        c.run_protocol(repo, out, config, review, engine=engine)


def test_outer_receipt_write_failure_never_remains_completed(
    inv, tmp_path, monkeypatch
):
    c = importlib.import_module("variable_gate_controller")
    out = tmp_path / "output"
    from fresh_rank_common import save_json, read_json

    def engine(*args, **kwargs):
        out.mkdir()
        save_json(out / "controller-report.json", {"status": "completed"})
        return dict(
            status="completed",
            process_monitor_scope="process_and_children",
            memory_samples=1,
            peak_sampled_rss_bytes=16 * 1024**2,
            memory_limit_mb=512,
        )

    def fail_completed(path, value):
        save_json(path, value)
        if path.name == "outer-monitor.json" and value["status"] == "completed":
            raise OSError("fixture after receipt write")

    monkeypatch.setattr(c, "save_json", fail_completed)
    with pytest.raises(OSError):
        c.launch(dict(output=str(out), repo=str(tmp_path)), engine=engine)
    assert read_json(out / "outer-monitor.json")["status"] == "failed"
