import importlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def module_path(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )


def inventory_module():
    assert importlib.util.find_spec("variable_response_inventory") is not None, (
        "read inventory missing"
    )
    return importlib.import_module("variable_response_inventory")


def common():
    assert importlib.util.find_spec("variable_response_protocol") is not None, (
        "read protocol missing"
    )
    return importlib.import_module("variable_response_protocol")


def test_scalar_read_inventory_covers_three_passes_and_prior_artifacts():
    v = inventory_module().inventory()
    assert v["pairs"] == 204 and v["archives"] == 408 and v["vector_rows"] == 430848
    assert v["raw_payload_bytes"] == 220594176
    assert v["encoded_pass_upper_bound"] == 222265344
    assert v["complete_archive_reads"] == 1224 and v["complete_array_decodes"] == 816
    assert v["encoded_read_bytes_upper_bound"] == 666796032
    assert (
        v["transport_upper_bound"] == 1093632
        and v["new_artifacts_upper_bound"] == 9482240
    )
    assert v["combined_artifacts_upper_bound"] == 679665861 < 1024**3
    assert v["descriptor_seal_upper_bound"] < 128 * 1024
    assert v["delivery_upper_bound"] < 128 * 1024


def test_atomic_bounded_json_and_exclusive_marker(tmp_path):
    m = common()
    p = tmp_path / "report.json"
    m.save(p, {"status": "ready"})
    assert m.read(p)["status"] == "ready"
    before = p.read_bytes()
    with pytest.raises(ValueError):
        m.save(p, {"huge": "x" * (128 * 1024)})
    assert p.read_bytes() == before
    marker = tmp_path / "attempt.json"
    h = m.marker(marker, {"source": "literal"})
    assert m.digest(marker) == h
    with pytest.raises(FileExistsError):
        m.marker(marker, {"source": "retry"})
    assert m.read(marker)["source"] == "literal"


def test_tree_receipt_rejects_process_only_boolean_and_zero():
    m = common()
    receipt = dict(
        status="passed",
        process_monitor_scope="process_and_children",
        memory_samples=2,
        peak_sampled_rss_bytes=1234,
        memory_limit_mb=512,
    )
    assert m.tree_ok(receipt)
    for key, value in [
        ("process_monitor_scope", "process_only"),
        ("memory_samples", True),
        ("peak_sampled_rss_bytes", 0),
        ("memory_limit_mb", 128),
    ]:
        bad = dict(receipt, **{key: value})
        assert not m.tree_ok(bad)


def test_delivery_chain_is_ordered_and_bounded(tmp_path):
    m = common()
    p = tmp_path / "delivery.jsonl"
    chain = m.DeliveryChain(p)
    for i in range(2):
        chain.append("job00", i, (4, 1), (4, 1), "1" * 64, "2" * 64)
    result = chain.verify()
    assert result["pairs"] == 2 and result["head"] == chain.head
    records = [json.loads(x) for x in p.read_text().splitlines()]
    assert records[1]["previous"] == records[0]["chain"]
    p.write_text("\n".join(json.dumps(x) for x in records[::-1]) + "\n")
    with pytest.raises(ValueError):
        chain.verify()


def test_duplicate_json_keys_and_oversized_marker_rejected(tmp_path):
    m = common()
    p = tmp_path / "bad.json"
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        m.read(p)
    marker = tmp_path / "attempt.json"
    with pytest.raises(ValueError):
        m.marker(marker, {"too_big": "x" * (128 * 1024)})
    assert not marker.exists()


def controller():
    assert importlib.util.find_spec("variable_response_controller") is not None, (
        "read controller missing"
    )
    return importlib.import_module("variable_response_controller")


@pytest.fixture
def controlled(tmp_path):
    import hashlib
    import subprocess
    import numpy as np
    from variable_response_manifest import _compile, _Profile
    from variable_response_view import _open_fixture_view

    c = controller()
    repo = tmp_path / "repository"
    repo.mkdir()
    origin = Path(__file__).parents[1]
    names = tuple(f"c{i:02d}-{ch}" for i in range(3) for ch in ("fp64", "fp32"))
    parent = repo.parent / "experiments"
    parent.mkdir()
    rawroot = parent / c.PARENT / "collection/truth"
    rawroot.mkdir(parents=True)
    public = tmp_path / "literal-public"
    public.mkdir()
    original = "/owned/collection"

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def save(p, v):
        p.write_text(json.dumps(v, sort_keys=True))
        return sha(p)

    hashes = {}
    for name in names:
        for step in range(2):
            for field in ("inputs", "responses"):
                p = rawroot / f"{name}-s{step}-x00-{field}.npz"
                np.savez(p, **{field: np.ones((4, 1)) * (step + 1)})
                hashes[original + "/truth/" + p.name] = sha(p)
    (rawroot / "manifest.jsonl").write_text("FORBIDDEN")
    counts = dict(
        calls_reserved=48,
        calls_attempted=48,
        calls_returned=48,
        calls_persisted=48,
        callback_batches=12,
    )
    cp, qp, cm, ap = [
        public / (n + ".json")
        for n in ("collection", "progress", "commitments", "audit")
    ]
    save(cp, dict(status="completed", source_commit=c.PARENT_SOURCE, **counts))
    save(qp, dict(status="persisted", batch="c02-fp32-s1-x00", **counts))
    save(
        cm,
        dict(
            artifact_sha256=hashes,
            metadata_sha256={
                "collection/collection-report.json": sha(cp),
                "collection/query-progress.json": sha(qp),
            },
        ),
    )
    save(
        ap,
        dict(
            approved=True,
            saved_outcome_audit_approved=True,
            actual_source_commit=c.PARENT_SOURCE,
            artifact_commitments_sha256=sha(cm),
            vector_calls=48,
            callback_batches=12,
            private_gate="closed",
        ),
    )
    profile = _Profile(1, 1, 1, names)
    pins = {
        n: (p, sha(p))
        for n, p in [
            ("audit", ap),
            ("commitments", cm),
            ("collection", cp),
            ("progress", qp),
        ]
    }
    manifest = _compile(pins, profile, original)
    for n in c.REQUIRED_REVIEW:
        q = repo / n
        q.parent.mkdir(parents=True, exist_ok=True)
        if (origin / n).exists():
            q.write_bytes((origin / n).read_bytes())
        else:
            q.write_text("literal fixture source")
    config = repo / "experiments/preregistrations/variable-response-view-v1.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    review = repo / "experiments/evidence/read-test/source-review.json"
    review.parent.mkdir(parents=True, exist_ok=True)
    value = c.config_template(repo)
    save(config, value)
    signed = {
        n: sha(repo / n) for n in c.REQUIRED_REVIEW | {str(config.relative_to(repo))}
    }
    save(
        review,
        dict(full_variable_response_read_protocol_approved=True, source_sha256=signed),
    )

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init", "-b", "discovery-lab")
    git("config", "user.name", "Literal")
    git("config", "user.email", "literal@example.invalid")
    git("add", ".")
    git("commit", "-qm", "literal source freeze")
    output = parent / "read-literal"
    calls = []
    receipt = dict(
        status="passed",
        process_monitor_scope="process_and_children",
        memory_samples=3,
        peak_sampled_rss_bytes=123456,
        memory_limit_mb=512,
    )

    def engine(command, payload, **kwargs):
        calls.append((command, payload, kwargs))
        assert kwargs["memory_limit_mb"] == 512 and kwargs["output_limit_bytes"] == 2048
        if payload.get("operation") == "transport":
            return dict(
                c.synthetic_transport(Path(payload["output"])),
                **{k: v for k, v in receipt.items() if k != "status"},
            )
        return dict(receipt)

    def open_literal(root, m):
        assert (parent / common().MARKER).exists(), "transcript read before marker"
        assert root == rawroot
        return _open_fixture_view(root, m)

    def run(**kwargs):
        return c.run_protocol(
            repo,
            output,
            config,
            review,
            engine=engine,
            _manifest_loader=lambda _: manifest,
            _view_factory=open_literal,
            **kwargs,
        )

    return dict(
        c=c,
        repo=repo,
        output=output,
        config=config,
        review=review,
        run=run,
        engine=engine,
        calls=calls,
        parent=parent,
        manifest=manifest,
        rawroot=rawroot,
        open=open_literal,
        git=git,
        sha=sha,
    )


def test_complete_literal_protocol_is_readonly_and_commits_every_pair(controlled):
    f = controlled
    result = f["run"]()
    assert result["status"] == "completed" and result["marker_consumed"] is True
    assert result["view"]["validation"]["archives_read"] == 24
    assert (
        result["view"]["delivery"]["pairs"] == 12
        and result["view"]["delivery"]["rows"] == 48
    )
    assert result["view"]["completion"]["archives_read"] == 24
    assert result["delivery_chain"]["pairs"] == 12
    assert (
        result["new_queries"] == 0
        and result["new_fits"] == 0
        and result["certificate"] is False
    )
    assert len(f["calls"]) == 2
    assert (
        common().read(f["output"] / "controller-report.json")["status"] == "completed"
    )
    assert (f["rawroot"] / "manifest.jsonl").read_text() == "FORBIDDEN"


@pytest.mark.parametrize("where", ["source", "config", "review", "dirty"])
def test_clean_freeze_gate_rejects_changed_source_before_output(controlled, where):
    f = controlled
    if where == "source":
        p = f["repo"] / next(iter(f["c"].DEPENDENCIES))
    elif where == "config":
        p = f["config"]
    elif where == "review":
        p = f["review"]
    else:
        p = f["repo"] / "untracked.txt"
    p.write_bytes((p.read_bytes() if p.exists() else b"") + b"changed")
    with pytest.raises(ValueError):
        f["run"]()
    assert not f["output"].exists() and not (f["parent"] / common().MARKER).exists()


def test_existing_marker_or_output_never_retries(controlled):
    f = controlled
    common().marker(f["parent"] / common().MARKER, dict(old=True))
    with pytest.raises(ValueError):
        f["run"]()
    assert not f["calls"]
    assert common().read(f["parent"] / common().MARKER) == dict(old=True)
    with pytest.raises((ValueError, FileExistsError)):
        f["run"]()
    assert not f["calls"]


@pytest.mark.parametrize("gate", ["scope", "transport", "disk"])
def test_failed_preflight_is_preserved_without_data_or_marker(
    controlled, monkeypatch, gate
):
    f = controlled
    engine = f["engine"]

    def bad_engine(command, payload, **kwargs):
        v = engine(command, payload, **kwargs)
        if (gate == "scope" and not payload) or (
            gate == "transport" and payload.get("operation") == "transport"
        ):
            v["process_monitor_scope"] = "process_only_children_unavailable"
        return v

    if gate == "disk":
        monkeypatch.setattr(
            f["c"].shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})()
        )
    with pytest.raises(ValueError):
        f["c"].run_protocol(
            f["repo"],
            f["output"],
            f["config"],
            f["review"],
            engine=bad_engine,
            _manifest_loader=lambda _: pytest.fail("data must not open"),
        )
    report = common().read(f["output"] / "controller-report.json")
    assert report["status"] == "failed" and report["marker_consumed"] is False
    assert report["scope_receipt"]
    if gate == "transport":
        assert report["transport_receipt"]["process_monitor_scope"].startswith(
            "process_only"
        )
    if gate == "disk":
        assert report["free_bytes_before"] == 1
    assert not (f["parent"] / common().MARKER).exists()


def test_initial_or_late_data_failure_keeps_consumed_marker_and_partial_counters(
    controlled,
):
    f = controlled
    p = f["rawroot"] / "c02-fp32-s1-x00-responses.npz"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        f["run"]()
    report = common().read(f["output"] / "controller-report.json")
    assert report["status"] == "failed" and report["marker_consumed"] is True
    assert report["view"]["validation"]["archives_read"] == 24
    assert report["view"]["delivery"]["pairs"] == 0
    assert (f["parent"] / common().MARKER).exists()


def test_late_marker_mutation_prevents_completed_status(controlled, monkeypatch):
    f = controlled
    original = f["c"].consume

    def altered(*args, **kwargs):
        value = original(*args, **kwargs)
        p = f["parent"] / common().MARKER
        p.write_bytes(p.read_bytes() + b"changed")
        return value

    monkeypatch.setattr(f["c"], "consume", altered)
    with pytest.raises(ValueError):
        f["run"]()
    r = common().read(f["output"] / "controller-report.json")
    assert r["status"] == "failed" and r["view"]["delivery"]["pairs"] == 12


def test_outer_bad_tree_does_not_promote_completed_controller(controlled):
    f = controlled
    # Outer fixture writes a tiny terminal child receipt; no data controller run.
    request = dict(
        repo=str(f["repo"]),
        output=str(f["parent"] / "outer-literal"),
        config=str(f["config"]),
        review=str(f["review"]),
    )
    out = Path(request["output"])

    def engine(command, payload, **kwargs):
        out.mkdir()
        common().save(out / "controller-report.json", dict(status="completed"))
        return dict(
            status="completed",
            process_monitor_scope="process_only",
            memory_samples=1,
            peak_sampled_rss_bytes=10,
            memory_limit_mb=512,
        )

    with pytest.raises(ValueError):
        f["c"].launch(request, engine=engine)
    r = common().read(out / "outer-monitor.json")
    assert (
        r["status"] == "failed"
        and r["receipt"]["process_monitor_scope"] == "process_only"
    )


@pytest.mark.parametrize("where", ["descriptor", "delivery", "view", "transport"])
def test_late_saved_evidence_mutation_prevents_completed_status(
    controlled, monkeypatch, where
):
    f = controlled
    original = f["c"].consume

    def altered(*args, **kwargs):
        value = original(*args, **kwargs)
        path = (
            f["output"]
            / (
                {
                    "descriptor": "descriptor-seal.json",
                    "delivery": "delivery.jsonl",
                    "view": "view-report.json",
                    "transport": "transport/extra.txt",
                }[where]
            )
        )
        path.write_bytes((path.read_bytes() if path.exists() else b"") + b"changed")
        return value

    monkeypatch.setattr(f["c"], "consume", altered)
    with pytest.raises(ValueError):
        f["run"]()
    assert common().read(f["output"] / "controller-report.json")["status"] == "failed"


def test_free_disk_is_rechecked_after_transport_before_marker(controlled, monkeypatch):
    f = controlled
    calls = []

    def disk(_):
        calls.append(None)
        return type("Disk", (), {"free": 3 * 1024**3 if len(calls) == 1 else 1})()

    monkeypatch.setattr(f["c"].shutil, "disk_usage", disk)
    with pytest.raises(ValueError):
        f["run"]()
    assert not (f["parent"] / common().MARKER).exists()
    assert (
        common().read(f["output"] / "controller-report.json")[
            "free_bytes_before_marker"
        ]
        == 1
    )


def test_late_chain_write_failure_distinguishes_delivered_from_committed(
    controlled, monkeypatch
):
    f = controlled
    original = f["c"].DeliveryChain.append

    def failing(self, *args, **kwargs):
        if self.count == 1:
            raise OSError("literal second chain write failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(f["c"].DeliveryChain, "append", failing)
    with pytest.raises(OSError):
        f["run"]()
    report = common().read(f["output"] / "controller-report.json")
    progress = common().read(f["output"] / "progress.json")
    assert report["status"] == "failed" and report["view"]["delivery"]["pairs"] == 2
    assert (
        progress["committed_pairs"] == 1
        and report["uncommitted_delivery"] == "unknown_after_interruption"
    )


def test_terminal_counter_validation_rejects_boolean_delivered_pair_count(controlled):
    f = controlled
    value = f["run"]()["view"]
    value["delivery"]["pairs"] = True
    with pytest.raises(ValueError):
        f["c"].validate_view(value, f["manifest"])
