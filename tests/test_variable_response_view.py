import dataclasses
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def module_path(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )


def module():
    assert importlib.util.find_spec("variable_response_manifest") is not None, (
        "public-only commitment compiler missing"
    )
    return importlib.import_module("variable_response_manifest")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "transcripts"
    root.mkdir()
    repo = tmp_path / "public"
    repo.mkdir()
    original = "/owned/prior/collection"
    names = ("c00-fp64", "c00-fp32")
    hashes = {}
    for name in names:
        for step in range(2):
            for probe in range(2):
                prefix = f"{name}-s{step}-x{probe:02d}"
                for field, columns in [("inputs", 2), ("responses", 1)]:
                    path = root / (prefix + "-" + field + ".npz")
                    data = np.full(
                        (12, columns), 1 + step + probe / 10, dtype=np.float64
                    )
                    np.savez(path, **{field: data})
                    hashes[original + "/truth/" + path.name] = digest(path)
    # These are forbidden adjacent files; no adapter may open/enumerate them.
    (root / "manifest.jsonl").write_text("FORBIDDEN")
    (root / "c00-context.npz").write_text("FORBIDDEN")
    rows, batches = 96, 8
    counts = dict(
        calls_reserved=rows,
        calls_attempted=rows,
        calls_returned=rows,
        calls_persisted=rows,
        callback_batches=batches,
    )
    collection = dict(
        status="completed",
        source_commit="07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a",
        **counts,
    )
    progress = dict(status="persisted", batch="c00-fp32-s1-x01", **counts)

    def save(name, value):
        path = repo / (name + ".json")
        path.write_text(json.dumps(value, sort_keys=True))
        return path

    cp = save("collection", collection)
    qp = save("progress", progress)
    cm = save(
        "commitments",
        dict(
            artifact_sha256=hashes,
            metadata_sha256={
                "collection/collection-report.json": digest(cp),
                "collection/query-progress.json": digest(qp),
            },
        ),
    )
    ap = save(
        "audit",
        dict(
            approved=True,
            saved_outcome_audit_approved=True,
            actual_source_commit=collection["source_commit"],
            artifact_commitments_sha256=digest(cm),
            vector_calls=rows,
            callback_batches=batches,
            private_gate="closed",
        ),
    )
    paths = dict(audit=ap, commitments=cm, collection=cp, progress=qp)

    def pins():
        return {key: (path, digest(path)) for key, path in paths.items()}

    def compile_view():
        m = module()
        return m._compile(pins(), m._Profile(2, 1, 2, names), original)

    return dict(
        root=root,
        repo=repo,
        paths=paths,
        pins=pins,
        compile=compile_view,
        original=original,
        names=names,
        hashes=hashes,
    )


def rewrite(f, name, change):
    path = f["paths"][name]
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, sort_keys=True))
    # Rebind public lineage deliberately for schema tests, not hash-mutation tests.
    if name in ("collection", "progress"):
        key = (
            "/collection-report.json"
            if name == "collection"
            else "/query-progress.json"
        )
        rewrite(
            f,
            "commitments",
            lambda d: d["metadata_sha256"].__setitem__(
                "collection" + key, digest(path)
            ),
        )
    elif name == "commitments":
        rewrite(
            f,
            "audit",
            lambda d: d.__setitem__("artifact_commitments_sha256", digest(path)),
        )


def test_public_compiler_never_reads_raw_or_lists_truth(fixture, monkeypatch):
    f = fixture
    pins = f["pins"]()
    m = module()
    opened = []
    original = Path.open

    def guarded(path, *a, **kw):
        assert f["root"] not in path.parents and path != f["root"]
        opened.append(path)
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(
        Path, "iterdir", lambda _: pytest.fail("directory enumeration forbidden")
    )
    result = m._compile(pins, m._Profile(2, 1, 2, f["names"]), f["original"])
    assert len(result.pairs) == 8 and result.profile.rows == 12
    assert [p.job_id for p in result.pairs] == ["job00"] * 4 + ["job01"] * 4
    assert result.pairs[-1].inputs.name == "c00-fp32-s1-x01-inputs.npz"
    assert result.pairs[-1].responses.field == "responses"
    assert result.profile.vector_rows == 96
    assert len(opened) == 4


def test_default_profile_and_production_pins_are_fixed():
    m = module()
    p = m.REGISTERED
    assert (p.n, p.outputs, p.probes, p.rows, p.vector_rows) == (
        32,
        32,
        17,
        2112,
        430848,
    )
    assert len(p.case_names) == 6 and len(m.PUBLIC_PINS) == 4
    assert m.ENCODED_CAP == 544768 and m.HEADER_CAP == 4096
    assert m.COMMITMENTS_CAP == 2 * 1024**2 and m.METADATA_CAP == 128 * 1024


@pytest.mark.parametrize("name", ["audit", "commitments", "collection", "progress"])
def test_wrong_public_hash_prevents_compilation(fixture, name):
    f = fixture
    pins = f["pins"]()
    pins[name] = (pins[name][0], "0" * 64)
    with pytest.raises(ValueError, match="hash"):
        module()._compile(pins, module()._Profile(2, 1, 2, f["names"]), f["original"])


@pytest.mark.parametrize(
    "where,kind",
    [
        ("commitments", "missing"),
        ("commitments", "digest"),
        ("collection", "boolean"),
        ("progress", "boolean"),
        ("audit", "approval"),
        ("audit", "source"),
    ],
)
def test_invalid_public_lineage_or_counter_rejected(fixture, where, kind):
    f = fixture

    def change(d):
        if kind == "missing":
            d["artifact_sha256"].pop(next(iter(d["artifact_sha256"])))
        elif kind == "digest":
            d["artifact_sha256"][next(iter(d["artifact_sha256"]))] = "not-a-digest"
        elif kind == "boolean":
            d["calls_persisted"] = True
        elif kind == "approval":
            d["saved_outcome_audit_approved"] = False
        else:
            d["actual_source_commit"] = "0" * 40

    rewrite(f, where, change)
    with pytest.raises(ValueError):
        f["compile"]()


def test_duplicate_public_json_keys_and_metadata_cap_fail_before_parse(fixture):
    f = fixture
    p = f["paths"]["audit"]
    p.write_text('{"approved":true,"approved":false}')
    with pytest.raises(ValueError, match="duplicate"):
        f["compile"]()
    p.write_bytes(b" " * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="cap|bounded"):
        f["compile"]()


def test_manifest_descriptors_are_immutable(fixture):
    result = fixture["compile"]()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.pairs[0].job_id = "other"
    assert isinstance(result.pairs, tuple) and isinstance(result.metadata, tuple)


def view_module():
    assert importlib.util.find_spec("variable_response_view") is not None, (
        "streamed view missing"
    )
    return importlib.import_module("variable_response_view")


def rebound_array(f, path, **fields):
    np.savez(path, **fields)
    rewrite(
        f,
        "commitments",
        lambda d: d["artifact_sha256"].__setitem__(
            f["original"] + "/truth/" + path.name, digest(path)
        ),
    )


def test_streamed_view_canonical_immutable_pairs_and_separate_counters(
    fixture, monkeypatch
):
    f = fixture
    manifest = f["compile"]()
    raw_open = Path.open

    def guarded(path, *args, **kw):
        if path.parent == f["root"]:
            assert path.name not in ("manifest.jsonl", "c00-context.npz")
        return raw_open(path, *args, **kw)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(
        Path, "iterdir", lambda _: pytest.fail("directory enumeration forbidden")
    )
    view = view_module()._open_fixture_view(f["root"], manifest)
    report = view.report()
    assert report["validation"]["archives_read"] == 16
    assert report["delivery"]["pairs"] == 0
    assert [job.job_id for job in view.jobs] == ["job00", "job01"]
    for job in view.jobs:
        assert [x.name for x in dataclasses.fields(job)] == [
            "job_id",
            "n",
            "outputs",
            "batch_count",
            "batches",
        ]
        assert (job.n, job.outputs, job.batch_count) == (2, 1, 4)
        batches = list(job.batches)
        assert len(batches) == 4
        assert [x.name for x in dataclasses.fields(batches[0])] == [
            "inputs",
            "responses",
        ]
        np.testing.assert_array_equal(batches[3].inputs, np.full((12, 2), 2.1))
        np.testing.assert_array_equal(batches[3].responses, np.full((12, 1), 2.1))
        with pytest.raises(ValueError):
            batches[0].inputs.setflags(write=True)
        with pytest.raises(ValueError):
            batches[0].responses[0, 0] = 9
        assert list(job.batches) == []  # Same exhausted iterator, no second read.
    final = view.finish()
    assert final["status"] == "completed" and final["delivery"]["pairs"] == 8
    assert final["delivery"]["rows"] == 96
    assert (
        final["delivery"]["archives_read"] == 16
        and final["completion"]["archives_read"] == 16
    )
    assert final["delivery"]["payload_bytes"] == 2304
    assert final["new_queries"] == 0 and final["certificate"] is False
    assert view.report() == final
    with pytest.raises(ValueError):
        view.finish()


def test_partial_delivery_cannot_be_completed_or_restarted(fixture):
    view = view_module()._open_fixture_view(fixture["root"], fixture["compile"]())
    first = next(view.jobs[0].batches)
    assert first.inputs.shape == (12, 2)
    assert view.report()["jobs"]["job00"]["status"] == "partial"
    with pytest.raises(ValueError, match="partial|delivered"):
        view.finish()
    assert view.report()["delivery"]["pairs"] == 1
    assert next(view.jobs[0].batches).inputs.shape == (12, 2)
    assert view.report()["delivery"]["pairs"] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "hash",
        "field",
        "shape",
        "dtype",
        "nonfinite",
        "oversize",
        "member",
        "duplicate",
        "header",
        "symlink",
        "hardlink",
    ],
)
def test_late_invalid_archive_aborts_before_any_job_exposure(fixture, kind):
    import io
    import os
    import zipfile

    f = fixture
    path = f["root"] / "c00-fp32-s1-x01-responses.npz"
    if kind == "hash":
        path.write_bytes(path.read_bytes() + b"changed")
    elif kind == "field":
        rebound_array(f, path, factors=np.ones((12, 1)))
    elif kind == "shape":
        rebound_array(f, path, responses=np.ones((11, 1)))
    elif kind == "dtype":
        rebound_array(f, path, responses=np.ones((12, 1), dtype=np.float32))
    elif kind == "nonfinite":
        rebound_array(f, path, responses=np.full((12, 1), np.nan))
    elif kind == "oversize":
        path.write_bytes(b"x" * 544769)
        rewrite(
            f,
            "commitments",
            lambda d: d["artifact_sha256"].__setitem__(
                f["original"] + "/truth/" + path.name, digest(path)
            ),
        )
    elif kind in ("member", "duplicate", "header"):
        buff = io.BytesIO()
        np.save(buff, np.ones((12, 1)))
        with zipfile.ZipFile(path, "w") as z:
            if kind == "header":
                # A malicious huge declared shape must be rejected before allocation.
                payload = io.BytesIO()
                np.lib.format.write_array_header_1_0(
                    payload, dict(descr="<f8", fortran_order=False, shape=(10**9, 1))
                )
                z.writestr("responses.npy", payload.getvalue())
            else:
                z.writestr("responses.npy", buff.getvalue())
                if kind == "member":
                    z.writestr("context.npy", buff.getvalue())
                else:
                    with pytest.warns(UserWarning):
                        z.writestr("responses.npy", buff.getvalue())
        rewrite(
            f,
            "commitments",
            lambda d: d["artifact_sha256"].__setitem__(
                f["original"] + "/truth/" + path.name, digest(path)
            ),
        )
    else:
        other = path.with_suffix(".original")
        path.rename(other)
        if kind == "symlink":
            path.symlink_to(other)
        else:
            os.link(other, path)
    with pytest.raises(ValueError):
        view_module()._open_fixture_view(f["root"], f["compile"]())


def test_pair_revalidated_before_delivery_and_failure_is_terminal(fixture):
    f = fixture
    view = view_module()._open_fixture_view(f["root"], f["compile"]())
    next(view.jobs[0].batches)
    p = f["root"] / "c00-fp64-s0-x01-responses.npz"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        next(view.jobs[0].batches)
    r = view.report()
    assert r["status"] == "failed" and r["delivery"]["pairs"] == 1
    assert r["jobs"]["job00"]["status"] == "failed"
    with pytest.raises(ValueError):
        next(view.jobs[1].batches)
    assert view.report()["delivery"]["pairs"] == 1
    with pytest.raises(ValueError):
        view.finish()


@pytest.mark.parametrize("where", ["data", "metadata"])
def test_completion_rehashes_all_files_and_public_metadata(fixture, where):
    f = fixture
    view = view_module()._open_fixture_view(f["root"], f["compile"]())
    for job in view.jobs:
        list(job.batches)
    p = (
        f["root"] / "c00-fp64-s0-x00-inputs.npz"
        if where == "data"
        else f["paths"]["audit"]
    )
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        view.finish()
    assert view.report()["status"] == "failed"
    assert view.report()["delivery"]["pairs"] == 8


def test_validator_does_not_cache_all_arrays(fixture, monkeypatch):
    import gc
    import weakref

    v = view_module()
    refs = []
    load = v._array

    def tracked(*args, **kwargs):
        a = load(*args, **kwargs)
        refs.append(weakref.ref(a))
        return a

    monkeypatch.setattr(v, "_array", tracked)
    view = v._open_fixture_view(fixture["root"], fixture["compile"]())
    gc.collect()
    assert len(refs) == 16 and all(ref() is None for ref in refs)
    assert view.report()["delivery"]["pairs"] == 0


def test_public_open_view_rejects_fixture_profile_and_forged_descriptor(fixture):
    f = fixture
    manifest = f["compile"]()
    v = view_module()
    with pytest.raises(ValueError, match="registered"):
        v.open_view(f["root"], manifest)
    pair = dataclasses.replace(
        manifest.pairs[0],
        inputs=dataclasses.replace(manifest.pairs[0].inputs, name="../context.npz"),
    )
    forged = dataclasses.replace(manifest, pairs=(pair, *manifest.pairs[1:]))
    with pytest.raises(ValueError, match="manifest|commitment"):
        v._open_fixture_view(f["root"], forged)


def test_registered_case_order_is_enforced_before_delivery(fixture):
    view = view_module()._open_fixture_view(fixture["root"], fixture["compile"]())
    with pytest.raises(ValueError, match="order"):
        next(view.jobs[1].batches)
    assert (
        view.report()["delivery"]["pairs"] == 0 and view.report()["status"] == "failed"
    )


def test_idle_generator_does_not_retain_a_delivered_pair(fixture):
    import gc
    import weakref

    view = view_module()._open_fixture_view(fixture["root"], fixture["compile"]())
    batch = next(view.jobs[0].batches)
    refs = (weakref.ref(batch.inputs), weakref.ref(batch.responses))
    del batch
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_failed_initial_validation_preserves_partial_receipt(fixture):
    f = fixture
    p = f["root"] / "c00-fp32-s1-x01-responses.npz"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError) as caught:
        view_module()._open_fixture_view(f["root"], f["compile"]())
    report = caught.value.report
    assert report["status"] == "failed" and report["delivery"]["pairs"] == 0
    assert report["validation"]["archives_read"] == 16
    assert report["validation"]["arrays_validated"] == 15
    assert all(j["status"] == "not_started" for j in report["jobs"].values())
