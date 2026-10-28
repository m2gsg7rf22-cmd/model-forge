import dataclasses
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    from fresh_rank_common import save_json, sha
    from scaling_domain_store import ShardStore
    from row_acquisition_view import RowView

    public = tmp_path / "public"
    public.mkdir()
    store = ShardStore(public / "arrays")
    for kind in ("fine", "coarse", "richardson"):
        for start in (0, 8, 16):
            store.save(
                f"c00-fp64-{kind}-h{start // 8:02d}.npz",
                hessian=np.zeros((min(8, 17 - start), 1, 2, 2)),
            )
    save_json(
        public / "cases.json",
        dict(
            schema="modelforge-row-acquisition-observations-v1",
            steps=[0.3, 0.6],
            cases=[dict(name="c00-fp64", n=2, outputs=1, probes=17, compute="fp64")],
        ),
    )
    view = RowView(
        public, sha(public / "cases.json"), sha(public / "arrays/manifest.jsonl")
    )
    for c in view.cases:
        view.packed(c.name)
    est = tmp_path / "estimator"
    est.mkdir()
    arrays = ShardStore(est / "arrays")
    sels = []
    for i, c in enumerate(view.cases):
        b = arrays.save(
            f"case{i:02d}-bases.npz",
            **{k + "_basis": np.eye(3) for k in ("fine", "coarse", "richardson")},
            **{
                k + "_spectrum": np.array([2.0, 1.0, 0.0])
                for k in ("fine", "coarse", "richardson")
            },
        )
        ch = arrays.save(
            f"case{i:02d}-choices.npz",
            **{
                k + "_projector": np.diag([1.0, 1.0, 0.0])
                for k in ("strict", "gap", "stable")
            },
            log_gaps=np.ones((3, 2)),
            gap_ratios=np.ones((3, 2)) * 10,
            eligible=np.ones((3, 2)),
            resolution_clipped=np.zeros((3, 2)),
            stability=np.zeros(2),
            resolution_floor=np.array([1e-12]),
            ranks=np.array([2.0, 2.0, 2.0]),
        )
        sels.append(
            dict(
                case=c.name,
                index=i,
                meta=dataclasses.asdict(c),
                choices={
                    k: dict(rank=2, certificate=False)
                    for k in ("strict", "gap", "stable")
                },
                basis=b,
                selected=ch,
            )
        )
    seal = dict(
        phase="all_public_choices_complete",
        selections=sels,
        input_commitments=view.verify_unchanged(),
        artifact_manifest_sha256=sha(est / "arrays/manifest.jsonl"),
        certificate=False,
        new_queries=0,
        new_fits=0,
    )

    def seal_again():
        save_json(est / "choices-sealed.json", seal)
        save_json(
            est / "estimator-report.json",
            dict(
                status="completed",
                selections=seal["selections"],
                seal_sha256=sha(est / "choices-sealed.json"),
                certificate=False,
                private_gate="closed",
                new_queries=0,
                new_fits=0,
                artifact_manifest_sha256=seal["artifact_manifest_sha256"],
                input_commitments=seal["input_commitments"],
            ),
        )
        return dict(
            cases_sha256=sha(public / "cases.json"),
            manifest_sha256=sha(public / "arrays/manifest.jsonl"),
            seal_sha256=sha(est / "choices-sealed.json"),
            report_sha256=sha(est / "estimator-report.json"),
        )

    pins = seal_again()
    return public, est, pins, seal, seal_again


def module():
    return importlib.import_module("variable_gate_view")


def test_sealed_rank_slices_full_basis(fixture):
    pub, est, pins, _, _ = fixture
    bundle = module()._load_fixture_bundle(pub, est, pins)
    assert len(bundle.jobs) == 1 and bundle.jobs[0].basis.shape == (2, 3)
    np.testing.assert_array_equal(bundle.jobs[0].basis, np.eye(3)[:2])
    assert bundle.jobs[0].n == 2 and bundle.jobs[0].rank == 2
    assert bundle.provenance["mapping"] == {"job00": "c00-fp64-p17"}


def test_optimizer_input_has_no_channel_metadata(fixture):
    pub, est, pins, _, _ = fixture
    job = module()._load_fixture_bundle(pub, est, pins).jobs[0]
    assert [x.name for x in dataclasses.fields(job)] == ["job_id", "basis"]
    assert job.job_id == "job00"
    assert not job.basis.flags.writeable
    with pytest.raises(ValueError):
        job.basis.setflags(write=True)


@pytest.mark.parametrize(
    "where", ["public", "unused16basis", "unused16choices", "seal", "report"]
)
def test_all_inputs_bound_before_jobs(fixture, where):
    pub, est, pins, _, _ = fixture
    path = {
        "public": pub / "arrays/c00-fp64-fine-h00.npz",
        "unused16basis": est / "arrays/case00-bases.npz",
        "unused16choices": est / "arrays/case00-choices.npz",
        "seal": est / "choices-sealed.json",
        "report": est / "estimator-report.json",
    }[where]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        module()._load_fixture_bundle(pub, est, pins)


@pytest.mark.parametrize(
    "change",
    [
        "boolrank",
        "indexbool",
        "wrongrank",
        "wrongorder",
        "missingused",
        "wrongmeta",
        "cert",
        "odd",
        "full",
        "null",
        "extra",
    ],
)
def test_inconsistent_resealed_metadata_rejected(fixture, change):
    pub, est, _, seal, again = fixture
    s = seal["selections"][1]
    if change == "boolrank":
        s["choices"]["stable"]["rank"] = True
    elif change == "indexbool":
        s["index"] = True
    elif change == "wrongrank":
        s["choices"]["stable"]["rank"] = 1
    elif change == "wrongorder":
        s["case"] = "c00-fp64-p16"
    elif change == "missingused":
        seal["input_commitments"]["used_input_sha256"].pop(
            next(iter(seal["input_commitments"]["used_input_sha256"]))
        )
    elif change == "wrongmeta":
        s["meta"]["n"] = True
    elif change == "cert":
        s["choices"]["stable"]["certificate"] = True
    elif change == "extra":
        s["width"] = 48
    else:
        s["choices"]["stable"]["rank"] = {"odd": 1, "full": 3, "null": None}[change]
    pins = again()
    with pytest.raises(ValueError):
        module()._load_fixture_bundle(pub, est, pins)


def test_registered_entry_cannot_use_fixture_domain(fixture):
    pub, est, pins, _, _ = fixture
    with pytest.raises(ValueError):
        module().load_jobs(pub, est, pins)


def test_job_copies_immutable_basis(fixture):
    m = module()
    b = np.eye(3)[:2]
    j = m.GateJob("job00", b)
    b[:] = 0
    np.testing.assert_array_equal(j.basis, np.eye(3)[:2])
    with pytest.raises(ValueError):
        m.GateJob("c00-fp64", np.eye(3)[:2])
    with pytest.raises(ValueError):
        m.GateJob("job00", np.eye(3)[:1])
    with pytest.raises(ValueError):
        m.GateJob("job00", np.eye(3))


def test_adapter_reads_only_explicit_public_roots(fixture, monkeypatch):
    pub, est, pins, _, _ = fixture
    m = module()
    original = Path.open
    seen = []

    def limited(path, *args, **kwargs):
        assert path.is_relative_to(pub) or path.is_relative_to(est), path
        seen.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", limited)
    assert len(m._load_fixture_bundle(pub, est, pins).jobs) == 1
    assert len(set(seen)) == 18  # public9+2metadata, estimator4+2metadata+manifest


def test_symlink_metadata_rejected_before_open(fixture, monkeypatch):
    pub, est, pins, _, _ = fixture
    m = module()
    path = est / "choices-sealed.json"
    target = est / "moved.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        m._load_fixture_bundle(pub, est, pins)


def test_metadata_size_gate_precedes_hash(fixture, monkeypatch):
    pub, est, pins, _, _ = fixture
    m = module()
    (est / "choices-sealed.json").write_bytes(b" " * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        m._load_fixture_bundle(pub, est, pins)


def test_late_estimator_extra_file_rejected(fixture, monkeypatch):
    pub, est, pins, _, _ = fixture
    m = module()
    original = m._pins
    calls = 0

    def inject(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            (est / "arrays/extra.npz").write_bytes(b"not registered")
        return original(*args)

    monkeypatch.setattr(m, "_pins", inject)
    with pytest.raises(ValueError, match="inventory"):
        m._load_fixture_bundle(pub, est, pins)


@pytest.mark.parametrize("n,rank", [(3, 2), (8, 32), (32, 192)])
def test_gatejob_validates_general_symmetric_dimensions(fixture, n, rank):
    p = n * (n + 1) // 2
    job = module().GateJob("job05", np.eye(p, dtype=np.float64)[:rank])
    assert job.n == n and job.rank == rank and job.basis.shape == (rank, p)


@pytest.mark.parametrize(
    "basis",
    [
        np.zeros((2, 3)),
        np.eye(3, dtype=np.float32)[:2],
        np.ones((2, 4)),
        np.zeros((0, 3)),
        np.full((2, 3), np.inf),
    ],
)
def test_gatejob_rejects_bad_geometry(fixture, basis):
    with pytest.raises(ValueError):
        module().GateJob("job00", basis)
