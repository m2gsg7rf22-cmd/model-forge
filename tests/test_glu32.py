"""Package-level kernel and strict artifact boundary tests, no sealed-label reuse."""

from pathlib import Path
import json
import zipfile
import io
import numpy as np
import pytest

from modelforge.glu32_io import (
    read_arrays,
    reconstruct_observations,
    load_candidate,
    PROFILE,
)


def test_declared_huge_npy_header_rejected_before_allocation(tmp_path):
    b = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        b, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)}
    )
    p = tmp_path / "huge.npz"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x.npy", b.getvalue())
    with pytest.raises(ValueError, match="shape"):
        read_arrays(p, {"x": (1,)})


@pytest.mark.parametrize(
    "change",
    [
        {"authorized": False},
        {"profile": "arbitrary_model"},
        {"extra": 1},
        {"schema_version": True},
    ],
)
def test_invalid_manifest_has_no_output(tmp_path, change):
    data = {
        "schema_version": 1,
        "profile": PROFILE,
        "authorized": True,
        "hessians_sha256": "a" * 64,
        "multiscale_sha256": "b" * 64,
    }
    data.update(change)
    p = tmp_path / "observations.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        reconstruct_observations(str(p), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_array_duplicates_and_nonfinite_rejected(tmp_path):
    p = tmp_path / "a.npz"
    np.savez(p, x=np.array([np.nan]))
    with pytest.raises(ValueError, match="finite"):
        read_arrays(p, {"x": (1,)})
    b = io.BytesIO()
    np.save(b, np.ones(1))
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x.npy", b.getvalue())
        with pytest.warns(UserWarning):
            z.writestr("x.npy", b.getvalue())
    with pytest.raises(ValueError, match="fields"):
        read_arrays(p, {"x": (1,)})


def test_candidate_tamper_rejected(tmp_path):
    np.savez(
        tmp_path / "student.npz",
        gates=np.ones((64, 32)),
        couplings=np.ones((64, 32, 32)),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "profile": PROFILE, "student_sha256": "0" * 64}
        )
    )
    with pytest.raises(ValueError, match="commitment"):
        load_candidate(tmp_path)


@pytest.fixture(scope="session")
def public_case(tmp_path_factory):
    import importlib.util

    root = tmp_path_factory.mktemp("glu32-public-case")
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "frozen_v4", repo / "examples/isolated_glu/sealed_v4.py"
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(572041)
    t = {
        "gate": rng.normal(size=(64, 32)) * 0.2,
        "up": rng.normal(size=(64, 32)) * 0.2,
        "down": rng.normal(size=(32, 64)) * 0.2,
    }
    held = np.zeros((4, 512, 32))  # unused public fixture input; never call verify.
    m.materialize(root / "public", root / "known-public-fixture", 1, t, held)
    m.solve(
        root / "public"
    )  # Actual frozen calibration learner, no secret truth access.
    from modelforge.glu32_io import digest, raw

    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "authorized": True,
        "hessians_sha256": digest(raw(root / "public/hessians.npz")),
        "multiscale_sha256": digest(raw(root / "public/multiscale-observations.npz")),
    }
    (root / "public/observations.json").write_text(json.dumps(manifest))
    return root, t


@pytest.fixture(scope="session")
def candidate(public_case):
    root, teacher = public_case
    report = reconstruct_observations(
        str(root / "public/observations.json"), str(root / "candidate")
    )
    return root, teacher, report


def test_actual_port_matches_frozen_learner_and_runs(candidate):
    from modelforge.glu32 import predict

    root, t, r = candidate
    g, c = load_candidate(root / "candidate")
    with np.load(root / "public/multiscale-solution.npz", allow_pickle=False) as z:
        np.testing.assert_allclose(g, z["gates"], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(c, z["couplings"], rtol=1e-9, atol=1e-10)
    x = np.random.default_rng(45728).normal(size=(128, 32))
    z = x @ t["gate"].T
    target = (z * 0.5 * (1 + np.tanh(z / 2)) * (x @ t["up"].T)) @ t["down"].T
    relative = np.linalg.norm(predict(g, c, x) - target) / np.linalg.norm(target)
    assert relative < 1e-3
    assert r["candidate_built"] is True and r["heldout_verified"] is False
    assert not r["original_weights_recovered"] and not r["numeric_certificate"]


def test_actual_service_candidate_predict_and_export(public_case, tmp_path):
    from modelforge.service import Service

    root, _ = public_case
    service = Service(tmp_path / "service")
    try:
        project = service.create_project("GLU32")["id"]
        report = service.reconstruct_glu32(
            project, str(root / "public/observations.json")
        )
        result = service.predict_glu32(project, report["artifact_id"], [[0.1] * 32])
        assert np.asarray(result["outputs"]).shape == (1, 32)
        target = tmp_path / "export.zip"
        service.export(project, str(target))
        with zipfile.ZipFile(target) as z:
            names = z.namelist()
            assert any(n.endswith("student.npz") for n in names)
            assert not any(
                "hessians.npz" in n
                or "teacher.npz" in n
                or "multiscale-observations.npz" in n
                for n in names
            )
        assert service.findings(project)[-1]["kind"] in {
            "glu32-reconstruction",
            "glu32-prediction",
        }
    finally:
        service.close()


def test_null_candidate_hash_is_never_optional(tmp_path):
    np.savez(
        tmp_path / "student.npz",
        gates=np.ones((64, 32)),
        couplings=np.ones((64, 32, 32)),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "profile": PROFILE, "student_sha256": None})
    )
    with pytest.raises(ValueError, match="manifest|commitment"):
        load_candidate(tmp_path)


@pytest.mark.parametrize("failure", ["incomplete-monitor", "timeout"])
def test_rejected_worker_artifact_is_not_exportable_or_predictable(
    tmp_path, monkeypatch, failure
):
    from pathlib import Path
    from modelforge.service import Service
    from modelforge.glu32_io import digest, raw

    service = Service(tmp_path / "workspace")
    destination = []

    def rejected(operation, arguments, **kwargs):
        p = Path(arguments["destination"])
        p.mkdir(parents=True)
        destination.append(p)
        np.savez(
            p / "student.npz", gates=np.ones((64, 32)), couplings=np.ones((64, 32, 32))
        )
        (p / "report.json").write_text("{}")
        (p / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": PROFILE,
                    "student_sha256": digest(raw(p / "student.npz")),
                    "report_sha256": digest(raw(p / "report.json")),
                }
            )
        )
        if failure == "timeout":
            raise TimeoutError("worker finished files but exceeded its budget")
        return {"process_monitor_scope": "process_only_children_unavailable"}

    try:
        project = service.create_project("Rejected artifact")["id"]
        monkeypatch.setattr(service, "_engine", rejected)
        with pytest.raises((TimeoutError, ValueError)):
            service.reconstruct_glu32(project, "unused-input")
        target = tmp_path / "export.zip"
        service.export(project, str(target))
        with zipfile.ZipFile(target) as z:
            assert not any(
                name.startswith("glu32-reconstruction/") for name in z.namelist()
            )
        with pytest.raises(ValueError, match="accepted"):
            service.predict_glu32(project, destination[0].name, [[0.0] * 32])
    finally:
        service.close()


@pytest.mark.parametrize("changed", ["student", "report"])
def test_success_record_binds_artifact_even_if_manifest_rehashed(tmp_path, changed):
    from modelforge.service import Service
    from modelforge.glu32_io import digest, raw

    service = Service(tmp_path / "workspace")
    try:
        project = service.create_project("Commitment")["id"]
        aid = "a" * 32
        p = service.root / project / "glu32-reconstruction" / aid
        p.mkdir(parents=True)
        np.savez(
            p / "student.npz", gates=np.ones((64, 32)), couplings=np.ones((64, 32, 32))
        )
        (p / "report.json").write_text("{}")
        manifest = {
            "schema_version": 1,
            "profile": PROFILE,
            "student_sha256": digest(raw(p / "student.npz")),
            "report_sha256": digest(raw(p / "report.json")),
        }
        (p / "manifest.json").write_text(json.dumps(manifest))
        service._record(
            project,
            "glu32-reconstruction",
            {
                "artifact_id": aid,
                "candidate_built": True,
                "process_monitor_scope": "process_and_children",
                "student_sha256": manifest["student_sha256"],
                "manifest_sha256": digest(raw(p / "manifest.json")),
            },
        )
        if changed == "student":
            np.savez(
                p / "student.npz",
                gates=np.ones((64, 32)) * 2,
                couplings=np.ones((64, 32, 32)),
            )
            manifest["student_sha256"] = digest(raw(p / "student.npz"))
        else:
            (p / "report.json").write_text('{"changed":true}')
            manifest["report_sha256"] = digest(raw(p / "report.json"))
        (p / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="commitment"):
            service.export(project, str(tmp_path / "export.zip"))
        with pytest.raises(ValueError, match="commitment"):
            service.predict_glu32(project, aid, [[0.0] * 32])
    finally:
        service.close()


def test_duplicate_authorization_key_rejected_before_any_array(tmp_path):
    source = tmp_path / "observations.json"
    source.write_text('{"authorized":false,"authorized":true}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        reconstruct_observations(str(source), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_no_curvature_rejects_without_candidate():
    from modelforge.glu32 import directions

    with pytest.raises(ValueError, match="rank|spectral"):
        directions(np.zeros((64, 32, 32, 32)))
