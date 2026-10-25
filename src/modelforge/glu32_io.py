"""Strict authorized observations and self-contained candidate artifacts."""

from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
import zipfile
import numpy as np
from . import glu32

PROFILE = "isolated-swiglu32x64-fp64-v4"
MAX_BYTES = 128 * 1024**2
SHAPES: dict[str, tuple[int, ...]] = {
    "small_x": (4096, 32),
    "small_odd": (4096, 32),
    "large_x": (512, 32),
    "large_odd": (512, 32),
    "even_pos": (528, 32),
    "even_neg": (528, 32),
}


def raw(path: Path, maximum: int = MAX_BYTES) -> bytes:
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > maximum
    ):
        raise ValueError("Requires bounded regular artifact without symlink ancestors")
    with path.open("rb") as f:
        result = f.read(maximum + 1)
    if len(result) > maximum:
        raise ValueError("Artifact exceeds byte budget")
    return result


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    r = json.loads(raw(path, 2_000_000), object_pairs_hook=unique)
    json.dumps(r, allow_nan=False)
    if not isinstance(r, dict):
        raise ValueError("Expected JSON object")
    return r


def read_arrays(
    path: Path, shapes: dict[str, tuple[int, ...]], expected_hash: str | None = None
) -> dict[str, glu32.Array]:
    data = raw(path)
    if expected_hash is not None and digest(data) != expected_hash:
        raise ValueError("Array commitment mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        expected = {k + ".npy" for k in shapes}
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Unexpected array fields or duplicates")
        if sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
            raise ValueError("Array allocation budget exceeded")
        for key, shape in shapes.items():
            with archive.open(key + ".npy") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    declared, _, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    declared, _, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError("Unsupported NPY header version")
                if declared != shape or dtype != np.dtype(np.float64):
                    raise ValueError("Invalid finite float64 array shape/dtype")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        result = {k: z[k] for k in shapes}
    if any(not np.isfinite(a).all() for a in result.values()):
        raise ValueError("Nonfinite observation array")
    return result


def reconstruct_observations(
    observations_path: str, destination: str
) -> dict[str, Any]:
    source, target = Path(observations_path), Path(destination)
    meta = read_json(source)
    fields = {
        "schema_version",
        "profile",
        "authorized",
        "hessians_sha256",
        "multiscale_sha256",
    }
    if (
        set(meta) != fields
        or type(meta["schema_version"]) is not int
        or meta["schema_version"] != 1
        or meta["profile"] != PROFILE
        or meta["authorized"] is not True
    ):
        raise ValueError("Requires authorized exact GLU32 observation profile/schema")
    for k in ("hessians_sha256", "multiscale_sha256"):
        h = meta[k]
        if (
            not isinstance(h, str)
            or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h)
        ):
            raise ValueError("Invalid observation commitment")
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Destination symlink forbidden")
    if target.exists():
        raise FileExistsError("Candidate destination already exists")
    h = read_arrays(
        source.parent / "hessians.npz",
        {"hessian": (64, 32, 32, 32)},
        meta["hessians_sha256"],
    )["hessian"]
    obs = read_arrays(
        source.parent / "multiscale-observations.npz", SHAPES, meta["multiscale_sha256"]
    )
    started = time.perf_counter()
    gates, couplings, diagnostics = glu32.fit(h, obs)
    singular = np.linalg.svd(couplings, compute_uv=False)
    if (
        not np.isfinite(gates).all()
        or not np.isfinite(couplings).all()
        or np.any(np.linalg.norm(gates, axis=1) <= 0)
        or np.any(singular[:, 0] <= 0)
        or np.any(singular[:, 1] > 1e-10 * singular[:, 0])
    ):
        raise ValueError("Invalid candidate before publication")
    target.mkdir(parents=True, exist_ok=False)
    np.savez(target / "student.npz", gates=gates, couplings=couplings)
    report = {
        "profile": PROFILE,
        "candidate_built": True,
        "diagnostics": diagnostics,
        "scope": "Candidate for declared isolated bias-free FP64 SwiGLU32/64/32; input metadata is not a certificate",
        "assumptions_verified": False,
        "original_weights_recovered": False,
        "all_inputs_equivalence_proven": False,
        "heldout_verified": False,
        "numeric_certificate": False,
        "remote_calls": 0,
        "input_commitments": meta,
        "seconds": time.perf_counter() - started,
        "source_sha256": {
            p.name: digest(raw(p))
            for p in (
                Path(__file__),
                Path(glu32.__file__),
                Path(__file__).with_name("antipodal.py"),
            )
        },
    }
    (target / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "student_sha256": digest(raw(target / "student.npz")),
        "report_sha256": digest(raw(target / "report.json")),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    load_candidate(target)
    return {
        **report,
        "artifact_directory": str(target),
        "student_sha256": manifest["student_sha256"],
        "manifest_sha256": digest(raw(target / "manifest.json")),
    }


def load_candidate(directory: Path) -> tuple[glu32.Array, glu32.Array]:
    manifest = read_json(directory / "manifest.json")
    if (
        set(manifest)
        != {"schema_version", "profile", "student_sha256", "report_sha256"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["profile"] != PROFILE
    ):
        raise ValueError("Invalid candidate manifest commitment schema")
    for key in ("student_sha256", "report_sha256"):
        value = manifest[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid candidate manifest commitment")
    arrays = read_arrays(
        directory / "student.npz",
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        manifest.get("student_sha256", ""),
    )
    if (
        "report_sha256" in manifest
        and digest(raw(directory / "report.json")) != manifest["report_sha256"]
    ):
        raise ValueError("Candidate report commitment mismatch")
    g, c = arrays["gates"], arrays["couplings"]
    s = np.linalg.svd(c, compute_uv=False)
    if (
        np.any(np.linalg.norm(g, axis=1) <= 0)
        or np.any(s[:, 0] <= 0)
        or np.any(s[:, 1] > 1e-10 * s[:, 0])
    ):
        raise ValueError("Invalid zero gates or non-rank-one couplings")
    return g, c


def predict_candidate(
    artifact_directory: str, inputs: list[list[float]], manifest_sha256: str
) -> dict[str, Any]:
    require_commitment(Path(artifact_directory), manifest_sha256)
    g, c = load_candidate(Path(artifact_directory))
    return {
        "outputs": glu32.predict(g, c, inputs).tolist(),
        "profile": PROFILE,
        "reference_verified": False,
    }


def export_files(directory: Path, accepted_ids: dict[str, str]) -> list[Path]:
    result: list[Path] = []
    if directory.is_symlink():
        raise ValueError("Candidate export directory symlink forbidden")
    for artifact_id in sorted(accepted_ids):
        if len(artifact_id) != 32 or any(
            c not in "0123456789abcdef" for c in artifact_id
        ):
            raise ValueError("Invalid accepted candidate id")
        candidate = directory / artifact_id
        require_commitment(candidate, accepted_ids[artifact_id])
        load_candidate(candidate)
        result.extend(
            candidate / name for name in ("student.npz", "manifest.json", "report.json")
        )
    return result


def require_commitment(directory: Path, expected: str) -> None:
    if digest(raw(directory / "manifest.json", 2_000_000)) != expected:
        raise ValueError("Accepted candidate commitment changed")
