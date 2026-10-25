"""Read-only normal geometry diagnostic; no target access, fitting or solves."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw, read_json, read_arrays

Array = NDArray[np.float64]
METHOD = "isolated-glu-fp32-guard-normal-geometry-v1"
CODES = {
    "empty": 1,
    "independent_within_domain": 2,
    "exact_dependent": 3,
    "near_independent_outside_domain": 4,
    "numerical_failure": 5,
}


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "frozen_" + name, Path(__file__).with_name(name + ".py")
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


metrics = load("guard_metric_normals")
cone = load("guard_aware_direction")
transport = load("evidence_transport")


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def inspect_geometry(a: Array) -> dict[str, Any]:
    if (
        a.ndim != 2
        or not 1 <= a.shape[1] <= 32
        or len(a) > 8
        or a.dtype != np.float64
        or not np.isfinite(a).all()
    ):
        raise ValueError("Bounded finite FP64 normals required")
    lengths = np.linalg.norm(a, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("Finite nonzero row norms required")
    b = a / lengths[:, None]
    records = []
    outside = 0
    failures = 0
    for mask in range(1 << len(a)):
        selected = [i for i in range(len(a)) if mask & (1 << i)]
        record: dict[str, Any] = {
            "mask": mask,
            "active": selected,
            "singular_values": [],
            "ratio": None,
            "exact_original_rank": None,
        }
        if not selected:
            record["status"] = "empty"
        else:
            try:
                singular = np.linalg.svd(b[selected], compute_uv=False)
                if not np.isfinite(singular).all() or singular[0] <= 0:
                    raise ValueError("Invalid subset spectrum")
                ratio = float(singular[-1] / singular[0])
                record.update(singular_values=singular.tolist(), ratio=ratio)
                if len(selected) > a.shape[1]:
                    record.update(
                        status="exact_dependent", reason="more_rows_than_dimensions"
                    )
                elif ratio <= cone.GEOMETRY_TOL:
                    rank = cone.exact_row_rank(a[selected])
                    record["exact_original_rank"] = rank
                    if rank == len(selected):
                        record["status"] = "near_independent_outside_domain"
                        outside += 1
                    else:
                        record.update(
                            status="exact_dependent",
                            reason="exact_original_binary64_dependency",
                        )
                else:
                    record["status"] = "independent_within_domain"
            except (ValueError, np.linalg.LinAlgError) as exc:
                record.update(status="numerical_failure", error=str(exc))
                failures += 1
        records.append(record)
    return {
        "subsets": records,
        "in_domain": outside == 0 and failures == 0,
        "outside_subsets": outside,
        "numerical_failures": failures,
        "empty_unconstrained": len(a) == 0,
        "row_norms": lengths.tolist(),
        "geometry_ratio_threshold": cone.GEOMETRY_TOL,
    }


def collect(
    g: Array, c: Array, x: Array, y: Array, rows: int
) -> tuple[dict[str, Array], dict[str, Any]]:
    prediction = metrics.gradient.forward(g, c, x)
    if (
        not metrics.integer(rows, 1, 1024)
        or len(x) != 4 * rows
        or y.shape != prediction.shape
        or y.dtype != np.float64
        or not np.isfinite(y).all()
    ):
        raise ValueError("Complete four-group physical guard profile required")
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction - y
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite complete physical guard error")
    width, dimension = g.shape
    arrays = {
        "guard_x": x,
        "guard_y": y,
        "prediction": prediction,
        "physical_error": error,
        "normals": np.zeros((width, 32, dimension)),
        "normal_counts": np.zeros(width),
        "mapping_valid": np.zeros(width),
        "subset_counts": np.zeros(width),
        "subset_singular_values": np.zeros((width, 256, 8)),
        "subset_spectrum_lengths": np.zeros((width, 256)),
        "subset_status_codes": np.zeros((width, 256)),
    }
    profile = []
    for group in range(4):
        block = error[group * rows : (group + 1) * rows]
        maximum = float(np.max(np.abs(block)))
        threshold = maximum * (1 - metrics.NEAR_MAX_RELATIVE_TOLERANCE)
        near = (
            []
            if maximum == 0
            else [
                {
                    "row": group * rows + int(i),
                    "output": int(j),
                    "signed_residual": float(block[i, j]),
                    "exact_maximum": abs(float(block[i, j])) == maximum,
                }
                for i, j in np.argwhere(np.abs(block) >= threshold)
            ]
        )
        profile.append(
            {
                "group": group,
                "maximum_absolute": maximum,
                "near_threshold": threshold,
                "zero_max_all_coordinates": maximum == 0,
                "near_coordinates": near,
            }
        )
    counts = {
        name: 0
        for name in [
            "units",
            "mapping_succeeded",
            "mapping_failed",
            "geometry_in_domain",
            "geometry_outside_domain",
            "geometry_failed",
            "empty_geometry",
        ]
    }
    counts["units"] = width
    units = []
    for unit in range(width):
        records: list[dict[str, Any]] = []
        group_reports = []
        matrices = []
        matrix_offset = 0
        errors = []
        for group in range(4):
            block = slice(group * rows, (group + 1) * rows)
            try:
                a, info = metrics.metric_group(
                    g[unit],
                    c[unit],
                    x[block],
                    y[block],
                    error[block],
                    group,
                    group * rows,
                )
                for original in info["records"]:
                    record = dict(original)
                    record["logical_index"] = len(records)
                    if record["matrix_row"] is not None:
                        record["matrix_row"] += matrix_offset
                    records.append(record)
                matrix_offset += len(a)
                matrices.append(a)
                group_reports.append({k: v for k, v in info.items() if k != "records"})
            except ValueError as exc:
                errors.append(
                    {
                        "group": group,
                        "error": str(exc),
                        "raw_provenance": "guard_profile and physical_error preserved; no complete normals for failed group",
                    }
                )
        a = np.vstack(matrices) if matrices else np.empty((0, dimension))
        if len(a) > 32:
            raise ValueError("Internal diagnostic storage bound exceeded")
        arrays["normals"][unit, : len(a)] = a
        arrays["normal_counts"][unit] = len(a)
        # Failed groups still have complete raw error/tie provenance. The
        # logical bound counts prospective records before zero omission.
        logical_count = (
            None
            if any(p["zero_max_all_coordinates"] for p in profile)
            else sum(1 + len(p["near_coordinates"]) for p in profile)
        )
        if logical_count is not None and logical_count > 8:
            errors.append(
                {"error": "Full profile logical plane budget exceeded; no truncation"}
            )
        info = {
            "unit": unit,
            "metrics": {
                "records": records,
                "groups": group_reports,
                "logical_plane_count": logical_count,
                "computed_record_count": len(records),
            },
            "mapping_status": "failed" if errors else "passed",
            "mapping_errors": errors,
        }
        if errors:
            counts["mapping_failed"] += 1
        else:
            counts["mapping_succeeded"] += 1
            arrays["mapping_valid"][unit] = 1
            geometry = inspect_geometry(a)
            info["geometry"] = geometry
            arrays["subset_counts"][unit] = len(geometry["subsets"])
            for item in geometry["subsets"]:
                mask = item["mask"]
                spectrum = item["singular_values"]
                arrays["subset_singular_values"][unit, mask, : len(spectrum)] = spectrum
                arrays["subset_spectrum_lengths"][unit, mask] = len(spectrum)
                arrays["subset_status_codes"][unit, mask] = CODES[item["status"]]
            if geometry["numerical_failures"]:
                counts["geometry_failed"] += 1
            elif not geometry["in_domain"]:
                counts["geometry_outside_domain"] += 1
            else:
                counts["geometry_in_domain"] += 1
            if geometry["empty_unconstrained"]:
                counts["empty_geometry"] += 1
        units.append(info)
    return arrays, {
        "method": METHOD,
        "counts": counts,
        "units": units,
        "guard_profile": profile,
        "subset_status_codes": CODES,
        "direction_solves": 0,
        "new_target_queries": 0,
        "candidate_updates": 0,
        "guard_is_adaptive_calibration": True,
        "private_gate": "closed",
    }


def reserve(output: Path) -> None:
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("No symlink output ancestors")
    output.mkdir(exist_ok=False)
    (output / "attempt.json").write_text(
        json.dumps(
            {
                "automatic_retry": False,
                "direction_solves": 0,
                "target_queries": 0,
                "candidate_updates": 0,
            }
        )
        + "\n"
    )


def persist(
    output: Path,
    arrays: dict[str, Array],
    details: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    encoded = (json.dumps(details, indent=2, allow_nan=False) + "\n").encode()
    metadata = {
        "method": METHOD,
        "details_file": "details.json",
        "details_sha256": hashlib.sha256(encoded).hexdigest(),
        "details_bytes": len(encoded),
        "counts": details["counts"],
        "inputs": inputs,
        "direction_solves": 0,
        "target_queries": 0,
        "candidate_updates": 0,
        "private_gate": "closed",
    }
    summary = transport.save_evidence(output / "evidence", arrays, metadata)
    (output / "evidence/details.json").write_bytes(encoded)
    return {
        "method": METHOD,
        "units": details["counts"]["units"],
        "counts": details["counts"],
        "subsets": int(np.sum(arrays["subset_counts"])),
        "details_bytes": len(encoded),
        "details_sha256": metadata["details_sha256"],
        "numeric_sha256": summary["evidence_sha256"],
        "summary_sha256": sha(output / "evidence/summary.json"),
        "direction_solves": 0,
        "target_queries": 0,
        "candidate_updates": 0,
        "private_gate": "closed",
    }


def synthetic_transport(output: Path) -> dict[str, Any]:
    reserve(output)
    geometry = inspect_geometry(np.eye(8, 32))
    normals = np.zeros((64, 32, 32))
    normals[:, :8] = np.eye(8, 32)
    spectra = np.zeros((64, 256, 8))
    lengths = np.zeros((64, 256))
    codes = np.zeros((64, 256))
    for item in geometry["subsets"]:
        mask = item["mask"]
        s = item["singular_values"]
        spectra[:, mask, : len(s)] = s
        lengths[:, mask] = len(s)
        codes[:, mask] = CODES[item["status"]]
    records = [
        {
            "logical_index": i,
            "matrix_row": i,
            "metric": "synthetic_plane",
            "normal": np.eye(8, 32)[i].tolist(),
            "group": i // 2,
            "row": i,
            "output": 0,
            "exact_maximum": True,
            "omitted_reason": None,
        }
        for i in range(8)
    ]
    units = [
        {
            "unit": i,
            "mapping_status": "passed",
            "metrics": {"records": records},
            "geometry": geometry,
        }
        for i in range(64)
    ]
    counts = {
        "units": 64,
        "mapping_succeeded": 64,
        "mapping_failed": 0,
        "geometry_in_domain": 64,
        "geometry_outside_domain": 0,
        "geometry_failed": 0,
        "empty_geometry": 0,
    }
    arrays = {
        "normals": normals,
        "normal_counts": np.full(64, 8.0),
        "mapping_valid": np.ones(64),
        "subset_counts": np.full(64, 256.0),
        "subset_singular_values": spectra,
        "subset_spectrum_lengths": lengths,
        "subset_status_codes": codes,
        "prediction": np.zeros((1024, 32)),
        "physical_error": np.zeros((1024, 32)),
    }
    return persist(
        output,
        arrays,
        {
            "synthetic_transport_only": True,
            "units": units,
            "counts": counts,
            "direction_solves": 0,
        },
        {},
    )


def build(
    paths: dict[str, Path], output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    reserve(output)
    names = {
        "candidate",
        "compact_metadata",
        "full_fit_report",
        "observations",
        "manifest",
        "original_manifest",
    }
    if set(paths) != names or set(config.get("inputs", {})) != names:
        raise ValueError("Exact diagnostic input inventory required")
    before = {k: sha(p) for k, p in paths.items()}
    if before != config["inputs"]:
        raise ValueError("Diagnostic input commitments changed")
    compact = read_json(paths["compact_metadata"])
    meta = read_json(paths["manifest"])
    original = read_json(paths["original_manifest"])
    if (
        compact.get("method") != "isolated-glu-fp32-guarded-gate-step-v1"
        or compact.get("candidate_built") is not True
        or compact.get("candidate_sha256") != before["candidate"]
        or compact.get("forward_raw_sha256") != before["observations"]
        or compact.get("forward_manifest_sha256") != before["manifest"]
        or compact.get("calibration_manifest_sha256") != before["original_manifest"]
        or compact.get("full_fit_report_sha256") != before["full_fit_report"]
        or compact.get("full_fit_details_preserved_externally") is not True
        or original.get("channel") != "fp32_compute"
    ):
        raise ValueError("Saved candidate/raw calibration provenance mismatch")
    expected = {
        "method": "isolated-glu-fp32-guarded-forward-polish-v1",
        "channel": "fp32_compute",
        "train_seed": 882030,
        "guard_seed": 882031,
        "scales": [0.01, 1.0, 10.0, 100.0],
        "group_rows": 256,
        "vector_calls": 2048,
        "baseline_sha256": "b2d5fdd4cf8be40954562c2ce9393cd5b9a3c3cbd3cd8cbfa9bc18728d1de41e",
        "raw_sha256": before["observations"],
    }
    if meta != expected:
        raise ValueError("Guard calibration manifest changed")
    observed = read_arrays(
        paths["observations"],
        {name: (1024, 32) for name in ["train_x", "train_y", "guard_x", "guard_y"]},
        before["observations"],
    )
    rng = np.random.default_rng(882031)
    expected_x = np.vstack(
        [rng.normal(size=(256, 32)) * s for s in [0.01, 1.0, 10.0, 100.0]]
    )
    if not np.array_equal(expected_x, observed["guard_x"]):
        raise ValueError("Guard input seed mismatch")
    candidate = read_arrays(
        paths["candidate"],
        {"gates": (64, 32), "couplings": (64, 32, 32)},
        before["candidate"],
    )
    arrays, details = collect(
        candidate["gates"],
        candidate["couplings"],
        observed["guard_x"],
        observed["guard_y"],
        256,
    )
    after = {k: sha(p) for k, p in paths.items()}
    if before != after:
        raise ValueError("Read-only diagnostic inputs changed")
    details.update(
        inputs_before=before,
        inputs_after=after,
        inputs_unchanged=True,
        original_full_fit_report_read_mode="hash_only_not_metadata_parser",
        reused_guard_vectors=1024,
        new_calibration_calls=0,
        new_evaluation_calls=0,
    )
    return persist(output, arrays, details, before)


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(
        json.dumps(
            build(
                {k: Path(v) for k, v in req["paths"].items()},
                Path(req["output"]),
                req["config"],
            ),
            allow_nan=False,
        )
    )
