"""Sealed public observations -> opaque development jobs; no truth or RNG.

Distinct adapter, not an experiment launcher or reconstruction certificate.
The private fixture seam checks the same seal contract on small literal data.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any
import numpy as np
from numpy.typing import NDArray
from fresh_rank_common import canonical, read_arrays, read_json, records
from modelforge.glu32_io import raw
from row_acquisition_view import RowView, registered_view, VARIANTS

Array = NDArray[np.float64]
RULES = ("strict", "gap", "stable")
PINS = {"cases_sha256", "manifest_sha256", "seal_sha256", "report_sha256"}


def _digest(path: Path, maximum: int = 8 * 1024**2) -> str:
    return hashlib.sha256(raw(path, maximum)).hexdigest()


def _same(a: Any, b: Any) -> bool:
    return canonical(a) == canonical(b)


def _orthogonal(b: Array, p: int) -> None:
    if (
        not isinstance(b, np.ndarray)
        or b.dtype != np.float64
        or b.ndim != 2
        or b.shape[1] != p
        or not 1 <= len(b) <= p
        or not np.isfinite(b).all()
        or not np.allclose(
            b @ b.T,
            np.eye(len(b)),
            rtol=0,
            atol=64 * np.finfo(float).eps * max(b.shape),
        )
    ):
        raise ValueError("Finite orthonormal rows required")


@dataclass(frozen=True, slots=True)
class GateJob:
    job_id: str
    basis: Array

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not re.fullmatch(
            r"job0[0-5]", self.job_id
        ):
            raise ValueError("Opaque job00..05 required")
        b = self.basis
        if not isinstance(b, np.ndarray) or b.ndim != 2 or b.dtype != np.float64:
            raise ValueError("Native FP64 row basis required")
        p = b.shape[1]
        n = (math.isqrt(8 * p + 1) - 1) // 2
        if (
            not 2 <= n <= 32
            or n * (n + 1) // 2 != p
            or not 0 < len(b) < p
            or len(b) % 2
            or len(b) > 192
        ):
            raise ValueError(
                "Proper even rank<=192 in bounded symmetric coordinates required"
            )
        _orthogonal(b, p)
        # A bytes-backed copy cannot be made writable by setflags or mutate a caller.
        object.__setattr__(
            self, "basis", np.frombuffer(b.tobytes(), dtype=np.float64).reshape(b.shape)
        )

    @property
    def n(self) -> int:
        return (math.isqrt(8 * self.basis.shape[1] + 1) - 1) // 2

    @property
    def rank(self) -> int:
        return len(self.basis)


@dataclass(frozen=True, slots=True)
class GateBundle:
    jobs: tuple[GateJob, ...]
    provenance: dict[str, Any]


def _pins(public: Path, est: Path, commitments: dict[str, str]) -> None:
    if (
        not isinstance(commitments, dict)
        or set(commitments) != PINS
        or any(
            not isinstance(x, str) or re.fullmatch("[0-9a-f]{64}", x) is None
            for x in commitments.values()
        )
    ):
        raise ValueError("Exact explicit input commitments required")
    for key, path in [
        ("cases_sha256", public / "cases.json"),
        ("manifest_sha256", public / "arrays/manifest.jsonl"),
        ("seal_sha256", est / "choices-sealed.json"),
        ("report_sha256", est / "estimator-report.json"),
    ]:
        # Bound and reject symlink ancestors before hashing metadata.
        if _digest(path, 128 * 1024) != commitments[key]:
            raise ValueError("Input metadata commitment mismatch")


def _schemas(
    q: int, p: int
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    count = min(q, p - 1)
    bases: dict[str, tuple[int, ...]] = {k + "_basis": (q, p) for k in VARIANTS}
    bases.update({k + "_spectrum": (p,) for k in VARIANTS})
    chosen: dict[str, tuple[int, ...]] = {k + "_projector": (p, p) for k in RULES}
    chosen.update(
        log_gaps=(3, count),
        gap_ratios=(3, count),
        eligible=(3, count),
        resolution_clipped=(3, count),
        stability=(count,),
        resolution_floor=(1,),
        ranks=(3,),
    )
    return bases, chosen


def _load(
    public: Path, est: Path, commitments: dict[str, str], *, fixture: bool
) -> GateBundle:
    _pins(public, est, commitments)
    view = RowView(public, commitments["cases_sha256"], commitments["manifest_sha256"])
    if not fixture:
        registered_view(view)
    if {p.name for p in est.iterdir()} != {
        "arrays",
        "choices-sealed.json",
        "estimator-report.json",
    }:
        raise ValueError("Exact estimator directory required")
    seal = read_json(est / "choices-sealed.json")
    report = read_json(est / "estimator-report.json")
    if (
        set(seal)
        != {
            "phase",
            "selections",
            "input_commitments",
            "artifact_manifest_sha256",
            "certificate",
            "new_queries",
            "new_fits",
        }
        or seal["phase"] != "all_public_choices_complete"
        or seal["certificate"] is not False
        or any(
            type(seal[k]) is not int or seal[k] != 0
            for k in ("new_queries", "new_fits")
        )
        or report.get("status") != "completed"
        or report.get("certificate") is not False
        or report.get("private_gate") != "closed"
        or any(
            type(report.get(k)) is not int or report[k] != 0
            for k in ("new_queries", "new_fits")
        )
        or report.get("seal_sha256") != commitments["seal_sha256"]
        or not _same(report.get("selections"), seal["selections"])
        or not _same(report.get("input_commitments"), seal["input_commitments"])
        or report.get("artifact_manifest_sha256") != seal["artifact_manifest_sha256"]
        or not isinstance(seal["selections"], list)
        or len(seal["selections"]) != len(view.cases)
    ):
        raise ValueError("Complete consistent public seal and report required")
    # Validate every physical public archive schema/hash, including unused prefixes.
    for case in view.base_cases:
        for kind in VARIANTS:
            for start in range(0, 17, 8):
                name = f"{case.name}-{kind}-h{start // 8:02d}.npz"
                record = view.records[name]
                read_arrays(
                    view.arrays / name,
                    {"hessian": (min(8, 17 - start), case.outputs, case.n, case.n)},
                    record["sha256"],
                )
                view.used[name] = record["sha256"]
    expected_inputs = view.verify_unchanged()
    if not _same(seal["input_commitments"], expected_inputs):
        raise ValueError("Complete public source binding required")
    inv = records(est / "arrays")
    expected = {
        f"case{i:02d}-{kind}.npz"
        for i in range(len(view.cases))
        for kind in ("bases", "choices")
    }
    manifest_sha = _digest(est / "arrays/manifest.jsonl", 128 * 1024)
    if (
        set(inv) != expected
        or manifest_sha != seal["artifact_manifest_sha256"]
        or {p.name for p in (est / "arrays").iterdir()} != expected | {"manifest.jsonl"}
    ):
        raise ValueError("Exact sealed estimator inventory required")
    pending: list[tuple[str, Array]] = []
    archive_hashes: dict[str, str] = {}
    for i, case in enumerate(view.cases):
        selection = seal["selections"][i]
        if (
            not isinstance(selection, dict)
            or set(selection)
            != {"case", "index", "meta", "choices", "basis", "selected"}
            or selection["case"] != case.name
            or type(selection["index"]) is not int
            or selection["index"] != i
            or not _same(selection["meta"], asdict(case))
            or not isinstance(selection["choices"], dict)
            or set(selection["choices"]) != set(RULES)
        ):
            raise ValueError("Exact ordered selection metadata required")
        p = case.n * (case.n + 1) // 2
        q = min(case.probes * case.outputs, p)
        fields, chosen = _schemas(q, p)
        loaded = []
        for key, kind, schema in (
            ("basis", "bases", fields),
            ("selected", "choices", chosen),
        ):
            name = f"case{i:02d}-{kind}.npz"
            record = inv[name]
            if (
                not _same(selection[key], record)
                or (est / "arrays" / name).stat().st_size != record["bytes"]
            ):
                raise ValueError("Archive descriptor or size mismatch")
            loaded.append(read_arrays(est / "arrays" / name, schema, record["sha256"]))
            archive_hashes[name] = record["sha256"]
        bases, saved = loaded
        for kind in VARIANTS:
            _orthogonal(bases[kind + "_basis"], p)
        right = bases["richardson_basis"]
        for j, rule in enumerate(RULES):
            choice = selection["choices"][rule]
            if not isinstance(choice, dict):
                raise ValueError("Choice object required")
            k = choice.get("rank")
            if (
                "rank" not in choice
                or choice.get("certificate") is not False
                or (k is None and rule == "strict")
                or (k is not None and (type(k) is not int or not 0 <= k <= q))
                or saved["ranks"][j] != (-1 if k is None else k)
            ):
                raise ValueError("Native sealed rank required")
            projector = np.zeros((p, p)) if k is None else right[:k].T @ right[:k]
            if not np.allclose(
                saved[rule + "_projector"],
                projector,
                rtol=0,
                atol=64 * np.finfo(float).eps * max(q, p),
            ):
                raise ValueError("Rank/projector mismatch")
        if case.probes == 17:
            k = selection["choices"]["stable"]["rank"]
            if type(k) is not int or not 0 < k < p or k % 2 or k > 192:
                raise ValueError(
                    "Stable rank outside candidate development domain; abstain"
                )
            pending.append((case.name, right[:k].copy()))
    # Close input mutation windows before constructing any optimizer job.
    _pins(public, est, commitments)
    view.verify_unchanged()
    if {x.name for x in est.iterdir()} != {
        "arrays",
        "choices-sealed.json",
        "estimator-report.json",
    } or {x.name for x in (est / "arrays").iterdir()} != expected | {"manifest.jsonl"}:
        raise ValueError("Estimator inventory changed")
    if _digest(est / "arrays/manifest.jsonl", 128 * 1024) != manifest_sha:
        raise ValueError("Estimator manifest changed")
    for name, h in archive_hashes.items():
        if _digest(est / "arrays" / name) != h:
            raise ValueError("Estimator archive changed")
    jobs = tuple(GateJob(f"job{i:02d}", basis) for i, (_, basis) in enumerate(pending))
    return GateBundle(
        jobs,
        dict(
            mapping={job.job_id: pending[i][0] for i, job in enumerate(jobs)},
            commitments=dict(commitments),
            estimator_sha256=archive_hashes,
            public_inputs=expected_inputs,
            certificate=False,
            development_only=True,
        ),
    )


def load_bundle(
    public_root: Path, estimator_root: Path, commitments: dict[str, str]
) -> GateBundle:
    """Registered six-job adapter; never execute fitting or touch sibling truth."""
    return _load(public_root, estimator_root, commitments, fixture=False)


def load_jobs(
    public_root: Path, estimator_root: Path, commitments: dict[str, str]
) -> tuple[GateJob, ...]:
    return load_bundle(public_root, estimator_root, commitments).jobs


def _load_fixture_bundle(
    public_root: Path, estimator_root: Path, commitments: dict[str, str]
) -> GateBundle:
    """Small literal fixture seam; never called by the registered controller."""
    return _load(public_root, estimator_root, commitments, fixture=True)
