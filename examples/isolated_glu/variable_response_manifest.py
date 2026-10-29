"""Pinned public receipts -> exact chosen-input/response whitelist, no raw data.

Only the parent provenance layer uses this module. No directory inventory, RNG,
teacher, evaluator, or fitting entry point. Production pins are immutable here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Callable

ENCODED_CAP = 544768
HEADER_CAP = 4096
COMMITMENTS_CAP = 2 * 1024**2
METADATA_CAP = 128 * 1024
SOURCE = "07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a"
ORIGINAL = "/Users/adamfaruhi123/.local/share/modelforge-research/experiments/paired-row-acquisition-v1-20261028-07235ac/collection"
BASE = "experiments/evidence/paired-row-acquisition-v1/"
PUBLIC_PINS = (
    (
        "audit",
        BASE + "outcome-audit/outcome-audit.json",
        "363e46db8c437b40e435ef9d3873e6f834d5a877fb688136d024d7d05aa553c7",
    ),
    (
        "commitments",
        BASE + "outcome-audit/artifact-commitments.json",
        "6024e60e35c9feaac11277f488325abca85909cedc0c1cc43b6fea45e6f9f23d",
    ),
    (
        "collection",
        BASE + "actual/collection/collection-report.json",
        "eba75ce5156e0ee13ca1e5cdac060bcb6d9028389b7d6d26a8cff2deeeb35167",
    ),
    (
        "progress",
        BASE + "actual/collection/query-progress.json",
        "b6d2af933eab4941889b969967d32c7ef570f4a6135942bf0e0315ebf7784946",
    ),
)


def _hash(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _safe(path: Path) -> None:
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink or aliased ancestry forbidden")
    st = path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise ValueError("Single-link regular file required")


def _read(path: Path, cap: int) -> bytes:
    _safe(path)
    if path.stat().st_size > cap:
        raise ValueError("File exceeds bounded cap before read")
    with path.open("rb") as f:
        blob = f.read(cap + 1)
    if len(blob) > cap:
        raise ValueError("File exceeds bounded cap during read")
    return blob


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(blob: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            blob,
            object_pairs_hook=_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded public JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Public JSON object required")
    return value


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class _Profile:
    n: int
    outputs: int
    probes: int
    case_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            any(type(x) is not int for x in (self.n, self.outputs, self.probes))
            or not 1 <= self.n <= 32
            or not 1 <= self.outputs <= 32
            or not 1 <= self.probes <= 17
            or not isinstance(self.case_names, tuple)
            or not 1 <= len(self.case_names) <= 6
            or self.case_names
            != tuple(
                f"c{i // 2:02d}-" + ("fp64" if i % 2 == 0 else "fp32")
                for i in range(len(self.case_names))
            )
        ):
            raise ValueError("Bounded canonical literal profile required")

    @property
    def rows(self) -> int:
        return 2 * self.n * (self.n + 1)

    @property
    def vector_rows(self) -> int:
        return len(self.case_names) * 2 * self.probes * self.rows


REGISTERED = _Profile(
    32, 32, 17, tuple(f"c{i:02d}-{c}" for i in range(3) for c in ("fp64", "fp32"))
)


@dataclass(frozen=True, slots=True)
class MetadataPin:
    name: str
    path: Path
    digest: str
    cap: int


@dataclass(frozen=True, slots=True)
class ArrayDescriptor:
    name: str
    field: str
    shape: tuple[int, int]
    digest: str


@dataclass(frozen=True, slots=True)
class PairDescriptor:
    job_id: str
    index: int
    inputs: ArrayDescriptor
    responses: ArrayDescriptor


@dataclass(frozen=True, slots=True)
class TranscriptManifest:
    committed_collection: str
    profile: _Profile
    metadata: tuple[MetadataPin, ...]
    pairs: tuple[PairDescriptor, ...]


def _compile(
    pins: dict[str, tuple[Path, str]],
    profile: _Profile,
    committed_collection: str,
    *,
    _reader: Callable[[Path, int], bytes] = _read,
) -> TranscriptManifest:
    """Explicit literal seam; production load_manifest has fixed profile/pins."""
    if set(pins) != {"audit", "commitments", "collection", "progress"}:
        raise ValueError("Exact public pins required")
    values: dict[str, Any] = {}
    metadata = []
    for key in ("audit", "commitments", "collection", "progress"):
        path, expected = pins[key]
        if not _digest(expected):
            raise ValueError("Public hash required")
        cap = COMMITMENTS_CAP if key == "commitments" else METADATA_CAP
        blob = _reader(path, cap)
        if _hash(blob) != expected:
            raise ValueError("Public hash mismatch")
        values[key] = _json(blob)
        metadata.append(MetadataPin(key, path, expected, cap))
    a, c, q, h = (values[k] for k in ("audit", "collection", "progress", "commitments"))
    if (
        a.get("approved") is not True
        or a.get("saved_outcome_audit_approved") is not True
        or a.get("actual_source_commit") != SOURCE
        or c.get("source_commit") != SOURCE
        or a.get("private_gate") != "closed"
        or c.get("status") != "completed"
        or q.get("status") != "persisted"
        or a.get("artifact_commitments_sha256") != pins["commitments"][1]
    ):
        raise ValueError("Public audit/source/collection lineage mismatch")
    expected_counts = dict(
        calls_reserved=profile.vector_rows,
        calls_attempted=profile.vector_rows,
        calls_returned=profile.vector_rows,
        calls_persisted=profile.vector_rows,
        callback_batches=len(profile.case_names) * 2 * profile.probes,
    )
    for receipt in (c, q):
        if any(
            type(receipt.get(k)) is not int or receipt[k] != v
            for k, v in expected_counts.items()
        ):
            raise ValueError("Native complete query counters required")
    if (
        type(a.get("vector_calls")) is not int
        or a["vector_calls"] != profile.vector_rows
        or type(a.get("callback_batches")) is not int
        or a["callback_batches"] != expected_counts["callback_batches"]
        or q.get("batch") != f"{profile.case_names[-1]}-s1-x{profile.probes - 1:02d}"
    ):
        raise ValueError("Public audit/final batch counter mismatch")
    if not isinstance(h.get("artifact_sha256"), dict) or not isinstance(
        h.get("metadata_sha256"), dict
    ):
        raise ValueError("Pinned public commitment maps required")
    for key, name in [
        ("collection", "collection-report.json"),
        ("progress", "query-progress.json"),
    ]:
        if h["metadata_sha256"].get("collection/" + name) != pins[key][1]:
            raise ValueError("Public receipt commitment mismatch")
    pairs = []
    for job, case in enumerate(profile.case_names):
        for step in range(2):
            for probe in range(profile.probes):
                arrays = []
                for field, cols in [
                    ("inputs", profile.n),
                    ("responses", profile.outputs),
                ]:
                    name = f"{case}-s{step}-x{probe:02d}-{field}.npz"
                    value = h["artifact_sha256"].get(
                        committed_collection + "/truth/" + name
                    )
                    if not _digest(value):
                        raise ValueError("Missing/invalid exact transcript commitment")
                    arrays.append(
                        ArrayDescriptor(name, field, (profile.rows, cols), value)
                    )
                pairs.append(
                    PairDescriptor(
                        f"job{job:02d}", step * profile.probes + probe, *arrays
                    )
                )
    return TranscriptManifest(
        committed_collection, profile, tuple(metadata), tuple(pairs)
    )


def load_manifest(repository: Path) -> TranscriptManifest:
    """Public receipts only. This does NOT authorize or open a transcript."""
    return _compile(
        {name: (repository / rel, h) for name, rel, h in PUBLIC_PINS},
        REGISTERED,
        ORIGINAL,
    )
