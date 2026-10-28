"""Public FD-only view for future owned variable-width research.

Component only: no teachers/RNG/query/forward/fitting/truth-manifest parser.
Source/view separation is not an OS security or statistical blinding guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw, read_arrays
from scaling_domain_math import svec
from scaling_domain_store import (
    _ancestry,
    _unique_object,
    sha,
)

Array = NDArray[np.float64]
VARIANTS = ("fine", "coarse", "richardson")
NAME = re.compile(r"c(?:0[0-9]|1[01])-(?:fp64|fp32)")
HEX = re.compile(r"[0-9a-f]{64}")
META_LIMIT = 128 * 1024


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    n: int
    outputs: int
    probes: int
    compute: str


def _commitment(value: str) -> None:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError("Explicit lowercase64hex commitment required")


def _pinned(path: Path, digest: str) -> bytes:
    _commitment(digest)
    data: bytes = raw(path, META_LIMIT)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Public metadata commitment mismatch")
    return data


class FDView:
    """Strict finite public schema; never read width-dependent truth metadata."""

    def __init__(self, root: Path, cases_sha: str, manifest_sha: str) -> None:
        self.root = root.absolute()
        self.cases_sha = cases_sha
        self.manifest_sha = manifest_sha
        self.arrays = self.root / "arrays"
        _ancestry(self.root)
        _ancestry(self.arrays)
        meta = json.loads(
            _pinned(self.root / "cases.json", cases_sha),
            object_pairs_hook=_unique_object,
        )
        if (
            not isinstance(meta, dict)
            or set(meta) != {"schema", "steps", "cases"}
            or meta["schema"] != "modelforge-fresh-rank-observations-v1"
            or meta["steps"] != [0.3, 0.6]
        ):
            raise ValueError("Exact observation-only metadata schema required")
        if not isinstance(meta["cases"], list) or not 1 <= len(meta["cases"]) <= 24:
            raise ValueError("One to24 public cases required")
        cases = []
        for item in meta["cases"]:
            if not isinstance(item, dict) or set(item) != {
                "name",
                "n",
                "outputs",
                "probes",
                "compute",
            }:
                raise ValueError(
                    "Public geometry/compute fields only; no width/seed/type"
                )
            if not isinstance(item["name"], str) or not NAME.fullmatch(item["name"]):
                raise ValueError("Anonymous bounded case identifier required")
            if (
                any(type(item[k]) is not int for k in ("n", "outputs", "probes"))
                or not 1 <= item["n"] <= 32
                or not 1 <= item["outputs"] <= 32
                or not 1 <= item["probes"] <= 16
            ):
                raise ValueError("Bounded native integer observation geometry required")
            expected = "fp64" if item["name"].endswith("-fp64") else "fp32_compute"
            if item["compute"] != expected:
                raise ValueError("Compute channel/case mismatch")
            cases.append(Case(**item))
        self.cases = tuple(cases)
        self._by_name = {case.name: case for case in cases}
        if len(self._by_name) != len(cases):
            raise ValueError("Duplicate public case")
        data = _pinned(self.arrays / "manifest.jsonl", manifest_sha)
        if not data.endswith(b"\n") or not 1 <= len(data.splitlines()) <= 144:
            raise ValueError("Complete bounded observation-only manifest required")
        self.records: dict[str, dict[str, Any]] = {}
        expected_names = {
            f"{case.name}-{kind}-h{i:02d}.npz"
            for case in cases
            for kind in VARIANTS
            for i in range((case.probes + 7) // 8)
        }
        for line in data.splitlines():
            item = json.loads(line, object_pairs_hook=_unique_object)
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "bytes", "sha256"}
                or not isinstance(item["name"], str)
                or item["name"] not in expected_names
                or item["name"] in self.records
                or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= 8 * 1024**2
            ):
                raise ValueError("Observation-only manifest schema/inventory required")
            _commitment(item["sha256"])
            self.records[item["name"]] = item
        if set(self.records) != expected_names:
            raise ValueError("Exact FD-only manifest inventory required")
        self.used: dict[str, str] = {}
        self._inventory()

    def _inventory(self) -> None:
        _ancestry(self.root)
        _ancestry(self.arrays)
        if {p.name for p in self.root.iterdir()} != {"cases.json", "arrays"} or {
            p.name for p in self.arrays.iterdir()
        } != set(self.records) | {"manifest.jsonl"}:
            raise ValueError("Exact public-only directory inventory required")
        for name, record in self.records.items():
            path = self.arrays / name
            _ancestry(path)
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError("Public array missing/size changed")

    def verify_unchanged(self) -> dict[str, Any]:
        _pinned(self.root / "cases.json", self.cases_sha)
        _pinned(self.arrays / "manifest.jsonl", self.manifest_sha)
        self._inventory()
        for name, digest in self.used.items():
            if sha(self.arrays / name) != digest:
                raise ValueError("Used public input changed")
        return dict(
            cases_sha256=self.cases_sha,
            manifest_sha256=self.manifest_sha,
            used_input_sha256=dict(self.used),
            inputs_unchanged=True,
        )

    def packed(self, case_name: str) -> dict[str, Array]:
        if case_name not in self._by_name:
            raise ValueError("Unknown public case")
        self.verify_unchanged()
        case = self._by_name[case_name]
        result: dict[str, Array] = {}
        for kind in VARIANTS:
            pieces = []
            for start in range(0, case.probes, 8):
                name = f"{case.name}-{kind}-h{start // 8:02d}.npz"
                record = self.records[name]
                shape = (min(8, case.probes - start), case.outputs, case.n, case.n)
                arrays: dict[str, Array] = read_arrays(
                    self.arrays / name, {"hessian": shape}, record["sha256"]
                )
                self.used[name] = record["sha256"]
                pieces.append(arrays["hessian"])
            matrix = svec(np.concatenate(pieces)).reshape(
                case.probes * case.outputs, case.n * (case.n + 1) // 2
            )
            matrix.setflags(write=False)
            result[kind] = matrix
        self.verify_unchanged()
        return result
