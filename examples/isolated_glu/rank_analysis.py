"""Zero-query PUBLIC saved-observation development; seal choices before truth.

Reviewed source/view separation, not OS sandboxing or blinded unknown families.
No RNG, model-forward, reference queries, fitting or private model access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from modelforge.glu32_io import raw, read_arrays
from rank_selection import select, _svd
from scaling_domain_math import svec, direction_residuals
from scaling_domain_store import (
    ShardStore,
    NAME,
    sha,
    _ancestry,
    _unique_object,
    verify_manifest,
)

Array = NDArray[np.float64]
VARIANTS = ("fine", "coarse", "richardson")
RULES = ("strict", "gap", "stable")
CASE = re.compile(r"p[0-3]-t[01]-fp(?:32|64)")
SPECS = [
    (f"p{p}-t{t}-{channel}", dict(n=4 << p, outputs=4 << p, probes=64))
    for p in range(4)
    for t in range(2)
    for channel in ("fp64", "fp32")
]


def save_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(encoded) > 128 * 1024:
        raise ValueError("Report exceeds128KiB")
    temporary = path.with_suffix(".tmp")
    if temporary.write_bytes(encoded) != len(encoded):
        raise OSError("Short JSON/seal write")
    temporary.replace(path)


class Inputs:
    """Hash-pinned whitelist; NO privileged read until immutable phase1 seal."""

    def __init__(
        self, root: Path, manifest_sha: str, specs: list[tuple[str, dict[str, int]]]
    ) -> None:
        self.root = root.absolute()
        _ancestry(self.root)
        data = raw(root / "manifest.jsonl", 1024**2)
        if hashlib.sha256(data).hexdigest() != manifest_sha:
            raise ValueError("Legacy manifest commitment mismatch")
        if not data.endswith(b"\n"):
            raise ValueError("Truncated manifest")
        self.manifest_sha = manifest_sha
        self.records: dict[str, Any] = {}
        for line in data.splitlines():
            item = json.loads(line, object_pairs_hook=_unique_object)
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "bytes", "sha256"}
                or not isinstance(item["name"], str)
                or not NAME.fullmatch(item["name"])
                or item["name"] in self.records
                or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= 8 * 1024**2
                or not isinstance(item["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            ):
                raise ValueError("Invalid legacy manifest")
            self.records[item["name"]] = item
        self.specs = {case: dict(meta) for case, meta in specs}
        if len(self.specs) != len(specs) or not 1 <= len(specs) <= 16:
            raise ValueError("Bounded unique cases required")
        for case, meta in specs:
            if (
                not CASE.fullmatch(case)
                or set(meta) != {"n", "outputs", "probes"}
                or any(type(v) is not int for v in meta.values())
                or not 1 <= meta["n"] <= 32
                or not 1 <= meta["outputs"] <= 32
                or not 1 <= meta["probes"] <= 64
            ):
                raise ValueError("Observation metadata cannot include width/truth")
        self.opened: list[str] = []
        self.used: dict[str, str] = {}
        self.seal_path: Path | None = None
        self.seal_sha: str | None = None

    def _read(
        self, name: str, shapes: dict[str, tuple[int, ...]], *, privileged: bool
    ) -> dict[str, Array]:
        if privileged:
            self.verify_manifest_commitment()
            if self.seal_path is None or sha(self.seal_path) != self.seal_sha:
                raise ValueError("Privileged read before valid choices seal")
        allowed = False
        for case in self.specs:
            suffix = name.removeprefix(case + "-")
            if name.startswith(case + "-"):
                allowed = (
                    bool(
                        re.fullmatch(r"(?:reference)-h0[0-7]\.npz", suffix)
                        or suffix == "context.npz"
                    )
                    if privileged
                    else bool(
                        re.fullmatch(r"(?:fine|coarse|richardson)-h0[0-7]\.npz", suffix)
                    )
                )
        if not allowed or name not in self.records:
            raise ValueError("Array outside allowed phase/view")
        record = self.records[name]
        path = self.root / name
        if (
            path.stat().st_size != record["bytes"]
            or not 0 < record["bytes"] <= 8 * 1024**2
        ):
            raise ValueError("Bounded source shard required")
        arrays: dict[str, Array] = read_arrays(path, shapes, record["sha256"])
        self.opened.append(name)
        self.used[name] = record["sha256"]
        return arrays

    def hessians(self, case: str, kind: str, meta: dict[str, int]) -> Array:
        if meta != self.specs.get(case) or kind not in (*VARIANTS, "reference"):
            raise ValueError("Invalid observation case/metadata")
        n, o, count = meta["n"], meta["outputs"], meta["probes"]
        values = []
        for start in range(0, count, 8):
            values.append(
                self._read(
                    f"{case}-{kind}-h{start // 8:02d}.npz",
                    {"hessian": (min(8, count - start), o, n, n)},
                    privileged=kind == "reference",
                )["hessian"]
            )
        return np.concatenate(values)

    def observations(self, case: str, meta: dict[str, int]) -> dict[str, Array]:
        return {
            kind: svec(self.hessians(case, kind, meta)).reshape(
                -1, meta["n"] * (meta["n"] + 1) // 2
            )
            for kind in VARIANTS
        }

    def context(self, case: str, meta: dict[str, int]) -> dict[str, Array]:
        if self.seal_path is None:
            raise ValueError("Privileged context before seal")
        if meta != self.specs.get(case):
            raise ValueError("Context metadata mismatch")
        # Width belongs ONLY to privileged evaluation, never observation loader.
        n, o, count = meta["n"], meta["outputs"], meta["probes"]
        m = 2 * n
        p = n * (n + 1) // 2
        return self._read(
            case + "-context.npz",
            dict(
                g=(m, n),
                u=(m, n),
                d=(o, m),
                reference_g=(m, n),
                reference_u=(m, n),
                reference_d=(o, m),
                probes=(count, n),
                random_directions=(64, n),
                atoms=(2 * m, p),
                preactivations=(count, m),
            ),
            privileged=True,
        )

    def seal(self, path: Path, report: dict[str, Any]) -> None:
        if path.exists() or self.seal_path is not None:
            raise ValueError("Immutable phase1 seal already exists")
        if {r["case"] for r in report["selections"]} != set(self.specs):
            raise ValueError("All choices must be persisted before truth")
        save_json(path, report)
        self.seal_path = path
        self.seal_sha = sha(path)

    def verify_manifest_commitment(self) -> None:
        if sha(self.root / "manifest.jsonl") != self.manifest_sha:
            raise ValueError("Original manifest commitment changed")

    def verify_unchanged(self) -> dict[str, Any]:
        self.verify_manifest_commitment()
        for name, digest in self.used.items():
            if sha(self.root / name) != digest:
                raise ValueError("Original selected input changed")
        if self.seal_path is None or sha(self.seal_path) != self.seal_sha:
            raise ValueError("Choices seal changed")
        return {
            "files": len(self.used),
            "inputs_unchanged": True,
            "input_sha256": dict(self.used),
            "seal_sha256": self.seal_sha,
        }


def _projector(basis: Array, rank: int | None, p: int) -> Array:
    return (
        np.zeros((p, p)) if rank is None or rank == 0 else basis[:rank].T @ basis[:rank]
    )


def _rms_residual(a: Array, b: Array) -> float | None:
    if len(a) == 0:
        return None
    residual = a - (a @ b.T) @ b
    scale = float(np.max(np.abs(residual)))
    if scale == 0:
        return 0.0
    value = float(scale * np.linalg.norm(residual / scale) / np.sqrt(len(a)))
    if value == 0:
        raise ValueError("Projector error underflow")
    return value


def evaluate(
    reader: Inputs,
    case: str,
    meta: dict[str, int],
    choices: dict[str, Any],
    rich_basis: Array,
) -> tuple[dict[str, Any], dict[str, Array]]:
    ctx = reader.context(case, meta)
    ref = svec(reader.hessians(case, "reference", meta)).reshape(
        -1, rich_basis.shape[1]
    )
    _, s, v, _ = _svd(ref)
    r = int(np.count_nonzero(s > 1e-10 * s[0])) if s[0] > 0 else 0
    rb = v[:r]
    p = rich_basis.shape[1]
    q = len(rich_basis)
    width = len(ctx["reference_g"])
    methods: list[tuple[str, int | None, dict[str, Any]]] = [
        (name, choices[name]["rank"], {}) for name in RULES
    ]
    known = min(2 * width, q, p)
    for name, assumed in [
        ("known_width", width),
        ("wrong_width_half", max(1, width // 2)),
        ("wrong_width_double", 2 * width),
    ]:
        requested = 2 * assumed
        effective = min(requested, q, p)
        methods.append(
            (
                name,
                effective,
                dict(
                    assumed_width=assumed,
                    requested_rank=requested,
                    effective_rank=effective,
                    clipped=requested != effective,
                    same_effective_rank_as_known=effective == known,
                ),
            )
        )
    records = []
    gates = np.full((6, width), -1.0)
    controls = np.full((6, 64), -1.0)
    for i, (name, k, metadata) in enumerate(methods):
        item: dict[str, Any] = dict(
            method=name, rank=k, certificate=False, metadata=metadata
        )
        if k is None:
            item.update(lost_signal=None, extra_subspace=None, rank_error=None)
        else:
            b = rich_basis[:k]
            gates[i] = direction_residuals(b, ctx["reference_g"])
            controls[i] = direction_residuals(b, ctx["random_directions"])
            item.update(
                lost_signal=_rms_residual(rb, b),
                extra_subspace=_rms_residual(b, rb),
                rank_error=k - r,
                full_reference_truncated=bool(r == p and k < p),
                gate_residual_max=float(gates[i].max()),
                random_residual_min=float(controls[i].min()),
                random_residual_max=float(controls[i].max()),
            )
        records.append(item)
    return dict(
        case=case,
        reference_numerical_rank=r,
        width_for_evaluation_only=width,
        methods=records,
    ), dict(
        gate_residuals=gates,
        random_residuals=controls,
        ranks=np.array(
            [-1 if k is None else k for _, k, _ in methods], dtype=np.float64
        ),
        reference_rank=np.array([r], dtype=np.float64),
    )


def run_views(
    reader: Inputs,
    store: ShardStore,
    output: Path,
    specs: list[tuple[str, dict[str, int]]],
) -> dict[str, Any]:
    selections = []
    for index, (case, meta) in enumerate(specs):
        obs = reader.observations(case, meta)
        result = select(obs["fine"], obs["coarse"], obs["richardson"])
        p = result["p"]
        v = result["bases"]["richardson"]
        bases = store.save(
            f"case{index:02d}-bases.npz",
            **{f"{k}_basis": a for k, a in result["bases"].items()},
            **{f"{k}_spectrum": a for k, a in result["spectra"].items()},
        )
        values = {
            f"{k}_projector": _projector(v, result["choices"][k]["rank"], p)
            for k in RULES
        }
        for k, a in result["diagnostics"].items():
            values[k] = (
                np.asarray(a, dtype=np.float64).reshape(1)
                if np.ndim(a) == 0
                else np.asarray(a, dtype=np.float64)
            )
        values["ranks"] = np.array(
            [
                -1
                if result["choices"][k]["rank"] is None
                else result["choices"][k]["rank"]
                for k in RULES
            ],
            dtype=np.float64,
        )
        selected = store.save(f"case{index:02d}-choices.npz", **values)
        selections.append(
            dict(
                case=case,
                index=index,
                choices=result["choices"],
                basis=bases,
                selected=selected,
            )
        )
    seal = dict(
        phase="all_observation_choices_complete",
        certificate=False,
        selections=selections,
        new_reference_queries=0,
        new_fits=0,
    )
    reader.seal(output / "choices-sealed.json", seal)
    evaluations = []
    for selection, (case, meta) in zip(selections, specs, strict=True):
        reader.verify_unchanged()
        # Verify ALL phase1 archives before each privileged evaluation. A later
        # case must not begin after an earlier or not-yet-used choice was altered.
        verify_manifest(store.root)
        for sealed_selection in selections:
            for key in ("basis", "selected"):
                record = sealed_selection[key]
                if sha(store.root / record["name"]) != record["sha256"]:
                    raise ValueError("Sealed output commitment changed")
        name = selection["basis"]["name"]
        digest = selection["basis"]["sha256"]
        path = store.root / name
        if sha(path) != digest:
            raise ValueError("Sealed estimator basis changed")
        with np.load(path, allow_pickle=False) as arrays:
            rich = np.array(arrays["richardson_basis"], copy=True)
        evaluation, arrays_out = evaluate(
            reader, case, meta, selection["choices"], rich
        )
        store.save(f"case{selection['index']:02d}-evaluation.npz", **arrays_out)
        evaluations.append(evaluation)
    checked = reader.verify_unchanged()
    save_json(output / "input-commitments.json", checked)
    return dict(
        status="completed",
        estimator_has_width_metadata=False,
        selections=selections,
        evaluations=evaluations,
        new_reference_queries=0,
        new_fits=0,
        inputs_unchanged=checked["inputs_unchanged"],
        selected_input_files=checked["files"],
        seal_sha256=checked["seal_sha256"],
        input_commitments_sha256=sha(output / "input-commitments.json"),
        artifact_inventory=verify_manifest(store.root),
        private_gate="closed",
        blind_or_reconstruction_claim=False,
    )
