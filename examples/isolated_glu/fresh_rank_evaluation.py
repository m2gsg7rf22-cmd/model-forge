"""Privileged evaluation ONLY after all public choices are immutably sealed."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
from rank_selection import _svd
from scaling_domain_math import svec, direction_residuals
from fresh_rank_common import (
    create_store,
    read_json,
    read_arrays,
    records,
    save_json,
    sha,
    verify_manifest,
)
from fresh_rank_protocol_inventory import profile_schemas
from fresh_rank_view import FDView
from fresh_rank_estimation import registered_view

Array = NDArray[np.float64]
RULES = ("strict", "gap", "stable")


def matrix(a: Array) -> Array:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.float64
        or a.ndim != 2
        or not 1 <= a.shape[0] <= 512
        or not 1 <= a.shape[1] <= 528
        or not np.isfinite(a).all()
    ):
        raise ValueError("Bounded finite native observation matrix required")
    return a


def norm(a: Array) -> float:
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    return 0.0 if scale == 0 else float(scale * np.linalg.norm(a / scale))


def basis_check(b: Array, p: int) -> None:
    if (
        not isinstance(b, np.ndarray)
        or b.dtype != np.float64
        or b.ndim != 2
        or b.shape[1] != p
        or b.shape[0] > p
        or not np.isfinite(b).all()
        or (
            len(b)
            and not np.allclose(
                b @ b.T,
                np.eye(len(b)),
                rtol=0,
                atol=64 * np.finfo(float).eps * max(b.shape),
            )
        )
    ):
        raise ValueError("Finite orthonormal row basis required")


def energy_loss(a: Array, b: Array) -> float | None:
    a = matrix(a)
    basis_check(b, a.shape[1])
    scale = float(np.abs(a).max())
    if scale == 0:
        return None
    normalized = a / scale
    if np.any((a != 0) & (normalized == 0)):
        raise ValueError("Energy normalization underflow")
    residual = normalized - (normalized @ b.T) @ b
    top = norm(residual)
    value = top / norm(normalized)
    if not np.isfinite(value) or (top > 0 and value == 0):
        raise ValueError("Energy ratio arithmetic failure")
    return value


def reference_summary(a: Array) -> dict[str, Any]:
    a = matrix(a)
    s, relative, v, _ = _svd(a)
    ranks = [
        int(np.count_nonzero(relative > cutoff * relative[0])) if relative[0] > 0 else 0
        for cutoff in (1e-11, 1e-10, 1e-9)
    ]
    return dict(
        rank=ranks[1],
        ranks=ranks,
        sensitive=ranks[0] != ranks[2],
        basis=v[: ranks[1]],
        spectrum=s,
        all_right=v,
    )


def component_residuals(reference: dict[str, Any], b: Array) -> Array | None:
    """Normalized amplitudes; squares are per-SVD-component energy fractions.

    Components inside degenerate singular spaces depend on SVD basis choice.
    Their squared sum is invariant; component IDs are not aligned across channels.
    """
    s, v = reference["spectrum"], reference["all_right"]
    basis_check(b, v.shape[1])
    maximum = float(s.max())
    if maximum == 0:
        return None
    weights = s[: len(v)] / maximum
    if np.any((s[: len(v)] != 0) & (weights == 0)):
        raise ValueError("Component normalization underflow")
    original_weights = weights.copy()
    weights /= np.linalg.norm(weights)
    if np.any((original_weights != 0) & (weights == 0)):
        raise ValueError("Component weight underflow")
    residual = v - (v @ b.T) @ b
    result = np.zeros(v.shape[1])
    row_norms = np.array([norm(row) for row in residual])
    result[: len(v)] = weights * row_norms
    if np.any((weights != 0) & (row_norms != 0) & (result[: len(v)] == 0)):
        raise ValueError("Component weighted underflow")
    if not np.isfinite(result).all():
        raise ValueError("Component residual arithmetic failure")
    return result


def qualify(
    reference: dict[str, Any],
    p: int,
    k: int | None,
    lost: float | None,
    extra: float | None,
    gate_max: float | None,
    random_min: float | None,
) -> bool:
    r = reference["rank"]
    if reference["sensitive"] or r == 0:
        return False
    if r == p:
        return k is None
    if k != r or any(
        v is None or not np.isfinite(v) for v in (lost, extra, gate_max, random_min)
    ):
        return False
    assert (
        lost is not None
        and extra is not None
        and gate_max is not None
        and random_min is not None
    )
    return bool(
        0 <= lost <= 0.01
        and 0 <= extra <= 0.01
        and 0 <= gate_max <= 0.01
        and random_min > gate_max
    )


def geometric(a: Array, b: Array) -> float | None:
    if not len(a):
        return None
    residual = a - (a @ b.T) @ b
    top = norm(residual)
    value = top / np.sqrt(len(a))
    if not np.isfinite(value) or (top > 0 and value == 0):
        raise ValueError("Geometric metric arithmetic failure")
    return float(value)


def evaluate_case(
    reference: Array,
    observed: Array,
    right: Array,
    choices: dict[str, Any],
    gates: Array,
    controls: Array,
) -> tuple[dict[str, Any], dict[str, Array]]:
    reference = matrix(reference)
    observed = matrix(observed)
    if reference.shape != observed.shape:
        raise ValueError("Matching reference/observed geometry required")
    p = observed.shape[1]
    basis_check(right, p)
    ref = reference_summary(reference)
    rb = ref["basis"]
    width = len(gates)
    q = len(right)
    if (
        gates.ndim != 2
        or not 1 <= width <= 96
        or controls.shape != (64, gates.shape[1])
        or gates.shape[1] * (gates.shape[1] + 1) // 2 != p
    ):
        raise ValueError("Bounded gate/control geometry required")
    methods: list[tuple[str, int | None, dict[str, Any]]] = [
        (k, choices[k]["rank"], {}) for k in RULES
    ]
    known = min(2 * width, q, p)
    for name, assumed in [
        ("known_width", width),
        ("wrong_width_half", max(1, width // 2)),
        ("wrong_width_double", 2 * width),
    ]:
        rank = min(2 * assumed, q, p)
        methods.append(
            (
                name,
                rank,
                dict(
                    assumed_width=assumed,
                    requested_rank=2 * assumed,
                    effective_rank=rank,
                    clipped=rank != 2 * assumed,
                    same_effective_rank_as_known=rank == known,
                ),
            )
        )
    arrays = dict(
        reference_spectrum=ref["spectrum"],
        reference_ranks=np.array(ref["ranks"], dtype=np.float64),
        ranks=np.array(
            [-1 if k is None else k for _, k, _ in methods], dtype=np.float64
        ),
        gate_residuals=np.full((6, width), -1.0),
        random_residuals=np.full((6, 64), -1.0),
        lost_subspace=np.full(6, -1.0),
        extra_subspace=np.full(6, -1.0),
        reference_relative_residual=np.full(6, -1.0),
        observed_relative_residual=np.full(6, -1.0),
        reference_component_residuals=np.full((6, p), -1.0),
        qualifies=np.zeros(6),
        reference_sensitive=np.array([ref["sensitive"]], dtype=np.float64),
    )
    reports = []
    for i, (name, k, metadata) in enumerate(methods):
        if k is not None and (type(k) is not int or not 0 <= k <= q):
            raise ValueError("Bounded integer candidate rank required")
        lost = extra = gate_max = random_min = None
        item: dict[str, Any] = dict(
            method=name, rank=k, metadata=metadata, certificate=False
        )
        if k is not None:
            b = right[:k]
            lost = geometric(rb, b)
            extra = geometric(b, rb)
            for field, value in [
                ("lost_subspace", lost),
                ("extra_subspace", extra),
                ("reference_relative_residual", energy_loss(reference, b)),
                ("observed_relative_residual", energy_loss(observed, b)),
            ]:
                item[field] = value
                if value is not None:
                    arrays[field][i] = value
            components = component_residuals(ref, b)
            if components is not None:
                arrays["reference_component_residuals"][i] = components
            if ref["rank"] > 0:
                arrays["gate_residuals"][i] = direction_residuals(b, gates)
                arrays["random_residuals"][i] = direction_residuals(b, controls)
                gate_max = float(arrays["gate_residuals"][i].max())
                random_min = float(arrays["random_residuals"][i].min())
                item.update(
                    gate_residual_max=gate_max,
                    random_residual_min=random_min,
                    random_residual_max=float(arrays["random_residuals"][i].max()),
                )
        passed = qualify(ref, p, k, lost, extra, gate_max, random_min)
        arrays["qualifies"][i] = passed
        item.update(
            qualifies=passed,
            rank_error=None if k is None else k - ref["rank"],
            reference_rank_sensitive=ref["sensitive"],
            null_reason="no_candidate"
            if k is None
            else ("zero_reference" if ref["rank"] == 0 else None),
        )
        reports.append(item)
    return dict(
        reference_rank=ref["rank"],
        reference_ranks=ref["ranks"],
        reference_sensitive=ref["sensitive"],
        methods=reports,
        certificate=False,
    ), arrays


def validate_public_seal(
    est: Path, view: FDView, seal: dict[str, Any], er: dict[str, Any]
) -> None:
    """Check every choice/archive using public geometry before any truth read."""
    if (
        er.get("status") != "completed"
        or er.get("selections") != seal.get("selections")
        or seal.get("phase") != "all_public_choices_complete"
        or seal.get("certificate") is not False
        or seal.get("new_queries") != 0
        or seal.get("new_fits") != 0
        or len(seal.get("selections", [])) != len(view.cases)
    ):
        raise ValueError("Incomplete public seal")
    inventory = records(est / "arrays")
    expected = {
        f"case{i:02d}-{kind}.npz"
        for i in range(len(view.cases))
        for kind in ("bases", "choices")
    }
    if (
        set(inventory) != expected
        or sha(est / "arrays/manifest.jsonl") != seal["artifact_manifest_sha256"]
    ):
        raise ValueError("Exact sealed archive inventory required")
    for i, case in enumerate(view.cases):
        selection = seal["selections"][i]
        if (
            selection.get("case") != case.name
            or type(selection.get("index")) is not int
            or selection["index"] != i
            or selection.get("meta") != asdict(case)
            or set(selection.get("choices", {})) != set(RULES)
        ):
            raise ValueError("Sealed case/order/metadata mismatch")
        p = case.n * (case.n + 1) // 2
        q = min(case.probes * case.outputs, p)
        count = min(q, p - 1)
        fields: dict[str, tuple[int, ...]] = {
            k + "_basis": (q, p) for k in ("fine", "coarse", "richardson")
        }
        fields.update({k + "_spectrum": (p,) for k in ("fine", "coarse", "richardson")})
        chosen_fields: dict[str, tuple[int, ...]] = {
            k + "_projector": (p, p) for k in RULES
        }
        chosen_fields.update(
            log_gaps=(3, count),
            gap_ratios=(3, count),
            eligible=(3, count),
            resolution_clipped=(3, count),
            stability=(count,),
            resolution_floor=(1,),
            ranks=(3,),
        )
        loaded = []
        for key, kind, schema in (
            ("basis", "bases", fields),
            ("selected", "choices", chosen_fields),
        ):
            name = f"case{i:02d}-{kind}.npz"
            if selection.get(key) != inventory[name]:
                raise ValueError("Sealed archive descriptor mismatch")
            loaded.append(
                read_arrays(est / "arrays" / name, schema, inventory[name]["sha256"])
            )
        bases, saved = loaded
        for kind in ("fine", "coarse", "richardson"):
            basis_check(bases[kind + "_basis"], p)
        right = bases["richardson_basis"]
        for j, rule in enumerate(RULES):
            choice = selection["choices"][rule]
            k = choice.get("rank")
            if (
                choice.get("certificate") is not False
                or (k is None and rule == "strict")
                or (k is not None and (type(k) is not int or not 0 <= k <= q))
                or saved["ranks"][j] != (-1 if k is None else k)
            ):
                raise ValueError("Sealed integer rank/metadata mismatch")
            projector = np.zeros((p, p)) if k is None else right[:k].T @ right[:k]
            if not np.allclose(
                saved[rule + "_projector"],
                projector,
                rtol=0,
                atol=64 * np.finfo(float).eps * max(q, p),
            ):
                raise ValueError("Sealed rank/projector mismatch")
    verify_manifest(est / "arrays")


def build(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["root"])
    est = root / "estimator"
    collection = root / "collection"
    # Validate all-choice seal and public sources BEFORE any truth metadata read.
    if (
        sha(est / "choices-sealed.json") != request["seal_sha256"]
        or sha(est / "estimator-report.json") != request["estimator_report_sha256"]
    ):
        raise ValueError("Immutable estimator receipt/seal mismatch")
    seal = read_json(est / "choices-sealed.json")
    er = read_json(est / "estimator-report.json")
    if er.get("status") != "completed" or er.get("selections") != seal.get(
        "selections"
    ):
        raise ValueError("Incomplete sealed choices")
    inp = seal["input_commitments"]
    view = FDView(collection / "public", inp["cases_sha256"], inp["manifest_sha256"])
    registered_view(view)
    if (
        set(inp["used_input_sha256"]) != set(view.records)
        or len(seal["selections"]) != 24
    ):
        raise ValueError("All144publicinputs/24choices required")
    for name, digest in inp["used_input_sha256"].items():
        if sha(view.arrays / name) != digest or view.records[name]["sha256"] != digest:
            raise ValueError("Public input changed before truth")
        view.used[name] = digest
    est_inventory = verify_manifest(est / "arrays")
    if (
        est_inventory["shards"] != 48
        or sha(est / "arrays/manifest.jsonl") != seal["artifact_manifest_sha256"]
    ):
        raise ValueError("All48sealedestimatorarchives required")
    validate_public_seal(est, view, seal, er)
    if sha(collection / "seed-commitment.json") != request["seed_commitment_sha256"]:
        raise ValueError("Seed commitment changed after collection")
    if sha(collection / "truth-commitment.json") != request["truth_commitment_sha256"]:
        raise ValueError("Pre-estimation truth commitment mismatch")
    commitment = read_json(collection / "truth-commitment.json")
    seed_commitment = read_json(collection / "seed-commitment.json")
    if (
        seed_commitment.get("source_commit") != request["source_commit"]
        or seed_commitment.get("mapping_withheld") is not True
        or sha(collection / "registration.json") != commitment["registration_sha256"]
        or commitment["registration_sha256"] != seed_commitment["sha256"]
        or sha(collection / "truth/manifest.jsonl")
        != commitment["truth_manifest_sha256"]
        or commitment["public_manifest_sha256"] != inp["manifest_sha256"]
        or commitment["public_cases_sha256"] != inp["cases_sha256"]
    ):
        raise ValueError("Committed source view changed")
    secret = read_json(collection / "registration.json")
    from fresh_rank_collection import assignment

    if (
        set(secret) != {"seed", "nonce", "mapping"}
        or len(bytes.fromhex(secret["nonce"])) != 32
        or secret["mapping"] != assignment(bytes.fromhex(secret["seed"]))
    ):
        raise ValueError("Nonce/seed/assignment reveal mismatch")
    truth_records = records(collection / "truth")
    est_records = records(est / "arrays")
    output = root / "evaluation"
    output.mkdir(exist_ok=False)
    store = create_store(output / "arrays", request["store_limits"])
    results = []
    for i, case in enumerate(view.cases):
        view.verify_unchanged()
        verify_manifest(est / "arrays")
        selection = seal["selections"][i]
        if selection["case"] != case.name or selection["index"] != i:
            raise ValueError("Sealed case order changed")
        teacher = secret["mapping"][i // 2]
        n, m = teacher["n"], teacher["width"]
        if teacher["id"] != case.name.split("-")[0] or n != case.n:
            raise ValueError("Truth/public case mismatch")
        schema = {s.kind: s for s in profile_schemas(n, m)}
        name = case.name + "-context.npz"
        ctx = read_arrays(
            collection / "truth" / name,
            dict(schema["context"].fields),
            truth_records[name]["sha256"],
        )
        h = np.concatenate(
            [
                read_arrays(
                    collection / "truth" / f"{case.name}-reference-h{j:02d}.npz",
                    dict(schema["reference"].fields),
                    truth_records[f"{case.name}-reference-h{j:02d}.npz"]["sha256"],
                )["hessian"]
                for j in range(2)
            ]
        )
        basename = selection["basis"]["name"]
        if (
            basename != f"case{i:02d}-bases.npz"
            or est_records[basename] != selection["basis"]
        ):
            raise ValueError("Sealed basis descriptor mismatch")
        bases = read_arrays(
            est / "arrays" / basename,
            dict(schema["bases"].fields),
            selection["basis"]["sha256"],
        )
        obs = view.packed(case.name)["richardson"]
        ref = svec(h).reshape(obs.shape)
        result, arrays = evaluate_case(
            ref,
            obs,
            bases["richardson_basis"],
            selection["choices"],
            ctx["reference_g"],
            ctx["random_directions"],
        )
        result.update(
            case=case.name, width_for_evaluation_only=m, teacher_kind=teacher["kind"]
        )
        results.append(result)
        store.save(f"case{i:02d}-evaluation.npz", **arrays)
    view.verify_unchanged()
    verify_manifest(collection / "truth")
    verify_manifest(est / "arrays")
    if (
        sha(collection / "truth/manifest.jsonl") != commitment["truth_manifest_sha256"]
        or sha(est / "choices-sealed.json") != request["seal_sha256"]
    ):
        raise ValueError("Source/seal changed during evaluation")
    inventory = verify_manifest(store.root)
    report = dict(
        status="completed",
        cases=results,
        artifact_inventory=inventory,
        artifact_manifest_sha256=sha(store.root / "manifest.jsonl"),
        rule_passes={
            k: all(
                next(m["qualifies"] for m in c["methods"] if m["method"] == k)
                for c in results
            )
            for k in ("gap", "stable")
        },
        certificate=False,
        new_queries=0,
        new_fits=0,
        private_gate="closed",
        source_commit=request["source_commit"],
    )
    save_json(output / "evaluation-report.json", report)
    return report


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        r = build(request)
        print(
            json.dumps(
                dict(
                    status=r["status"],
                    report_sha256=sha(
                        Path(request["root"]) / "evaluation/evaluation-report.json"
                    ),
                    rule_passes=r["rule_passes"],
                )
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                dict(
                    status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
                )
            )
        )
