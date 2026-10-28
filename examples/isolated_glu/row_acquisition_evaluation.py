"""Own-prefix privileged evaluator for paired16/17, fixed metric formulas."""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
from rank_selection import _svd
from scaling_domain_math import svec, direction_residuals
from fresh_rank_evaluation import (
    norm,
    basis_check,
    geometric,
    component_residuals,
    qualify as base_qualify,
    validate_public_seal as base_validate_seal,
)
from fresh_rank_common import (
    create_store,
    read_json,
    read_arrays,
    records,
    save_json,
    sha,
    verify_manifest,
)
from row_acquisition_view import RowView, registered_view
from row_acquisition_inventory import profile_schemas

Array = NDArray[np.float64]
RULES = ("strict", "gap", "stable")


def matrix(a: Array) -> Array:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.float64
        or a.ndim != 2
        or not 1 <= a.shape[0] <= 544
        or not 1 <= a.shape[1] <= 528
        or not np.isfinite(a).all()
    ):
        raise ValueError("Bounded finite native observation matrix required")
    return a


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


def qualify(
    reference: dict[str, Any],
    p: int,
    k: int | None,
    lost: float | None,
    extra: float | None,
    gate_max: float | None,
    random_min: float | None,
) -> bool:
    return bool(
        reference["rank"] < p
        and base_qualify(reference, p, k, lost, extra, gate_max, random_min)
    )


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


def condition_evaluation(
    hessian: Array,
    observed: Array,
    right: Array,
    choices: dict[str, Any],
    ctx: dict[str, Array],
    budget: int,
) -> tuple[dict[str, Any], dict[str, Array]]:
    if (
        type(budget) is not int
        or budget not in (16, 17)
        or hessian.ndim != 4
        or len(hessian) != 17
    ):
        raise ValueError("Exact17source/16or17prefix required")
    expected = (
        budget * hessian.shape[1],
        hessian.shape[2] * (hessian.shape[2] + 1) // 2,
    )
    if observed.shape != expected:
        raise ValueError("Own-prefix observation/reference geometry required")
    reference = svec(hessian[:budget]).reshape(expected)
    return evaluate_case(
        reference,
        observed,
        right,
        choices,
        ctx["reference_g"],
        ctx["random_directions"],
    )


def validate_public_seal(
    est: Path, view: RowView, seal: dict[str, Any], er: dict[str, Any]
) -> None:
    base_validate_seal(est, cast(Any, view), seal, er)


def public_boundary(
    root: Path, collection: dict[str, Any], est: dict[str, Any]
) -> RowView:
    directory = root / "estimator"
    if sha(directory / "choices-sealed.json") != est.get("seal_sha256"):
        raise ValueError("Incomplete public condition seal")
    seal = read_json(directory / "choices-sealed.json")
    inp = seal["input_commitments"]
    if (
        inp["cases_sha256"] != collection["public_cases_sha256"]
        or inp["manifest_sha256"] != collection["public_manifest_sha256"]
    ):
        raise ValueError("Prefix source commitments changed")
    view = RowView(
        root / "collection/public", inp["cases_sha256"], inp["manifest_sha256"]
    )
    registered_view(view)
    if set(inp["used_input_sha256"]) != set(view.records):
        raise ValueError("All54sourcearchives required before truth")
    for name, digest in inp["used_input_sha256"].items():
        if sha(view.arrays / name) != digest or view.records[name]["sha256"] != digest:
            raise ValueError("Public input changed before truth")
        view.used[name] = digest
    validate_public_seal(directory, view, seal, est)
    return view


def build(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["root"])
    est = root / "estimator"
    collection = root / "collection"
    if (
        sha(est / "choices-sealed.json") != request["seal_sha256"]
        or sha(est / "estimator-report.json") != request["estimator_report_sha256"]
    ):
        raise ValueError("Sealed estimator source mismatch")
    er = read_json(est / "estimator-report.json")
    seal = read_json(est / "choices-sealed.json")
    # This validates ALL12 choices/24headers/projectors/54inputhashes before truth.
    view = public_boundary(root, request["collection_commitments"], er)
    for name, key in (
        ("truth-commitment.json", "truth_commitment_sha256"),
        ("seed-commitment.json", "seed_commitment_sha256"),
    ):
        if sha(collection / name) != request[key]:
            raise ValueError("Pre-estimation commitment changed")
    commitment = read_json(collection / "truth-commitment.json")
    seed_commitment = read_json(collection / "seed-commitment.json")
    if (
        seed_commitment.get("source_commit") != request["source_commit"]
        or seed_commitment.get("mapping_withheld") is not True
        or sha(collection / "registration.json") != commitment["registration_sha256"]
        or commitment["registration_sha256"] != seed_commitment["sha256"]
        or sha(collection / "truth/manifest.jsonl")
        != commitment["truth_manifest_sha256"]
        or commitment["public_manifest_sha256"] != view.manifest_sha
        or commitment["public_cases_sha256"] != view.cases_sha
    ):
        raise ValueError("Committed truth/seed/public view changed")
    secret = read_json(collection / "registration.json")
    from row_acquisition_collection import assignment

    if (
        set(secret) != {"seed", "nonce", "mapping"}
        or len(bytes.fromhex(secret["nonce"])) != 32
        or secret["mapping"] != assignment(bytes.fromhex(secret["seed"]))
    ):
        raise ValueError("Seed/nonce/width assignment mismatch")
    truth_records = records(collection / "truth")
    est_records = records(est / "arrays")
    output = root / "evaluation"
    output.mkdir(exist_ok=False)
    store = create_store(output / "arrays", request["store_limits"])
    results = []
    for i, case in enumerate(view.cases):
        view.verify_unchanged()
        verify_manifest(est / "arrays")
        base = view.base_cases[i // 2]
        teacher = secret["mapping"][i // 4]
        if teacher["id"] != base.name.split("-")[0] or teacher["n"] != case.n:
            raise ValueError("Truth/public condition mismatch")
        schema = {s.kind: dict(s.fields) for s in profile_schemas(teacher["width"])}
        name = base.name + "-context.npz"
        ctx = read_arrays(
            collection / "truth" / name,
            schema["context"],
            truth_records[name]["sha256"],
        )
        pieces = []
        for j in range(3):
            name = f"{base.name}-reference-h{j:02d}.npz"
            pieces.append(
                read_arrays(
                    collection / "truth" / name,
                    schema["reference" if j < 2 else "reference_tail"],
                    truth_records[name]["sha256"],
                )["hessian"]
            )
        selected = seal["selections"][i]
        name = selected["basis"]["name"]
        if est_records[name] != selected["basis"]:
            raise ValueError("Sealed basis changed")
        bases = read_arrays(
            est / "arrays" / name,
            schema[f"bases{case.probes}"],
            selected["basis"]["sha256"],
        )
        observed = view.packed(case.name)["richardson"]
        result, arrays = condition_evaluation(
            np.concatenate(pieces),
            observed,
            bases["richardson_basis"],
            selected["choices"],
            ctx,
            case.probes,
        )
        result.update(
            case=case.name,
            base_case=base.name,
            probes=case.probes,
            width_for_evaluation_only=teacher["width"],
            teacher_kind="generic",
            shared_context_sha256=truth_records[base.name + "-context.npz"]["sha256"],
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
        raise ValueError("Source changed during evaluation")
    paired = []
    for i, base in enumerate(view.base_cases):
        a, b = results[2 * i : 2 * i + 2]
        if a["shared_context_sha256"] != b["shared_context_sha256"]:
            raise ValueError("Paired controls changed")
        paired.append(
            dict(
                base_case=base.name,
                reference_ranks16=a["reference_ranks"],
                reference_ranks17=b["reference_ranks"],
                reference_sensitive16=a["reference_sensitive"],
                reference_sensitive17=b["reference_sensitive"],
                rules={
                    k: dict(
                        pass16=next(
                            m["qualifies"] for m in a["methods"] if m["method"] == k
                        ),
                        pass17=next(
                            m["qualifies"] for m in b["methods"] if m["method"] == k
                        ),
                    )
                    for k in ("gap", "stable")
                },
            )
        )
    by_budget = {
        str(b): {
            k: all(
                next(m["qualifies"] for m in c["methods"] if m["method"] == k)
                for c in results
                if c["probes"] == b
            )
            for k in ("gap", "stable")
        }
        for b in (16, 17)
    }
    inventory = verify_manifest(store.root)
    report = dict(
        status="completed",
        cases=results,
        paired=paired,
        by_budget=by_budget,
        rule_passes=by_budget["17"],
        artifact_inventory=inventory,
        artifact_manifest_sha256=sha(store.root / "manifest.jsonl"),
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
