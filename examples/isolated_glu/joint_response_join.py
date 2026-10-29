"""Strict parent join of frozen public rank/response sources; never a launcher.

Actual load_join calls require a distinct fully approved experiment protocol.
The learner receives no basis, source path, precision channel or true width.
"""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from variable_gate_view import GateBundle, load_bundle
from variable_response_manifest import (
    ORIGINAL,
    REGISTERED,
    TranscriptManifest,
    load_manifest,
)
from variable_response_view import ResponseBatch, ResponseView, open_view

RANK_PINS = (
    (
        "cases_sha256",
        "2d51bfe26130a1eebb054d71f1817a7e246005e9ea5a6f5824edc04cebc64cd6",
    ),
    (
        "manifest_sha256",
        "ffe53561cd35a5cd6de6d27b7537a98a05ab804b3c49477654ded3723b705e7f",
    ),
    ("seal_sha256", "f436992bc2f3b24fd593a373c2127235124626a58c82f95a5626594af2f76901"),
    (
        "report_sha256",
        "bf546c416a65e4d46b6fbd544cf0d19b130c17cf2f799a55f2cd365c5a51d2ec",
    ),
)


@dataclass(frozen=True, slots=True)
class JointContract:
    width_hypotheses: tuple[int, ...]
    paired_input_commitments: int


def compile_contract(rank: GateBundle, manifest: TranscriptManifest) -> JointContract:
    """No IO/RNG: cross-bind already validated source descriptors.

    This does not itself authenticate arbitrary constructed Python objects. The
    registered load_join calls BOTH frozen source validators before numeric jobs.
    """
    expected_mapping = {
        f"job{i:02d}": case + "-p17" for i, case in enumerate(REGISTERED.case_names)
    }
    if (
        manifest.profile != REGISTERED
        or manifest.committed_collection != ORIGINAL
        or len(rank.jobs) != 6
        or rank.provenance.get("mapping") != expected_mapping
        or rank.provenance.get("commitments") != dict(RANK_PINS)
        or tuple(job.job_id for job in rank.jobs) != tuple(expected_mapping)
    ):
        raise ValueError("exact registered source commitments/order/mapping required")
    widths = []
    for job in rank.jobs:
        if (
            job.n != 32
            or type(job.rank) is not int
            or not 0 < job.rank <= 192
            or job.rank % 2
        ):
            raise ValueError("proper even numeric rank<=192 required")
        widths.append(job.rank // 2)
    if any(widths[i] != widths[i + 1] for i in (0, 2, 4)):
        raise ValueError("paired width hypotheses differ; abstain")
    if len(manifest.pairs) != 204:
        raise ValueError("exact 204 ordered pairs required")
    for i, pair in enumerate(manifest.pairs):
        job_index, batch = divmod(i, 34)
        if (
            pair.job_id != f"job{job_index:02d}"
            or type(pair.index) is not int
            or pair.index != batch
        ):
            raise ValueError("canonical response descriptor order required")
        for field, descriptor in (
            ("inputs", pair.inputs),
            ("responses", pair.responses),
        ):
            name = f"{REGISTERED.case_names[job_index]}-s{batch // 17}-x{batch % 17:02d}-{field}.npz"
            if (
                descriptor.name != name
                or descriptor.field != field
                or descriptor.shape != (2112, 32)
                or not isinstance(descriptor.digest, str)
                or re.fullmatch("[0-9a-f]{64}", descriptor.digest) is None
            ):
                raise ValueError("exact response archive descriptor required")
    for group in range(3):
        for batch in range(34):
            left = manifest.pairs[group * 68 + batch]
            right = manifest.pairs[group * 68 + 34 + batch]
            if left.inputs.digest != right.inputs.digest:
                raise ValueError("paired encoded input commitments differ; abstain")
    return JointContract(tuple(widths), 102)


@dataclass(frozen=True, slots=True)
class JointJob:
    job_id: str
    n: int
    outputs: int
    width_hypothesis: int
    batch_count: int
    batches: Iterator[ResponseBatch]


class JoinedView:
    """Parent provenance owner; the original one-use response view enforces order."""

    def __init__(self, contract: JointContract, view: ResponseView):
        source_jobs = view.jobs
        if len(source_jobs) != 6 or any(
            j.job_id != f"job{i:02d}"
            or j.n != 32
            or j.outputs != 32
            or j.batch_count != 34
            for i, j in enumerate(source_jobs)
        ):
            raise ValueError("exact registered response jobs required")
        self._contract = contract
        self._view = view
        self.jobs = tuple(
            JointJob(
                j.job_id,
                j.n,
                j.outputs,
                contract.width_hypotheses[i],
                j.batch_count,
                j.batches,
            )
            for i, j in enumerate(source_jobs)
        )

    def report(self) -> dict[str, Any]:
        return dict(
            response=self._view.report(),
            width_hypotheses=list(self._contract.width_hypotheses),
            paired_input_commitments=self._contract.paired_input_commitments,
            width_is_hypothesis=True,
            development_only=True,
            certificate=False,
            private_gate="closed",
        )

    def finish(self) -> dict[str, Any]:
        return self._view.finish()


def load_join(
    repository: Path, public_root: Path, estimator_root: Path, response_root: Path
) -> JoinedView:
    """Actual reads: call only AFTER full approved source/resource/attempt gates.

    No registered RNG/fit is invoked. Equal archive digests become an input-value
    guarantee only after open_view validates every archive/schema against them.
    """
    manifest = load_manifest(repository)
    rank = load_bundle(public_root, estimator_root, dict(RANK_PINS))
    contract = compile_contract(rank, manifest)
    del rank  # Basis never enters numeric jobs or remains cached by this join.
    view = open_view(response_root, manifest)
    return JoinedView(contract, view)
