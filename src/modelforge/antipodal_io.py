"""Bounded public paired-observation artifacts for isolated GLU reconstruction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .antipodal import recover_even_quadratic

MAX_BYTES = 16 * 1024**2
FIELDS = {
    "schema_version",
    "observation_kind",
    "family",
    "authorized",
    "dimension",
    "positive_outputs",
    "negative_outputs",
}


def reconstruct_observations(
    observations_path: str, destination: str
) -> dict[str, Any]:
    """Recover the even quadratic component only; read no teacher or heldout files."""
    source, target = Path(observations_path), Path(destination)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
        raise ValueError("Supply a regular public observation JSON file within 16MiB")
    raw = source.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Observation file grew beyond 16MiB")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError(
            "Observation schema requires exactly the documented public fields"
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported observation schema")
    if data["observation_kind"] != "isolated_glu_antipodal":
        raise ValueError("Requires isolated GLU responses at canonical signed probes")
    if data["family"] not in {"bias_free_swiglu", "bias_free_geglu"}:
        raise ValueError("Only declared bias-free SwiGLU/GeGLU families are supported")
    if data["authorized"] is not True:
        raise ValueError("Explicit authorization for these observations is required")
    q = recover_even_quadratic(
        data["positive_outputs"], data["negative_outputs"], data["dimension"]
    )
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Artifact destination cannot use symlink ancestors")
    target.mkdir(parents=True, exist_ok=False)
    artifact = json.dumps(
        {"schema_version": 1, "quadratic_even": q.tolist()}, allow_nan=False
    ).encode()
    (target / "quadratic-even.json").write_bytes(artifact)
    (target / "observations.json").write_bytes(raw)
    report: dict[str, Any] = {
        "method": "canonical antipodal quadratic reconstruction",
        "family_declared": data["family"],
        "dimension": data["dimension"],
        "outputs": int(q.shape[0]),
        "signed_observation_rows": data["dimension"] * (data["dimension"] + 1),
        "observations_sha256": hashlib.sha256(raw).hexdigest(),
        "quadratic_artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("antipodal.py"))
        },
        "scope": "Even component E(x)=(f(x)+f(-x))/2 of a declared isolated bias-free GLU; assumptions are not certified by input metadata",
        "full_glu_recovered": False,
        "original_weights_recovered": False,
        "heldout_verified": False,
        "observation_precision": "unspecified; float64 JSON decoding is not a source-precision certificate",
        "numeric_certificate": False,
        "remote_calls": 0,
        "reference": "https://arxiv.org/abs/2608.06631",
    }
    (target / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
