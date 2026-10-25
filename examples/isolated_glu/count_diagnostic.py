"""Disclosed public replay telemetry; no target queries, teacher truth or student."""

from __future__ import annotations
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys
from types import CodeType, FrameType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from modelforge import glu32
from modelforge.glu32_io import read_arrays, read_json, raw, SHAPES

Array = NDArray[np.float64]


def sha(path: Path) -> str:
    return hashlib.sha256(raw(path)).hexdigest()


def distance(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - (a @ b.T) @ b) / np.sqrt(len(a)))


def capture(
    operation: Callable[[], Any], roles: dict[CodeType, str]
) -> tuple[Any, str | None, list[dict[str, Any]], dict[str, Array]]:
    events: list[dict[str, Any]] = []
    snapshot: dict[str, Array] = {}

    def profiler(frame: FrameType, event: str, arg: Any) -> None:
        if event != "return" or frame.f_code not in roles:
            return
        role = roles[frame.f_code]
        local = frame.f_locals
        if role == "merge" and isinstance(arg, np.ndarray):
            events.append(
                {
                    "role": role,
                    "input_count": len(local["known"]),
                    "candidate_count": len(local["candidates"]),
                    "retained_count": len(arg),
                }
            )
        elif role == "search" and isinstance(arg, tuple):
            events.append(
                {"role": role, "retained_count": len(arg[0]), "rounds": arg[1]}
            )
        elif role == "directions":
            for key in ["known", "v", "residual", "initial_loss", "b"]:
                if isinstance(local.get(key), np.ndarray):
                    snapshot[key] = local[key].copy()
            events.append(
                {
                    "role": role,
                    "final_count": len(local["known"])
                    if isinstance(local.get("known"), np.ndarray)
                    else None,
                    "returned_directions": arg is not None,
                }
            )

    previous = sys.getprofile()
    try:
        sys.setprofile(profiler)
        return operation(), None, events, snapshot
    except ValueError as exc:
        return None, str(exc), events, snapshot
    finally:
        sys.setprofile(previous)


def spectrum(h: Array) -> tuple[dict[str, float], Array]:
    i, j = np.triu_indices(32)
    matrix = h[:, :, i, j].reshape(-1, len(i)) * np.where(i == j, 1.0, np.sqrt(2.0))
    _, s, v = np.linalg.svd(matrix, full_matrices=False)
    return {
        "singular_128": float(s[127]),
        "singular_129": float(s[128]),
        "gap": float(s[127] / s[128]),
    }, v[:128]


def diagnose(source: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    with (output / "replay-attempt.json").open("x") as f:
        json.dump(
            {
                "profiled_direction_calls_reserved": 1,
                "target_vector_calls": 0,
                "disclosed_replay": True,
            },
            f,
        )
    public = source / "public"
    if sha(public / "manifest.json") != config["observation_manifest_sha256"]:
        raise ValueError("Prior observation manifest commitment changed")
    meta = read_json(public / "manifest.json")
    if meta != {"channel": "fp32_compute", "files": config["expected_files"]}:
        raise ValueError("Prior calibration inventory changed")
    for name, expected in config["expected_files"].items():
        if sha(public / name) != expected:
            raise ValueError("Prior calibration commitment changed")
    pair = read_arrays(
        public / "curvature-pair.npz",
        {"fine": (64, 32, 32, 32), "coarse": (64, 32, 32, 32)},
        meta["files"]["curvature-pair.npz"],
    )
    h = read_arrays(
        public / "hessians.npz",
        {"hessian": (64, 32, 32, 32)},
        meta["files"]["hessians.npz"],
    )["hessian"]
    # Validate the saved multiscale payload without exposing it to this direction replay.
    read_arrays(public / "multiscale.npz", SHAPES, meta["files"]["multiscale.npz"])
    if not np.array_equal((4 * pair["fine"] - pair["coarse"]) / 3, h):
        raise ValueError("Richardson combination changed")
    spectra: dict[str, Any] = {}
    bases: dict[str, Array] = {}
    for name, array in [
        ("fine", pair["fine"]),
        ("coarse", pair["coarse"]),
        ("richardson", h),
    ]:
        spectra[name], bases[name] = spectrum(array)
    comparisons = {
        a + "_" + b: distance(bases[a], bases[b])
        for a, b in [
            ("fine", "coarse"),
            ("fine", "richardson"),
            ("coarse", "richardson"),
        ]
    }

    # The algorithm receives only savedH. No callbacks, truth or altered globals.
    def operation() -> Any:
        return glu32.directions(h)

    result, error, events, snapshot = capture(
        operation,
        {
            glu32.directions.__code__: "directions",
            glu32.search.__code__: "search",
            glu32.merge.__code__: "merge",
        },
    )
    if set(snapshot) != {"known", "b", "v", "residual", "initial_loss"}:
        raise ValueError("No final direction telemetry captured")
    known = snapshot["known"]
    q = glu32.pack(known[:, :, None] * known[:, None, :])
    residual = q - (q @ snapshot["b"].T) @ snapshot["b"]
    losses = np.sum(residual**2, axis=1)
    np.savez(
        output / "telemetry.npz",
        known=snapshot["known"],
        b=snapshot["b"],
        v=snapshot["v"],
        residual=snapshot["residual"],
        initial_loss=snapshot["initial_loss"],
    )
    report = {
        "scope": "Disclosed replay on prior public calibration; not independentreplication",
        "target_vector_calls": 0,
        "profiled_direction_calls": 1,
        "source_manifest_sha256": config["observation_manifest_sha256"],
        "spectra": spectra,
        "subspace_disagreement_rms_sines": comparisons,
        "disagreement_is_certified_noise": False,
        "replay_returned_directions": result is not None,
        "error": error,
        "final_count": len(known),
        "events": events,
        "retained_loss_quantiles": np.quantile(
            losses, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "retained_count_passing_original_residual_guard": int(np.sum(losses <= 1e-8)),
        "telemetry_sha256": sha(output / "telemetry.npz"),
        "prior_fit_entrypoint_called": False,
        "direct_frozen_direction_replay": True,
        "candidate_exported": False,
        "heldout_calls": 0,
        "private_gate": "closed",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(
        json.dumps(
            diagnose(
                Path(request["source"]), Path(request["output"]), request["config"]
            ),
            allow_nan=False,
        )
    )
