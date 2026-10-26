"""Observation-only rank hypotheses, never rank/identifiability certificates.

No paths, metadata-width, teacher values, RNG, model imports or fitting. Common
structured error can mimic a stable spectral gap; every choice is uncertified.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
STRICT_RTOL = 1e-10
RESOLUTION_FACTOR = 64.0
MIN_GAP = 10.0
MAX_STABILITY = 0.01


def choose(scores: Array, eligible: NDArray[np.bool_]) -> int | None:
    """Maximum finite score, exact tie chooses smallest index."""
    if (
        eligible.dtype != np.bool_
        or scores.ndim != 1
        or eligible.shape != scores.shape
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Finite matching candidate vectors required")
    indices = np.flatnonzero(eligible)
    return None if not len(indices) else int(indices[np.argmax(scores[indices])])


def _svd(a: Array) -> tuple[Array, Array, Array, float]:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.float64
        or a.ndim != 2
        or not 1 <= a.shape[0] <= 2048
        or not 1 <= a.shape[1] <= 528
        or a.nbytes > 16 * 1024**2
    ):
        raise ValueError("Bounded nativeFP64 observation matrix required")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite observations")
    scale = float(np.max(np.abs(a)))
    normalized = a / (scale if scale else 1.0)
    if np.any((a != 0) & (normalized == 0)):
        raise ValueError("Observation normalization underflow")
    _, s, v = np.linalg.svd(normalized, full_matrices=False)
    normalized_s = np.pad(s, (0, a.shape[1] - len(s)))
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            values = normalized_s * scale
    except FloatingPointError as exc:
        raise ValueError("Spectral rescaling overflow/underflow") from exc
    if not np.isfinite(values).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite spectrum/basis")
    normalized_s = normalized_s / (s[0] if s[0] > 0 else 1.0)
    return values, normalized_s, v, scale


def _prefix_distance(a: Array, b: Array, count: int) -> Array:
    if count == 0:
        return np.empty(0)
    # ||P_A-P_B||F/sqrt(2k), computed for all prefixes withoutp projectors.
    overlap = (a @ b.T) ** 2
    prefix = np.cumsum(np.cumsum(overlap, axis=0), axis=1)
    k = np.arange(1, count + 1, dtype=np.float64)
    deficit = 1 - np.diag(prefix)[:count] / k
    tolerance = 64 * np.finfo(float).eps * max(a.shape[1], a.shape[0])
    if np.any(deficit < -tolerance) or np.any(deficit > 1 + tolerance):
        raise ValueError("Prefix projector arithmetic outside numerical bounds")
    return np.asarray(np.sqrt(np.clip(deficit, 0, 1)), dtype=np.float64)


def select(fine: Array, coarse: Array, richardson: Array) -> dict[str, Any]:
    """Return all3fixed hypotheses; privileged evaluation happens elsewhere."""
    arrays = (fine, coarse, richardson)
    results = [_svd(a) for a in arrays]
    if any(a.shape != fine.shape for a in arrays):
        raise ValueError("Matching observation views required")
    rows, p = fine.shape
    q = min(rows, p)
    count = min(q, p - 1)
    floor = RESOLUTION_FACTOR * np.finfo(float).eps * max(rows, p)
    gaps = np.zeros((3, count))
    ratios = np.ones((3, count))
    eligible = np.zeros((3, count), dtype=np.bool_)
    clipped = np.zeros((3, count), dtype=np.bool_)
    for i, (_, s, _, _) in enumerate(results):
        if count:
            top = s[:count]
            bottom = s[1 : count + 1]
            eligible[i] = top > floor
            clipped[i] = bottom < floor
            ratios[i] = np.maximum(top, floor) / np.maximum(bottom, floor)
            gaps[i] = np.log(ratios[i])
    stability = np.maximum.reduce(
        [
            _prefix_distance(results[i][2], results[j][2], count)
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ]
    )
    rich_s = results[2][1]
    strict = (
        int(np.count_nonzero(rich_s > STRICT_RTOL * rich_s[0])) if rich_s[0] > 0 else 0
    )
    gap_index = choose(gaps[2], eligible[2] & (ratios[2] >= MIN_GAP))
    joint_scores = np.min(gaps, axis=0)
    joint_mask = (
        np.all(eligible, axis=0)
        & np.all(ratios >= MIN_GAP, axis=0)
        & (stability <= MAX_STABILITY)
    )
    stable_index = choose(joint_scores, joint_mask)

    def candidate(index: int | None, method: str) -> dict[str, Any]:
        if index is None:
            return dict(
                rank=None,
                certificate=False,
                status="no_candidate",
                reason="no_decisive_or_stable_gap",
                next_resolution_clipped=None,
            )
        return dict(
            rank=index + 1,
            certificate=False,
            status="heuristic_only",
            reason=None,
            next_resolution_clipped=bool(clipped[2, index])
            if method == "gap"
            else bool(np.any(clipped[:, index])),
        )

    return dict(
        rows=rows,
        p=p,
        certificate=False,
        choices=dict(
            strict=dict(
                rank=strict,
                certificate=False,
                status="heuristic_only"
                if 0 < strict < p
                else ("zero_observations" if strict == 0 else "observed_full_space"),
                observed_membership_domain="full_space"
                if strict == p
                else ("zero_space" if strict == 0 else "proper_subspace"),
            ),
            gap=candidate(gap_index, "gap"),
            stable=candidate(stable_index, "stable"),
        ),
        spectra={
            name: results[i][0]
            for i, name in enumerate(("fine", "coarse", "richardson"))
        },
        bases={
            name: results[i][2]
            for i, name in enumerate(("fine", "coarse", "richardson"))
        },
        diagnostics=dict(
            log_gaps=gaps,
            gap_ratios=ratios,
            eligible=eligible,
            resolution_clipped=clipped,
            stability=stability,
            resolution_floor=floor,
        ),
        warning="No model/error bound: stable gaps can discard a full-rank signal.",
    )
