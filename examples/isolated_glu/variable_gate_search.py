"""Pure observations-only candidate search; no experiment launcher or oracle.

Dimension-adapted fixed GLU32 initial search and rescue, with injected PCG64
streams and full trace. A target count is a family hypothesis, not coverage.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Callable, Protocol, cast
import numpy as np
from numpy.typing import NDArray
from modelforge import glu32
from variable_gate_view import GateJob

Array = NDArray[np.float64]


class TraceSink(Protocol):
    def write(self, name: str, arrays: dict[str, Array]) -> None: ...


@dataclass(frozen=True, slots=True)
class GateResult:
    status: str
    candidates: Array
    report: dict[str, Any]


def _finite(a: Array) -> None:
    if (
        not isinstance(a, np.ndarray)
        or a.dtype != np.float64
        or not np.isfinite(a).all()
    ):
        raise FloatingPointError("Nonfinite or non-native derived array")


def _units(a: Array, d: int) -> None:
    _finite(a)
    if (
        a.ndim != 2
        or a.shape[1] != d
        or not np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-10, rtol=0)
    ):
        raise FloatingPointError("Nonunit candidate directions")


def _normalized(a: Array) -> Array:
    _finite(a)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        norm = np.linalg.norm(a, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise FloatingPointError("Zero normalization")
        out = np.asarray(a / norm, dtype=np.float64)
    _finite(out)
    return out


def _emit(sink: TraceSink, name: str, **arrays: Array) -> None:
    frozen = {}
    for k, a in arrays.items():
        _finite(a)
        if not a.size:
            raise ValueError("Trace fields must be nonempty; pad with count")
        frozen[k] = np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)
    sink.write(name, frozen)


def _rows(a: Array) -> Array:
    return a if len(a) else np.zeros((1, *a.shape[1:]), dtype=np.float64)


def _states(
    initial: np.random.Generator, rescue: np.random.Generator
) -> dict[str, Array]:
    result = {}
    for name, rng in [("initial", initial), ("rescue", rescue)]:
        data = json.dumps(
            rng.bit_generator.state,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        if len(data) > 512:
            raise ValueError("PCG64 state exceeds fixed trace capacity")
        result[name] = np.pad(
            np.frombuffer(data, dtype=np.uint8).astype(np.float64), (0, 512 - len(data))
        )
        result[name + "_length"] = np.array([len(data)], dtype=np.float64)
    return result


def losses(directions: Array, basis: Array) -> Array:
    _finite(directions)
    _finite(basis)
    if (
        directions.ndim != 2
        or basis.ndim != 2
        or directions.shape[1] * (directions.shape[1] + 1) // 2 != basis.shape[1]
    ):
        raise ValueError("Matching symmetric coordinates required")
    q = glu32.pack(directions[:, :, None] * directions[:, None, :])
    residual = q - (q @ basis.T) @ basis
    values = np.asarray(np.sum(residual * residual, axis=1), dtype=np.float64)
    _finite(values)
    return values


def merge_candidates(
    known: Array, candidates: Array, error: Array
) -> tuple[Array, dict[str, Array]]:
    _units(candidates, candidates.shape[1])
    _units(known, candidates.shape[1])
    _finite(error)
    if error.shape != (len(candidates),) or np.any(error < 0):
        raise ValueError("Matching nonnegative losses required")
    order = np.argsort(error)
    decision = np.zeros(len(candidates), dtype=np.float64)
    maxcos = np.zeros(len(candidates), dtype=np.float64)
    nearest = np.full(len(candidates), -1.0, dtype=np.float64)
    selected = list(known)
    for k in order:
        if selected:
            cs = np.abs(np.stack(selected) @ candidates[k])
            nearest[k] = int(np.argmax(cs))
            maxcos[k] = float(np.max(cs))
        if error[k] <= 1e-8:
            if not selected or maxcos[k] < 0.9995:
                selected.append(candidates[k])
                decision[k] = 2
            else:
                decision[k] = 1
    out = (
        np.stack(selected)
        if selected
        else np.empty((0, candidates.shape[1]), dtype=np.float64)
    )
    return out, dict(
        order=order.astype(np.float64),
        decision=decision,
        max_cosine=maxcos,
        nearest_index=nearest,
        losses=error.copy(),
    )


def _adam(starts: Array, basis: Array, *, steps: int = 1600) -> Array:
    """Dimension adaptation of frozen directions' Adam; test seam can shorten steps."""
    if type(steps) is not int or not 1 <= steps <= 1600:
        raise ValueError("Bounded integer Adam steps")
    v = starts.copy()
    d = v.shape[1]
    _units(v, d)
    B = np.stack([glu32.unpack(b, d) for b in basis])
    rank = len(basis)
    adam_m = np.zeros_like(v)
    adam_v = np.zeros_like(v)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for t in range(1, steps + 1):
            vv = v[:, :, None] * v[:, None, :]
            coef = vv.reshape(len(v), -1) @ B.reshape(rank, -1).T
            proj = (coef @ B.reshape(rank, -1)).reshape(-1, d, d)
            grad = -4 * np.einsum("bij,bj->bi", proj, v)
            grad -= np.sum(grad * v, axis=1, keepdims=True) * v
            adam_m = 0.9 * adam_m + 0.1 * grad
            adam_v = 0.999 * adam_v + 0.001 * grad * grad
            v -= (
                0.02
                * (adam_m / (1 - 0.9**t))
                / (np.sqrt(adam_v / (1 - 0.999**t)) + 1e-8)
            )
            v = _normalized(v)
    return v


def _refine(x: Array, basis: Array, steps: int) -> Array:
    y, _, _ = glu32.refine(x.copy(), basis, steps=steps)
    _units(y[None, :], len(x))
    return cast(Array, y)


def _initial(
    job: GateJob, rng: np.random.Generator, sink: TraceSink
) -> tuple[Array, Array]:
    raw = rng.normal(size=(256, job.n))
    _emit(sink, job.job_id + "-initial-raw", raw=raw)
    starts = _normalized(raw)
    _emit(sink, job.job_id + "-initial-starts", starts=starts)
    return starts, _adam(starts, job.basis)


def _rescue(
    known: Array, basis: Array, rng: np.random.Generator, sink: TraceSink, prefix: str
) -> tuple[Array | None, dict[str, Any]]:
    residual, geometry = glu32.residual_subspace(known, basis)
    _finite(residual)
    _emit(
        sink,
        prefix + "-geometry",
        known=known,
        residual=_rows(residual),
        geometry=np.array(
            [
                geometry["component_rank"],
                geometry["residual_dimension"],
                geometry["smallest_component_singular"],
            ],
            dtype=np.float64,
        ),
        tangent_singular=np.asarray(
            geometry["known_tangent_smallest_three"], dtype=np.float64
        ),
    )
    if geometry["component_rank"] != 2 * len(known) or len(residual) != len(
        basis
    ) - 2 * len(known):
        return None, geometry
    d = known.shape[1]
    raw: list[Array] = []
    coeffs: list[Array] = []
    eigenvalues: list[Array] = []
    eigenvectors: list[Array] = []
    indices: list[Array] = []
    starts: list[Array] = []
    # Fixed chunks preserve raw inputs before eigensolves. An external kill may
    # leave up to8 completed-but-uncommitted outputs unknown; never infer them.
    for start in range(0, 256, 8):
        suffix = f"{start // 8:03d}"
        _emit(
            sink,
            prefix + "-mix-intent" + suffix,
            index=np.array([start], dtype=np.float64),
            requested=np.array([8.0], dtype=np.float64),
        )
        chunk_raw = np.stack([rng.normal(size=len(residual)) for _ in range(8)])
        chunk_coeff = _normalized(chunk_raw)
        _emit(
            sink,
            prefix + "-mix-pre" + suffix,
            raw=chunk_raw,
            normalized=chunk_coeff,
            index=np.array([start], dtype=np.float64),
        )
        vals = np.zeros((8, d))
        vecs = np.zeros((8, d, d))
        ixs = np.zeros((8, 2))
        out = np.zeros((16, d))
        completed = 0
        attempted = 0

        def flush(complete: bool) -> None:
            _emit(
                sink,
                prefix + "-eigen-chunk" + suffix,
                values=vals,
                vectors=vecs,
                indices=ixs,
                starts=out,
                count=np.array([completed], dtype=np.float64),
                attempted=np.array([attempted], dtype=np.float64),
                complete=np.array([float(complete)]),
                index=np.array([start], dtype=np.float64),
            )

        try:
            for k in range(8):
                attempted = k + 1
                matrix = glu32.unpack(chunk_coeff[k] @ residual, d)
                eig, vectors = np.linalg.eigh(matrix)
                _finite(eig)
                _finite(vectors)
                chosen = np.argsort(np.abs(eig))[-2:]
                vals[k] = eig
                vecs[k] = vectors
                ixs[k] = chosen
                out[2 * k : 2 * k + 2] = vectors[:, chosen].T
                completed = k + 1
        except Exception as exc:
            try:
                flush(False)
            except Exception as flush_error:
                exc.add_note(
                    "Partial eigen trace flush failed: " + type(flush_error).__name__
                )
            raise
        flush(True)
        raw.extend(chunk_raw)
        coeffs.extend(chunk_coeff)
        eigenvalues.extend(vals)
        eigenvectors.extend(vecs)
        indices.extend(ixs)
        starts.extend(out)
    _emit(sink, prefix + "-mixtures", raw=np.stack(raw), normalized=np.stack(coeffs))
    _emit(
        sink,
        prefix + "-eigen",
        values=np.stack(eigenvalues),
        vectors=np.stack(eigenvectors),
        indices=np.stack(indices),
    )
    return np.stack(starts), geometry


def _batch_refine(
    inputs: Array,
    basis: Array,
    steps: int,
    sink: TraceSink,
    prefix: str,
    refiner: Refiner,
) -> Array:
    """Save every completed8-row chunk; flush ordinary exceptions without masking.

    Hard kill during a chunk leaves up to8 uncommitted outputs unknown. Inputs
    are already persisted by the caller; a marker/outer controller owns failure.
    """
    result = np.zeros_like(inputs)
    for start in range(0, len(inputs), 8):
        requested = min(8, len(inputs) - start)
        completed = 0
        attempted = 0
        out = np.zeros((8, inputs.shape[1]))
        error = np.zeros(8)
        suffix = f"{start // 8:03d}"
        _emit(
            sink,
            prefix + "-refine-intent" + suffix,
            index=np.array([start], dtype=np.float64),
            requested=np.array([requested], dtype=np.float64),
        )

        def flush(complete: bool) -> None:
            _emit(
                sink,
                prefix + "-refine-chunk" + suffix,
                refined=out,
                losses=error,
                count=np.array([completed], dtype=np.float64),
                attempted=np.array([attempted], dtype=np.float64),
                complete=np.array([float(complete)]),
                index=np.array([start], dtype=np.float64),
            )

        try:
            for k in range(requested):
                attempted = k + 1
                value = refiner(inputs[start + k].copy(), basis, steps)
                _units(value[None, :], inputs.shape[1])
                loss = losses(value[None, :], basis)[0]
                out[k] = value
                error[k] = loss
                completed = k + 1
        except Exception as exc:
            try:
                flush(False)
            except Exception as flush_error:
                exc.add_note(
                    "Partial refinement trace flush failed: "
                    + type(flush_error).__name__
                )
            raise
        flush(True)
        result[start : start + requested] = out[:requested]
    return result


Initializer = Callable[[GateJob, np.random.Generator, TraceSink], tuple[Array, Array]]
Rescuer = Callable[
    [Array, Array, np.random.Generator, TraceSink, str],
    tuple[Array | None, dict[str, Any]],
]
Refiner = Callable[[Array, Array, int], Array]


def _run(
    job: GateJob,
    initial_rng: np.random.Generator,
    rescue_rng: np.random.Generator,
    sink: TraceSink,
    *,
    initializer: Initializer,
    rescuer: Rescuer,
    refiner: Refiner,
) -> GateResult:
    if not isinstance(job, GateJob):
        raise ValueError("Validated immutable GateJob required")
    if (
        not isinstance(initial_rng, np.random.Generator)
        or not isinstance(rescue_rng, np.random.Generator)
        or initial_rng is rescue_rng
        or initial_rng.bit_generator is rescue_rng.bit_generator
        or type(initial_rng.bit_generator) is not np.random.PCG64
        or type(rescue_rng.bit_generator) is not np.random.PCG64
        or initial_rng.bit_generator.state == rescue_rng.bit_generator.state
    ):
        raise ValueError("Distinct injected PCG64 streams required")
    target = job.rank // 2
    prefix = job.job_id
    b = job.basis
    before_initial = initial_rng.bit_generator.state
    before_rescue = rescue_rng.bit_generator.state
    _emit(sink, prefix + "-rng-before", **_states(initial_rng, rescue_rng))
    starts, optimized = initializer(job, initial_rng, sink)
    _units(starts, job.n)
    _units(optimized, job.n)
    if starts.shape != optimized.shape or not 1 <= len(starts) <= 256:
        raise ValueError("Bounded matching initial arrays required")
    _emit(
        sink,
        prefix + "-initial-pre-refine",
        starts=starts,
        optimized=optimized,
        before_losses=losses(optimized, b),
    )
    refined = _batch_refine(optimized, b, 12, sink, prefix + "-initial", refiner)
    _units(refined, job.n)
    err = losses(refined, b)
    _emit(
        sink,
        prefix + "-initial-refine",
        starts=starts,
        optimized=optimized,
        refined=refined,
        before_losses=losses(optimized, b),
        losses=err,
    )
    known, trace = merge_candidates(np.empty((0, job.n)), refined, err)
    _emit(sink, prefix + "-initial-merge", **trace)
    base_count = len(known)
    rounds = []
    status = (
        "zero_candidates"
        if not len(known)
        else "overshoot"
        if len(known) > target
        else "candidate_target"
        if len(known) == target
        else "rounds_exhausted"
    )
    if 0 < len(known) < target:
        for i in range(4):
            rp = prefix + f"-r{i}"
            _emit(sink, rp + "-input", known=known)
            rescue_starts, geometry = rescuer(known, b, rescue_rng, sink, rp)
            if rescue_starts is None:
                status = "unexpected_component_geometry"
                rounds.append(
                    dict(
                        round=i,
                        status=status,
                        input_count=len(known),
                        geometry=geometry,
                    )
                )
                break
            starts = rescue_starts
            _units(starts, job.n)
            if not 1 <= len(starts) <= 512:
                raise ValueError("Bounded rescue starts required")
            _emit(
                sink, rp + "-pre-refine", starts=starts, before_losses=losses(starts, b)
            )
            refined = _batch_refine(starts, b, 24, sink, rp, refiner)
            _units(refined, job.n)
            err = losses(refined, b)
            _emit(
                sink,
                rp + "-refine",
                starts=starts,
                refined=refined,
                before_losses=losses(starts, b),
                losses=err,
            )
            count = len(known)
            known, trace = merge_candidates(known, refined, err)
            _emit(sink, rp + "-merge", **trace)
            rounds.append(
                dict(
                    round=i,
                    input_count=count,
                    retained_count=len(known),
                    geometry=geometry,
                )
            )
            if len(known) >= target:
                status = "candidate_target" if len(known) == target else "overshoot"
                break
    final_loss = losses(known, b)
    cos = np.abs(known @ known.T)
    if len(known):
        np.fill_diagonal(cos, 0)
        if np.any(final_loss > 1e-8) or np.any(cos >= 0.9995):
            raise FloatingPointError("Final candidate geometry inconsistent")
    _emit(
        sink,
        prefix + "-final",
        directions=_rows(known),
        losses=_rows(final_loss),
        cosines=cos if len(known) else np.zeros((1, 1), dtype=np.float64),
        count=np.array([len(known)], dtype=np.float64),
    )
    _emit(sink, prefix + "-rng-after", **_states(initial_rng, rescue_rng))
    result = np.frombuffer(known.tobytes(), dtype=np.float64).reshape(known.shape)
    return GateResult(
        status,
        result,
        dict(
            job_id=prefix,
            target_hypothesis=target,
            base_count=base_count,
            final_count=len(known),
            rounds=rounds,
            certificate=False,
            coverage_certified=False,
            private_gate="closed",
            rng_before=dict(initial=before_initial, rescue=before_rescue),
            rng_after=dict(
                initial=initial_rng.bit_generator.state,
                rescue=rescue_rng.bit_generator.state,
            ),
            max_retained_loss=float(final_loss.max()) if len(known) else None,
            warning="Counts and low loss do not establish gate identity or full-function reconstruction",
        ),
    )


def search_job(
    job: GateJob,
    initial_rng: np.random.Generator,
    rescue_rng: np.random.Generator,
    sink: TraceSink,
) -> GateResult:
    """Fixed production constants, with no public override or hidden seed creation."""
    return _run(
        job,
        initial_rng,
        rescue_rng,
        sink,
        initializer=_initial,
        rescuer=_rescue,
        refiner=_refine,
    )


def _search_fixture(
    job: GateJob,
    initial_rng: np.random.Generator,
    rescue_rng: np.random.Generator,
    sink: TraceSink,
    *,
    initializer: Initializer,
    refiner: Refiner,
    rescuer: Rescuer = _rescue,
) -> GateResult:
    """Explicit private test seam; never called by the registered controller."""
    return _run(
        job,
        initial_rng,
        rescue_rng,
        sink,
        initializer=initializer,
        rescuer=rescuer,
        refiner=refiner,
    )
