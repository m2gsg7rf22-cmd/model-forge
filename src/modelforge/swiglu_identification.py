"""Truth-free SwiGLU gate identification from exposed sublayer input/output pairs.

Pipeline discovered in the 2026-10-22/23 backbone pilots (docs/RESEARCH_LOG.md):

1. Fit overparameterized SwiGLU models (width > true width) from several seeds;
   the output projection is eliminated by least squares (variable projection).
2. Keep gate rows reproduced across independent seeds (consensus) -- no teacher
   access is needed for this selection.
3. Freeze the deduplicated consensus gates and refit at the target width.

Gate rows have no scale gauge (silu is not homogeneous), so they are compared
directly; up/down rows are only identified as rank-one products. Results are
research evidence under O3 observations, not a closed-model extraction method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class FitResult:
    gate: Any
    up: Any
    down: Any
    relative_loss: float
    seconds: float


def _features(z: Any, gate: Any, up: Any) -> Any:
    import torch

    return torch.nn.functional.silu(z @ gate.T) * (z @ up.T)


def fit_swiglu(z: Any, y: Any, width: int, *, seed: int, seconds: float, frozen_gate: Any = None,
               init_scale: float = 0.3, max_rounds: int = 10_000) -> FitResult:
    """VarPro L-BFGS fit of y ≈ D(silu(G z) * U z); optional frozen leading gate rows."""
    import torch

    if z.ndim != 2 or y.ndim != 2 or len(z) != len(y):
        raise ValueError("z and y must be [rows, dim] with equal rows")
    dim = z.shape[1]
    n_frozen = 0 if frozen_gate is None else len(frozen_gate)
    if not 0 <= n_frozen <= width:
        raise ValueError("Frozen gates exceed the target width")
    gen = torch.Generator().manual_seed(seed)
    free_gate = (init_scale * torch.randn((width - n_frozen, dim), generator=gen, dtype=torch.float64)).requires_grad_()
    up = (init_scale * torch.randn((width, dim), generator=gen, dtype=torch.float64)).requires_grad_()
    fixed = torch.zeros((0, dim), dtype=torch.float64) if frozen_gate is None else frozen_gate.detach().double()
    norm = y.pow(2).mean()

    def gate() -> Any:
        return torch.cat([fixed, free_gate])

    def loss() -> Any:
        f = _features(z, gate(), up)
        d = torch.linalg.lstsq(f, y).solution.T
        return ((f @ d.T - y) ** 2).mean() / norm

    params = [up] + ([free_gate] if width > n_frozen else [])
    opt = torch.optim.LBFGS(params, lr=1, max_iter=200, tolerance_grad=1e-15, tolerance_change=1e-18,
                            history_size=100, line_search_fn="strong_wolfe")

    def closure() -> Any:
        opt.zero_grad()
        value = loss()
        value.backward()
        return value

    start = time.perf_counter()
    for _ in range(max_rounds):
        if time.perf_counter() - start >= seconds:
            break
        opt.step(closure)
    with torch.no_grad():
        g, u = gate().detach(), up.detach()
        f = _features(z, g, u)
        d = torch.linalg.lstsq(f, y).solution.T
        rel = float(((f @ d.T - y) ** 2).mean() / norm)
    return FitResult(g, u, d, rel, time.perf_counter() - start)


def consensus_gates(a: Any, b: Any, tolerance: float = 0.05) -> Any:
    """Rows of ``a`` reproduced by some row of ``b`` within relative distance, deduplicated."""
    import torch

    rel = torch.cdist(a, b) / a.norm(dim=1, keepdim=True)
    candidates = a[rel.min(1).values < tolerance]
    kept: list[int] = []
    for j in range(len(candidates)):
        if all(float((candidates[j] - candidates[q]).norm() / candidates[j].norm()) > tolerance for q in kept):
            kept.append(j)
    return candidates[kept]


def identify(z: Any, y: Any, width: int, *, over_width: int | None = None, seeds: tuple[int, ...] = (11, 23, 37),
             seconds: float = 600.0, tolerance: float = 0.05) -> dict[str, Any]:
    """Full truth-free pipeline; consensus = rows of the first fit reproduced by any other seed.

    Three seeds recovered more gates than two on the second pilot teacher (53 vs 41 of 64 within 5%).
    """
    import torch

    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for consensus")
    over = over_width or 3 * width
    fits = [fit_swiglu(z, y, over, seed=s, seconds=seconds) for s in seeds]
    others = torch.cat([f.gate for f in fits[1:]])
    consensus = consensus_gates(fits[0].gate, others, tolerance)[:width]
    final = fit_swiglu(z, y, width, seed=seeds[0] + 1000, seconds=seconds, frozen_gate=consensus)
    return {"result": final, "consensus_count": len(consensus), "overparameterized_losses": [f.relative_loss for f in fits]}


def match_gates(estimate: Any, truth: Any) -> dict[str, Any]:
    """Verifier-side comparison (needs the private teacher; never used during fitting)."""
    import torch

    d = torch.cdist(truth, estimate) / truth.norm(dim=1, keepdim=True)
    nearest = d.min(1).values
    owner = d.argmin(1)
    return {"within_1pct": int((nearest < 0.01).sum()), "within_5pct": int((nearest < 0.05).sum()),
            "true_units": len(truth), "bijective": len(set(owner.tolist())) == len(truth),
            "max_relative_error": float(nearest.max())}
