# Isolated GLU candidate coverage — calibration-only development v1

The frozen v2 experiment remains FAILED overall (63/64/64); no teacher or
heldout evaluation from it was repeated. This distinct development experiment
uses only teacher0's saved public `hessians.npz` and `refined-solution.npz`.
No new oracle calls, private arrays, or final-model assets were used.

## Hypothesis and limitations

For a candidate unit direction v, form an orthonormal Frobenius basis for
T_v={vaᵀ+avᵀ}. Find its two least-residual components against the original
rank128 Hessian subspace V. Project these into V for each of63 input directions,
then remove their span to obtain initialization matrices. Search64 fixed angles
of the resulting2D residual, take two extremal absolute-eigenvalue eigenvectors
per angle, and refine128 starts for24 tangent Gauss–Newton steps on ORIGINAL V.
The separate source is `examples/isolated_glu/coverage.py`; the frozen v2
solvers/wrapper/protocol and original outcome files were not changed.

This is an implementation of a residual-rescue research direction, not a claim
to invent the general idea: see [Shi/Fu's isolated GLU paper](https://arxiv.org/html/2608.06631v1).
Its isolated continuous-query access is stronger than a final text API.

The independent synthetic review exposed important failure modes:

- A single tangent singular minimum vanishes on an entire wrong-direction line.
  Inspect both smallest values and the gap to the third.
- The rank-one objective is quartically flat along a value-direction perturbation.
- Orthogonal deflation does not preserve rank-one matrices.
- Residual dimension2/energy√2 after projecting126 components into V128 is
  structural, even for wrong directions. It is not evidence of correct coverage.

## Actual development result

A single bounded run retained64 directions from63 saved inputs; it found one
additional nonduplicate candidate with original-subspace squared residual
4.700318187055265e-17 and max absolute cosine0.4138916754849132 to the input63.
Its tangent singular values (smallest first) were6.85586e-9,1.63726e-7,0.676074.
These show calibration consistency, not identity with the missing teacher gate.

Controller time1.58697s, sampled peak RSS100171776bytes (~95.53MiB),24samples,
process_and_children monitor,512MiB/180s cap. Periodic samples are not a proof
of the true instantaneous peak. Only this development run was executed.
Three synthetic geometry/refinement/control tests pass; strict mypy and Ruff
pass for the new script. No native packages rebuilt and no broad CI claim.

The actual teacher direction, coupling, orientation, functional fidelity,
generalization and exact stored weights remain unverified for this candidate.
Finding64 is not a coverage certificate. Even a later positive fresh isolated
block evaluation would not demonstrate universal or whole-Transformer recovery.
The private final-model gate remains CLOSED.

## Next action

Freeze a DISTINCT v3 protocol and its executable source/readiness before drawing
new owned teachers. Keep v1 engineering failure and v2 scientific failure visible.
The draft below is not runnable authorization/readiness. Do not evaluate the old
teacher0 heldout to tune this development rescue or reuse consumed labels.
Evidence and independent math source/results are in
`experiments/evidence/isolated-glu-coverage-development-v1/`.
