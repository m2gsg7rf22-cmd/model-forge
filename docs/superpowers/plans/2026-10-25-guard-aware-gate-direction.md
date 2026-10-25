# Guard-aware candidate gate directions — DRAFT, synthetic prerequisite only

The completed fixed-C raw Gauss–Newton gate step is not a convergence proof. Read the outcome audit and retain all quality failures. A new direction should account prospectively for calibration retention, rather than relaxing the final criterion or fitting to consumed PUBLIC941032 labels. Do not rerun the completed fit/evaluation.

## Narrow hypothesis

A clipped unconstrained training Gauss–Newton direction can improve training and relative calibration errors while increasing a currently largest absolute guard residual. Exact trial rejection then discards it. A direction satisfying selected first-order guard inequalities MAY permit an exact retained training improvement. This concerns local known-candidate derivatives and adaptive calibration, not target gradients, unseen generalization, original weights or universal reconstruction.

First implement only a bounded SYNTHETIC quadratic-program primitive: minimize d.T z + z.T K z subject to A z<=0, where K is finite strictly positive definite, dimension<=32, at most8finite nonzero constraint normals. Zero is feasible. Enumerate at most256active subsets; solve equality KKT systems on independent active rows, retain independently checked primal-feasible/dual-nonnegative/complementary/stationary proposals, and select the least objective with deterministic ties. Do not silently ridge/pseudoinvert, declare success without a numerical KKT certificate or confuse singular active-row subsets with an invalid full Gram. Explicitly support no constraints, redundant/dependent rows, zero objective/direction, all-directions-blocked and conditioning rejection. Prove which subsets can represent an optimum; define finite tolerances before implementation.

Only after finding an unconstrained-in-norm cone minimizer, radial scaling to5%gate norm preserves homogeneous cone feasibility; this is NOT an exact trust-region minimizer. Report pre/post clipping objective, norms, residuals and multipliers. A mathematically exact convex argument is distinct from a floatingpoint KKT-residual check. Derive/check all signs and the no-half cost convention independently.

Use hand-computable1D/2D controls, independently enumerated geometry/QR solutions and random feasible perturbations only as secondary checks. An independent agent must challenge KKT sufficiency, degeneracy, sign conventions, budgets and numerical claims. No actual saved candidate/calibration/teacher/consumed evaluation arrays or fitting in this prerequisite.

## Later experiment remains unregistered

Mapping relative/maxabsolute calibration metrics into selected constraint normals needs its own reviewed derivation, especially maxabs ties, group-normalization factors, inactive coordinates and the distinction between current/initial bounds. At most8chosen planes cannot be advertised as protecting every guard coordinate or nonlinear finite steps. Any later actual experiment MUST retain exact original complete current+initial relative AND maxabsolute trial checks, strict finite budgets, all rejected-trial evidence and a separately registered fresh evaluation (941032 is consumed). Existing labels must not select direction details or optimize candidates. No actual candidate update is authorized by this draft alone.

## Evidence transport requirement

The completed gate fit produced a3,095,641byte report and exceeded the2MB engine output plus strict metadata-reader bounds. Preserve that failure and its independently audited evaluation-only recovery; never refit to obtain a receipt. Future runners must write large numeric evidence outside Git, emit bounded compact JSON and have a synthetic large-evidence transport control before any actual fit. Original build memory peak remains unknown. Native packages/stable releases unchanged; private final-model gate CLOSED.

## Primary mathematical source

The constrained quadratic/KKT basis is standard convex optimization, not a new reconstruction theorem: [Boyd and Vandenberghe, Convex Optimization, official Stanford book](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), quadratic optimization and optimality conditions. The ModelForge-specific mapping from guard derivatives to a bounded experimental direction is a prospective hypothesis, not established by that source.
