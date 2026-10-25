# Synthetic known-candidate guard metric normals

This is a derivative-mapping prerequisite, with no fitter, target query, candidate update or actual saved-data use. It does not repair previous quality failures. The original exact full-forward training/current+initial relative/maxabsolute guards remain mandatory. Private final-model gate CLOSED.

## Mathematical contract

Physical complete candidate error is e=f-Y, with Y unweighted. For one selected gate g_i and fixed coupling C_i, J_i(x)delta=silu'(x.g_i)*(x.delta)*(C_i x). The group squared-relative metric R=||e||²/||Y||² has gradient2J_i*e/||Y||². The derivative includes the full candidate residual, not only that unit's prediction. This is a derivative of known candidate computation; no target gradients or inaccessible parameters are observed.

For M=max_j|e_j|>0, each exact maximizing coordinate contributes the signed normal sign(e_j)J_j. The directional derivative of M is the maximum of these normal-dot-direction values, so selecting only one argmax can miss a worsening tied coordinate. The implementation additionally includes every coordinate with abs(e_j)>=M*(1-1e-12). Nonmaximal near-tie planes are a conservative inclusion policy, NOT exact active derivatives. They carry exact_maximum flags and signed-residual/row/output/group provenance in deterministic row-major order.

The local pointwise maximum of affine directional models follows the active-function calculus in [Boyd, Stanford subgradient notes](https://web.stanford.edu/class/ee392o/subgrad.pdf). This supports the local max-of-normals construction; it does NOT make the original nonlinear GLU objective convex or give a global subgradient bound. The derivative formula is also derived directly from a finite maximum and first-order expansions of its nonzero signed residual branches.

## Bounds, ties and degeneracy

metric_group accepts a single known gate/coupling and a supplied PHYSICAL full error/reference group. It cannot infer from values whether the caller has secretly preweighted Y/error; a future actual caller must bind raw calibration provenance. guard_normals computes e itself from the complete candidate, accepts exactly four equal-row groups and selects only one gate derivative. Known profiles are finite exactdtypefloat64, dimensions<=32, outputs<=32, units<=64, rows<=4096; group/offset/unit integers reject booleans and out-of-range values.

There are at most8LOGICAL plane records, counted BEFORE omission. Each group adds one relative plane plus ALL exact/near maximum coordinates. Four groups with unique maxima already use8records, so ANY additional exact/near tie rejects the full profile. This version supports tie-correct group derivatives and tie-safe full-profile rejection; it does NOT return a QP for tied four-group profiles. No silent truncation or deduplication changes the intended constraints.

Computed FP64 zero normals remain in the logical records with matrix_row=null and omitted_reason=computed_FP64_zero, but are omitted from A because0.delta<=0 is a redundant computed first-order inequality. This is not a claim of ideal mathematical zero or nonlinear finite-step retention. A nonzero normal whose norm underflows, or a normal/norm that overflows, fails closed. Logical zero records still consume the full budget.

M=0 is explicitly rejected: its directional derivative is max_j|J_j delta| and cannot be represented by sign(0)=0. Zero/overflowed reference squared norm is invalid. A nonzero physical residual whose squared norm underflows to0, or a positive squared error whose ratio to reference norm underflows to0, is rejected rather than reported as perfect relative error. Computed metrics/derivatives must be finite. Partial underflow/cancellation and general FP64 uncertainty still have no exact interval-error bound.

Current and initial retention bounds constrain the SAME current metric; they are not gradients evaluated at two separate candidate points. The returned normals represent current first-order nonincrease only. Inactive max coordinates may become active and nonlinear curvature may reverse an infinitesimal prediction at finite step, requiring exact full-forward trial checks on every original metric.

## Software validation

10NEWfocused controls passed: independent finite differences of squared-relative/unique-max metrics, full other-unit residual and selected-unit/group indexing, tied counterexample where one argmax admits a worsening direction, near-tie flags, negative response-rescaling invariance, zero normal provenance, zero maximum rejection, logical plane budget including zero/tied records, nonmutation, finite/profile/reference guards and two explicit metric-underflow regressions. Strict mypy on the new module and Ruff on new source/test passed. Existing successful experiments/tests/native packages were not rerun.

Independent complex-step/Jacobian source review is pending. No actual saved candidate/calibration/consumed evaluation arrays were read. Before real use, the mapping must pass independent review/source freeze and the actual calibration normal geometry must be examined under a separately preregistered READ-ONLY diagnostic. A later optimizer needs its own fixed budget/provenance/failure policy, full unchanged exact guards and prospective fresh paired evaluation. Consumed PUBLIC941027..941032cannot be used for tuning or represented as fresh data. Universal reconstruction remains open.
