# Bounded guarded gate-vector step (PUBLIC development)

## Protocol before execution

This is a distinct experiment following the failed four-sweep fixed-gate study, not an uncapped continuation. The starting candidate SHA256 is `3fa13a8f6337bfc9d317a95dde95cd0d87ab6ae04568a392bf6b227a517f40b3`. No quality improvement is claimed before actual execution and independent outcome review.

The fitter reads only the hash-committed candidate, preceding fit report, original public calibration manifest and saved1024train+1024guard physical FP32 responses. The reused observations hash is `c8414e9215ac3e14967c23bccf4f0240efe83c6265b5339398f1851b074be210`. These are adaptively reused calibration, not heldout data. No new calibration/teacher-gradient/teacher-weight access is used. Training residual multipliers remain the inverse original group response norms, with physical targets weighted once.

Full gate vectors may change direction and magnitude; fixed finite/nonzero rank-one couplings remain byte-identical. For weighted residual R=w(Y-f), the known-candidate gate gradient is d=-2J*R and K=J*J. With cost sum(R²), delta0=-.5 solve(K,d). Strict eigenvalue min>1e-12max>0 is required, without ridge/pseudoinverse/skipping. Delta is clipped to5%of current gate norm, with an explicit zero branch. The score -(d.T delta+delta.T K delta) is a predicted LINEARIZED decrease, not a measured exact decrease.

Screen64units once, rank descending with original-index ties, and select at most8scores above the cost margin. Recompute direction for each selected unit in that fixed order once. Try only1,1/2,...,1/128; accept the first trial that reduces complete exact training cost by more than max(1e-12oldcost,1e-24), while EVERY guard group's relative error stays within both current and initial +1e-10. Every guard group's maximum absolute error must also stay within both current and initial +1e-10max(1,initial_group_maxabs). Maximum72linear solves and64trials; no extra units/passes or evaluation-driven retries.

Per-unit prediction deltas avoid a full large Jacobian. Initial/final complete forwards check cached predictions independently. Complete terminal runs save all screen/sequential gradients, Grams/eigenspectra, original and clipped directions, gates/scores/order, trial gate rows/decisions, raw/weighted targets, weights/predictions/residuals and final gate norms outside Git. A singular/invalid intermediate stage fails closed: no candidate/evaluation; the current exception path saves the error but may lack partial proposal arrays. Such a failure cannot support a claim that the whole scan was verified.

A changed finite candidate alone receives prospective PUBLIC seed941032:2048FP32 reference vectors,512per scale.01/1/10/100, same inputs/references for incumbent and candidate. Candidate/incumbent digests are checked BEFORE querying. Acceptance remains all four relative Frobenius errors<=.001; report all relative and maximum-absolute regressions. This is a new public development set, not sealed proof. Seeds941027..941031stay consumed and unread. No changed candidate means no final evaluation.

Controller:180s total,512MiB polled worker tree, BLAS1. Sampled RSS can miss peaks and must not be represented as an exact peak. An exclusive attempt marker prevents automatic retries. Source/protocol/controller must pass independent review and be committed clean before actual use. The reviewed controller is also preserved with evidence.8focused synthetic tests, Ruff and strict mypy are the software validation scope; successful unrelated/native checks are not repeated.

This isolated owned NumPy GLU32/64/32 experiment does not establish original-weight recovery, full Transformer reconstruction, production FP32 certification, convergence, a gate-error floor, universal reconstruction or frontier capability. Private final-model gate CLOSED. Stable releases unchanged.

## Execution and outcome

Pending source freeze and the one registered run. Outcome must be filled from preserved receipts and independent audit, including negative outcomes.
