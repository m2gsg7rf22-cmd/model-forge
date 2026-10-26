# Dimension/width observation-domain mathematics implementation plan

> **For agentic workers:** use superpowers:executing-plans inline; parent owns repository files, independent verifier writes only outside Git.

**Goal:** Implement independently testable pure mathematics for the saved dimension/width diagnostic, before a data runner exists.
**Architecture:** A new pure NumPy module receives explicit finite arrays. It performs no RNG, teacher creation, queries, fitting, file access or import-time work. A later separately reviewed runner/controller owns resource, transport, reservations and protocol execution.
**Tech Stack:** Python3.12+, NumPy, pytest, mypy, Ruff.
**Spec:** docs/glu-scaling-observation-domain-draft.md and its independent design review.

## Global constraints

Private gate CLOSED; consumed941027..941033 untouched. No registered teacher/probe creation or actual diagnostic execution. Existing signed source unchanged. Fixed rank tolerance1e-10, STRICT singular>tol*largest. svec upper triangular row-major, sqrt2 offdiagonal. Atom rows are UNNORMALIZED gg.T and gu.T+ug.T, interleaved per unit; right singular vectors span rows. Save full p singular values, padding structural zeros if rows<p. Gap null for zero/full rank or zero next value. No exact-rank/recovery claim.

## Review focus

Nonfinite inputs fail. Atom/stencil/projection underflow guards and a scaled residual norm prevent false zero geometry; contracted analytic FP64 references may contain computed zeros and are explicitly not exact arithmetic/rank truth. Asymmetric tensors are rejected. Empty/shape mismatches fail. Full/zero/deficient spaces do not manufacture gaps. Four-corner diagonal has effective2h and all four positions remain represented; no query implementation in this module.

## Task1: pure core (no shared external interfaces)

Files: examples/isolated_glu/scaling_domain_math.py; tests/test_glu_scaling_domain_math.py; docs/glu-scaling-domain-math-validation.md.
Interfaces: svec; symmetric_atoms; analytic_hessian; spectrum; direction_residuals; four_corner_inputs; curvature_from_responses; richardson. Explicit float64 arrays in, finite arrays/metrics out. No oracle callbacks.

- [ ] Write synthetic tests for Frobenius metric/order; hand-derived SiLU Hessian and independent finite differences; atom containment; deficient/full/zero spans; strict threshold and null gaps; normalized direction membership; four-corner polynomial stencil incl diagonal/coarse Richardson; shape/nonfinite/asymmetric/overflow/underflow failures.
- [ ] Observe missing module failure; implement minimum pure core; run targeted tests, strict mypy and Ruff.
- [ ] Independent read-only verifier challenges numerical/scientific scope with additional hand-sized synthetic tests outside Git; fix material findings with RED→GREEN evidence.
- [ ] Save source digests/test evidence and checkpoint via ResearchMemory; commit/push discovery-lab. The parent dimension protocol remains QUEUED until separately reviewed runner/controller/preregistration freeze. No actual teacher/query is authorized by this pure-core review.

## Follow-on work (not implemented by this plan)

Bounded sharded sequential diagnostic runner/controller, exact reservation/counters, artifact/disk/pre-data tree monitoring, worst-size synthetic transport, preregistration and source freeze; then one fixed owned public diagnostic and independent saved-raw audit. Bias/HVP/access reduction remain open. This subdivision avoids claiming an unreviewed transport is executable because its mathematical helpers pass.

Ruling: analytic contracted FP64 computed zeros remain allowed (e.g. ideal1e-400 unrepresentable), as the source already states. No global all-arithmetic underflow certification is claimed; atom/stencil/projection guards and stable residual norms are the enforceable contract. Cost if ignored: treating computed analytic rank as ideal identifiability evidence. Literal regression documents this boundary.
