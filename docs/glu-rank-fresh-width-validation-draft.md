# Distinct owned variable-width rank validation — DRAFT, NOT executable

The previous zero-query PUBLIC analysis is now independently audited: gap/stable match numerical reference ranks in12 nonfull cases and abstain in4 full cases. Every nonnull result equals the existing known-width control. It neither improves the failed FP32 full-map candidate nor demonstrates unknown-width generalization. See [audited outcome](glu-noise-aware-rank-analysis-outcome.md).

This proposal tests a necessary broader-domain question: do the EXACT frozen observation-only rules retain useful Hessian subspaces when width is no longer a known2*n relationship, including degeneracy/fullspace challenges? It is still isolated bias-free SwiGLU access, not internal-block access from a final-model API. No new teacher, RNG seed, probe or query has been reserved/generated. Full protocol/source/resource approval is required BEFORE data; this draft grants none. Private gate stays CLOSED.

## Fixed candidate scope proposed before fresh data

Reuse byte-frozen rank_selection.py (source hash1382f141fd4adf7a80106df75ccedfba4b55b9dd2b877ba4399becc4646f870b): strict1e-10, gap ratio10, cross-view RMS sine.01, arithmetic floor64eps*max(rows,p), exact computed-score tie rule, same null/certificatefalse semantics. No candidate is declared intrinsically identifiable from a gap. Both gap and stable remain reported; no choose-the-winning-rule after labels. The generic factory must be distinct from existing width=2*n saved-data loaders and must not modify signed sources.

Proposed finite challenge family (not a distribution-wide benchmark):

|Input/output n|Three owned widths|Challenge types|
|---:|---|---|
|4|2,4,8|generic/generic/generic, last potentially full symmetric span|
|8|4,12,20|generic/generic/generic, last potentially full symmetric span|
|16|8,24,48|generic/exact duplicate gate-pair/near-parallel gate-pair|
|32|16,48,96|generic/generic/generic|

Twelve teachers, both FP64-compute and FP32-compute channels,24cases. Proposed16 probes per teacher, probe std4, weight std.2, two unchanged finite steps.3/.6, sign-major4p rows per probe/step without dedup; all full output vectors persisted. For each duplicate case, final gate and up-factor rows exactly copy the first pair; output columns remain independently drawn. This admits a smaller equivalent parameterization, so neither numeric rank nor requested width is an original-width certificate. Proposed near-parallel pair differs by1e-6 times a separately drawn standard-normal perturbation; exact perturbation/scaling/compute-rounding rules must be frozen in executable source first. Other factors remain ordinary draws. Shared public hyperparameters are known; per-case width/type assignment and generation seeds are committed by independent verifier and withheld from estimator until ALLchoices are sealed. No existing consumed684xxx/941xxxseed may be reused.

This reduced16-probe design is intentional query-cost investigation, not an assertion that it preserves the old64-probe result. At n32, packed matrix has512rows and528columns; row-limited structural-zero clipping must remain explicit. Exact planned vector count if this design is frozen:6cases per n *16probes *2steps *4p =545,280vectors,768callback batches. Counts are reservations/attempts/returns/persistence separately; no automatic retries. No expensive synthetic worker should generate the registered family as a preflight.

## Provisional accept/reject contract to freeze BEFORE generation

The scientific endpoint is bounded finite-case subspace utility; it is NOT full-function reconstruction. The candidate rules always return certificatefalse. Predeclare both rules' outcomes and evaluate each independently on every case; never promote whichever wins. Known/half/double width controls are evaluator-only, with requested/effective/clipped collisions preserved.

Proposed rule-level pass requires ALL cases to meet their applicable conditions, including hard cases, with no post-label exclusions:

- Where analytic numerical reference has0<r<p: chosen rank equals r, unweighted lost/extra reference-subspace RMS<=.01, maximum truegate residual<=.01, and minimum of the64 independent random direction residuals exceeds that maximum. Exact/near-degenerate cases use measured reference rank, never force2m equality. Report any weak-spectrum numerical uncertainty; it cannot be relabeled as a pass.
- Where analytic numerical reference is fullp: return null. A proper subspace with a large gap must not be called informative on truth-full cases. Full reference membership is uninformative; passing this finite test is not a general fullspace detector theorem.
- Zero reference, nonfinite, underflow/overflow, source/resource/inventory/transport failure: explicit failure/abstention diagnostics, never success or exclusion. A resource/controller failure dominates metrics. Any unexpected valid zero-reference case counts as endpoint failure rather than changing the denominator.

Report all counts/errors, not just pass/fail. Numerical analytic rank is approximate1e-10 reference convention, not exact rank. Also retain singular spectra and per-component signal-energy diagnostics so geometric loss of a weak direction is visible; such auxiliary metrics cannot rescue a failed frozen criterion. If either rule fails, preserve that failure without tuning or repeated sealed evaluation. Even a pass only supports these finite owned challenges; factor extraction, full-map heldout prediction and a separate backend contract remain necessary. Existing941033 failed.001 functional criterion and candidate7f7d are unchanged.

## Required engineering and evidence before any data

1. Independently review mathematical/data contract, fixed seed commitments/assignment policy, exact counts and per-array/global byte inventory. Independent teacher assignment must not be readable by the estimator via metadata, filenames or parsed reports. Source-level view isolation is not an OS sandbox or protection against a malicious process; do not overclaim blinding.
2. New bounded source/loader/runner for variable width, two compute channels and16probes, synthetic literal tests for widths independent of n, duplicate/near-parallel/full/row-limited cases, wrongmetadata/nulls, no privileged view leakage, partial response/IO/source mutations and all query counters. Do not modify prior signed math/store/selection or consumed markers.
3. ONE global exclusive marker, source/policy/seed/config commitments, clean frozen discovery-lab and independent full-source approval. Pre-data actual free disk, monitored worst-shape archives and TREE availability; shared180s/512MiBpolledTREE/BLAS1/2KiBstdout; every source/query input before callback and raw return before derivatives. Exact total/shard/report caps must be proven rather than copied from a different width/probe family.
4. Independent verifier generates fresh owned data only after gates. Estimator gets permitted FD matrices and n/output/step/probe metadata only, seals ALL24choices before teacher assignment/context/reference is read for evaluation. Persist every outcome/control and source/input/output commitments, then terminal checkpoint BEFORE independent saved-output audit. Audit successful bounded math once with working TREE preflight; preserve failed receipts, diagnose before retry.

This draft is not protocol completion and creates no new reconstruction route. Bias support, Hessian-vector access/cost, final-output-to-internal-block reduction, full decoder backbone and universal reconstruction remain unresolved; the private final-model prerequisite is unmet.
