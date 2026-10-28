# Variable-width gate candidate development — design only

Status: DRAFT, independent design review pending. No fitting, RNG, new data, or experiment is authorized by this document alone. Main reconstruction science remains open; private final-model gate CLOSED.

## Motivation and evidence

The independently audited paired acquisition v1 selects numeric reference ranks32/96/192 from17 probes in all six registered channel cases. These are three owned teachers, with dependent channels and prefixes, not six independent teachers. See [outcome](glu-row-acquisition-outcome.md). The earlier24case failure is preserved. Numeric subspace rank does not identify gate directions, minimal width, stored width, magnitudes, couplings, or full function.

The packaged `src/modelforge/glu32.py:directions` fixes input32/output32/width64, Hessian shape64x32x32x32, rank128 and gap100. Its successful fixed-profile v4 does not validate a variable-width solver. Its `refine`, `merge`, `search`, `pack` and `unpack` expose useful candidate primitives. They remain frozen; a distinct experimental module must call only explicitly pinned primitives or copy a reviewed adaptation.

The [existing coverage study](isolated-glu-coverage.md) already establishes quartic flatness along value-direction perturbations, structural tangent singular minima and count-versus-coverage ambiguity. Do not repeat that completed experiment. In the illustrative exact span of e1e1^T and sym(e1e2^T), for unit v=cos(theta)e1+sin(theta)e2, projection residual norm is sin(theta)^2 and squared loss sin(theta)^4. Thus loss1e-8 can allow angle about1e-2 radians in this example; it is not a general angle bound or recovery certificate. Finite-noise subspace proximity alone does not cure this instability.

## Approaches considered — maintainer choice

1. Generalize the existing bounded rank-one candidate search to a sealed observation-only basis. This directly tests the missing interface, reuses independently reviewed numerical primitives and gives interpretable failure evidence. It retains local optimization/coverage limitations.
2. Develop a new algebraic intersection/polynomial solver. This may avoid some optimization weaknesses but requires new conditioning theory, implementation and independent evidence. It is a later research option, not an asserted solution.
3. Resume full-function joint fitting immediately. Reject for this stage: it mixes gate, magnitude and coupling errors, needs additional observation types and makes a candidate-count failure harder to diagnose.

Choose1 as a development step only. No production backend expansion, certificates, or fresh success claims follow automatically.

## Observation contract and separation

Use only existing paired-v1 PUBLIC17-probe observations and the already sealed public estimator outputs. These now constitute disclosed development data. No teacher generation, forward callback, new queries, labels, truth context, seed, analytic reference, stored width, private assets, or consumed heldout are available to the optimizer. Physical/public paths, input manifests, choice seals, basis hashes and sources must be pinned. The optimizer accepts exactly a finite orthonormal basis with rank r, input dimension n, and anonymous case ID; it receives neither compute-channel hints nor a configured true width. A parent adapter may validate public schemas, never read sibling truth metadata.

Process six17probe cases in canonical order; do not inspect16prefixes for choosing parameters or a winning rule. Use the frozen stable selector's sealed basis by a choice fixed here, not because of an outcome comparison: cross-view agreement is the acquisition property being investigated. Gap is retained as prior diagnostic only. A later changed rule requires a distinct protocol, with the previous result preserved.

Check 0<r<p=n(n+1)/2, even r, finite orthonormal rows and an explicit generic bias-free nondegenerate isolated-SwiGLU hypothesis. Infer the candidate target r/2 solely as an experimental generic-family hypothesis. Odd/full/null rank must abstain. Even rank alone cannot establish this hypothesis. Duplicate or near-parallel units, other activation families, bias or structured noise can invalidate it. Never fall back to teacher width or silently clip candidates to the target.

## Proposed fixed search and report

Adapt the fixed-profile initial search to dimension n/rank r:256 unit starts,1600 fixed Adam steps with the same learning rate and moments,12 tangent refinement steps, original-basis loss threshold1e-8 and absolute-cosine merge threshold0.9995. One fixed seed domain per anonymous case, committed before any RNG; no seed retries. Target r/2 remains a stopping heuristic. Initial count0 or greater than target is a recorded failure, not truncation. For1..target-1, at most4 frozen512start multi-rescue rounds (256 mixtures, two eigenvectors each),24 refinement steps, with original-subspace geometry checks and no target truncation. Exact implementation must compare against the pinned fixed-profile code and document any difference before source review.

Persist all initial starts, pre/post directions and losses, every rescue input/mixture/eigenvector start/refinement/merge decision, component ranks, count trajectory, final candidates, pairwise absolute cosines and per-view projection residuals. Independently recompute all reports from saved arrays; optimizer labels are not trusted. A target-count match plus low loss is called candidate extraction only. Failure, resource stop, overshoot and abstention are first-class outcomes. No post-hoc relaxation, extra restart or retry.

Cross-view residuals are descriptive, not heldout evidence: the selection used all three views. Count and agreement cannot certify identity or coverage. Do not compute teacher-angle/full-function/private-reference metrics on this consumed dataset. A later properly reviewed fresh-owned study must evaluate matching gates, magnitudes, couplings and sealed full function with explicit access and error bounds; this development cannot substitute for it.

## Preconditions before execution

A separate implementation plan, source module(s), pinned preregistration and independent full-source approval are required. Design approval alone permits implementation, never experiment execution. Meaningful small literal fixture tests must cover dimension independence, r/2 hypothesis/abstention, zero/overshoot/no truncation, merger boundaries, noisy quartic counterexample assertions, deterministic seed mapping and absence of truth access. Existing successful actual runs are not rerun.

Compute an exact shape/byte/shard inventory for every persisted stage and maximum transport before approval. Include all six cases, maximum rank192/target96 and all four rescue rounds; do not budget only an expected early exit. Require180s shared process+children,512MiB sampled TREE, BLAS1, stdout+stderr2KiB,>=2GiB actual free disk and combined artifact bound<1GiB before generation. If the complete inventory cannot fit, redesign and review before any RNG, rather than reducing logging mid-run. Resource monitoring is sampled RSS, not a continuous OS allocation bound. No paid calls/background scheduling.

Require clean committed frozen sources/dependencies/config, verified parent input commitments, monitor/transport preflight and one exclusive global development marker before RNG or fit. Existing markers/source/data unchanged. Partial failure consumes the marker; no automatic retry. A terminal checkpoint precedes separate saved-output outcome audit. Source/fixture review must include a tiny synthetic geometry case and default outer launcher failure behavior, without running the registered development data.

## Decision gate after development

Record all six cases and full resource costs even if none works. Candidate count is descriptive only; no method promotion or production API support. Identify the next discriminating full-map/fresh-validation hypothesis from results; any newly tuned setting remains development and requires separately unseen validation. Precision, bias, HVP/scaling, weaker access and decoder-backbone work remain open. Final outputs/logits do not imply isolated Hessian access. Universal reconstruction and original parameter recovery remain unproved.
