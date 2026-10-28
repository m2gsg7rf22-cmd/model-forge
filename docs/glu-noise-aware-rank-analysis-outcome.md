# PUBLIC observation-only rank analysis v1 outcome

Status: actual controller COMPLETED; independent saved-output numerical and sampled-TREE resource audit APPROVED on2026-10-28. All16cases were reproduced without new queries/fits/RNG. This is PUBLIC exploratory development, not blind/fresh/unknown-family validation or reconstruction.

One approved analysis of existing dimension-domain v1 observations, no new reference/model-forward/teacher/RNG/fit/private/941xxx use. Source20ccebb; no retry. Every certificate remains false. Exact protocol and frozen heuristics: [protocol](glu-noise-aware-rank-analysis-protocol.md). Source review and terminal evidence: experiments/evidence/noise-aware-rank-analysis-v1/.

## All16cases (no rule selected after seeing results)

|Case|Numerical analytic rank|Strict1e-10|Gap|Stable|Stable lost-subspace RMS|Stable gate residual max|Stable random residual min|
|---|---:|---:|---:|---:|---:|---:|---:|
|p0-t0-fp64|10|10|null|null|—|—|—|
|p0-t0-fp32|10|10|null|null|—|—|—|
|p0-t1-fp64|10|10|null|null|—|—|—|
|p0-t1-fp32|10|10|null|null|—|—|—|
|p1-t0-fp64|32|36|32|32|2.31955e-05|1.40007e-05|0.0131606|
|p1-t0-fp32|32|36|32|32|3.09669e-05|2.0821e-05|0.0131557|
|p1-t1-fp64|32|36|32|32|1.96487e-05|1.52711e-05|0.097365|
|p1-t1-fp32|32|36|32|32|3.80698e-05|3.43872e-05|0.0973547|
|p2-t0-fp64|64|136|64|64|3.0889e-05|5.96258e-05|0.454031|
|p2-t0-fp32|64|136|64|64|7.15781e-05|0.000136701|0.454015|
|p2-t1-fp64|64|136|64|64|2.88051e-05|4.69552e-05|0.547354|
|p2-t1-fp32|64|136|64|64|5.69394e-05|7.59779e-05|0.547358|
|p3-t0-fp64|128|402|128|128|3.01863e-05|8.99764e-05|0.79275|
|p3-t0-fp32|128|528|128|128|0.00012662|0.000253235|0.792756|
|p3-t1-fp64|128|398|128|128|2.56318e-05|9.59125e-05|0.789315|
|p3-t1-fp32|128|528|128|128|0.000122256|0.000232696|0.789316|

Lost-subspace RMS is unweighted geometric reference-subspace loss, NOT Hessian energy loss. Reference rank means numerical1e-10 rank of a finite analytic contraction, not exact intrinsic truth. Null candidates are retained. All spectra, bases, projectors, gap/ratio/eligibility/clipping/stability/floor diagnostics, gate/random vectors and metadata controls are saved, not just this table.

## Interpretation limited to these observations

Both observation-only gap rules chose the same ranks as numerical reference and known-width top2m in all12 nonfull-reference cases:32/64/128. All4 n4 full-reference cases returned null for both gap rules rather than selecting an informative proper subspace. These public families use width=2*n and are already known; excluding an explicit width argument is not statistical blinding or evidence of unknown-width generalization.

Strict numerical cutoff selected36 instead of32 for n8,136 instead of64 for n16, and402/398(FP64) or528(FP32) instead of128 for n32. Full-space membership controls are nondiscriminative. Agreement of gap/stable/knownwidth in12cases means NO improvement over the existing known-width route or over prior reconstruction errors has been demonstrated. No gate search, full function prediction, candidate fitting or heldout model evaluation occurred.

The heuristic subspaces approximately retain analytic signal and separate truegate/random directions on this development set. Their nonzero subspace/gate errors, shared-bias synthetic counterexamples and public-data design preclude rank certification, universal identifiability, original-weight or full-map claims. Similar gap/stable outcomes do not show that the stability filter adds value here. No iid-noise theorem applies without its assumptions.

Known/half/double-width controls preserve assumed/requested/effective ranks and clipping collisions. In n4 known and double controls both clip to fullp10 and cannot distinguish metadata errors; half rank8 discards full-reference signal. Other half controls drop roughly half reference dimensions; double controls add extraneous subspace. These are evaluator diagnostics, never estimator inputs or recovery certificates.

## Resources and provenance

Shared controller elapsed4.7152676669938955s before final report; polled worker TREE301416448bytes/63samples. Monitor scope process_and_children; sampled peak is not continuous/hard memory proof. Pre-data synthetic worst archives largest6748236bytes, actual free disk4438097920bytes.48outputshards/store57870416bytes, total directory71439817beforefinalreport.528original selected inputs checked before/after; original4704shards/globaldimensionmarker remain unchanged.

Durable root `/Users/adamfaruhi123/.local/share/modelforge-research/experiments/noise-aware-rank-analysis-v1-20261026-20ccebb`. Source `20ccebbabe355fd2dedbe6cd5be8e970f9d178ae`. Controller SHA256 `6cd99df29e303a192e6ac40dd05d641f18b9934eb54d502f3f466acd5e1300f3`. Worker SHA256 `aae3f8f305a851b7f92aea64db316ff06f050125002675b65aaee2a67489cd64`. Output manifest SHA256 `3bb22d7103f77ec8e64c17aca6929e94b1069b29d4494d94aab5ec587ac0731a`. Canonical analysis marker SHA256 `abebff1a6a262ef4c1f46690b48c9047d5e3993ed1ca64f28c0daf21adfa7087` CONSUMED/NOretry.

## Required next evidence

Independent saved-output audit is complete. A DISTINCT preregistered owned validation must vary hidden width independently of input dimension and include exact/near degeneracy, full/spiky/common-bias negatives before any unknown-width utility claim. Rules/thresholds stay fixed; no tuning against a sealed test. Separate gate/full-map tests would still be necessary. This does not resolve FP32 reconstruction errors in the existing known-width candidate. Bias, HVP cost, full-output-to-internal-block access, decoder backbone and universal closed-weight reconstruction remain OPEN; private final-model gate CLOSED.


## Independent saved-outcome audit (2026-10-28)

Signed audit under experiments/evidence/noise-aware-rank-analysis-v1/outcome-audit/:17reviewed source files at actual commit/current,10runtime pins,20prior signed files unchanged,528input/48output schemas+hashes,all16case order/seal/report bindings,independent spectra/basis/projector/rank/gap/stability,all6methods/metadata controls/rawresiduals/aggregates and before/afterhashes verified. Largest independent residual/metric absolute difference5.551115123125783e-16, stability absolute difference1.1737095670372177e-13; spectrum/projector differences0. Raw original oracle values deliberately not replayed.

Audit history is explicit: attempts1/2 failed before numerical cases (lost stderr; auditor manifest-byte accounting corrected), attempt3 passed16numericcases with children monitoring unavailable, attempt4 repeated only saved-data audit after a successful TREE preflight and byte-identical durable review-path substitutions. Attempt4 numerical results SHA313e2115548da552b809b53bc1b99281eee4974329ba7524372b54e2f3e1465b equals attempt3. Actual analysis remained ONE throughout. Final audit polledTREE219709440bytes/26samples,1.7166753339988645s; not continuous peak/hardRT. Failed/incomplete receipts remain preserved.

Next proposal, NOT executable/approved/data-generated: [distinct variable-width challenge](glu-rank-fresh-width-validation-draft.md). No teacher/seed has been reserved. Every gap/stable result remains heuristic, certificatefalse; no causal explanation of prior FP32 full-map errors or improvement over knownwidth baseline is established.
