# Variable-width gate development v1: independently audited outcome

The saved-output audit passed. The six-case candidate-count objective failed: only one case reached its hypothesized rank/2 count. No gate identities, coverage, magnitudes, couplings, full function or original weights were verified. The private final-model gate stays CLOSED.

| Public case | Channel | Numeric rank | Hypothesized count | Initial candidates | Final candidates | Rescue rounds | Status |
|---|---|---:|---:|---:|---:|---:|---|
| c00 | FP64 | 192 | 96 | 81 | 93 | 4 | rounds_exhausted |
| c00 | FP32 compute | 192 | 96 | 1 | 2 | 4 | rounds_exhausted |
| c01 | FP64 | 96 | 48 | 47 | 47 | 4 | rounds_exhausted |
| c01 | FP32 compute | 96 | 48 | 18 | 18 | 4 | rounds_exhausted |
| c02 | FP64 | 32 | 16 | 16 | 16 | 0 | candidate_target |
| c02 | FP32 compute | 32 | 16 | 13 | 13 | 4 | rounds_exhausted |

These are three owned teachers with dependent precision cases. Optimizer seeds differ between jobs; the counts do not isolate a causal precision effect. Rank/2 is a generic-family hypothesis, not a disclosed true width. A candidate count is not gate recall. Even job04 has no identity or functional certificate.

Source was frozen at `43cf7a6cc04b88d1cf1b3071fcf1fe437dd38eae`. The one registered attempt consumed its marker; no retries occurred. All six cases terminated. Time before the final report was 95.066099916 seconds; sampled process-and-children RSS reached392,577,024bytes across1499samples. Retained artifacts after the outer receipt occupy135,203,053bytes. Sampling does not certify a continuous peak. No new teacher queries were made by this development or its outcome audit.

The independent audit checked all28source pins/14runtime pins, parent54public+24estimator archives, all5064trace+20transport archives, all20rescue rounds, source/marker/seal/basis binding, pretraces, complete chunks/aggregates, unit norms, original-subspace losses, merge decisions/order and final trajectories, saved RNG state metadata, residual/tangent geometry and saved eigen-equations. Fixed cutoffs stayed1e-8 and.9995. The11,776merge candidates comprise189accepted,1667duplicate-rejected and9920loss-rejected decisions. These descriptive categories do not identify the remaining error's cause or justify relaxing a threshold. Rescue added12then0/0/0 to job00 and1then0/0/0 to job01; the other deficient jobs gained zero.

The auditor used saved arrays with independent NumPy/stdlib calculations, no teacher forward/query, private factors/context/heldout, registered RNG reconstruction, optimizer/Adam/refine replay or eigensolver. SVD only verified saved geometry. Seed-to-state and intermediate optimizer-step fidelity rely on signed source/provenance; the audit verifies endpoints and recorded numerical relationships, not every optimization step. Maximum saved eigen-equation residual was1.0131e-15; chunk-vs-aggregate loss discrepancy1.0546e-19. The one audit used154,402,816sampled TREEbytes/68samples with4.511106seconds including preflight, within180s/512MiB/BLAS1/2KiB.

Evidence: [independent report](../experiments/evidence/variable-width-gate-development-v1/outcome-audit/outcome-audit.md), [signed result](../experiments/evidence/variable-width-gate-development-v1/outcome-audit/outcome-audit.json), and [merge decisions](../experiments/evidence/variable-width-gate-development-v1/outcome-audit/merge-counts.json). Raw arrays remain outside Git. Prior rank and full-map failures are preserved; stable native releases are unchanged.

Next investigate a distinct observation contract that preserves chosen inputs and raw full-vector responses for all six cases, instead of treating subspace count as reconstruction. First review the design and strict public-only transcript whitelist; no read of old truth-directory arrays, source modification, RNG, fit, teacher generation or new query is approved by this outcome report. Reuse of disclosed calibration cannot become heldout evidence. Fresh independently sealed full-function evaluation remains necessary after a complete separately approved protocol. Bias, HVP/scaling, weaker access, decoder backbone and universal reconstruction remain unresolved.
