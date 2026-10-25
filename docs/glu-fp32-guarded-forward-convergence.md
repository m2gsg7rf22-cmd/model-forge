# Bounded guarded-forward study: improved relative errors, quality FAILED

Reviewed source 4da51d4e55ee8a19d58d1bd16d4a2014d9fba4e8; one DISTINCT preregistered run, reusing the preceding1024training+1024guard calibration vectors explicitly as calibration. No new calibration target queries. Starting point is the quality-failed experimental guarded-forward candidate, not a production model. Gates stayed byte-identical.

| Sweep | Initial selected proposals | Accepted updates | Local solves | Final squared training cost |
|---:|---:|---:|---:|---:|
| 1 | 63 | 32 | 127 | 1.57444523615e-05 |
| 2 | 63 | 15 | 127 | 1.52667707508e-05 |
| 3 | 63 | 3 | 127 | 1.52506125588e-05 |
| 4 | 63 | 2 | 127 | 1.51718319754e-05 |

The bounded study reached its FOUR-sweep cap, with508local rank-one solves and52accepted updates (updates may revisit a unit). Training cost fell from2.06931413866e-05 to1.51718319754e-05, a26.681833% decrease. It did not stop for zero accepted updates and does NOT prove convergence, stationarity or a fixed-gate functional floor. Many proposals were rejected by the declared criteria; this does not identify the cause of remaining error.

Every proposed update had to decrease the complete weighted training cost by the fixed margin and obey BOTH current and initial guard-group bounds. The initial bound prevents tolerance drift above1e-10per group over the run. Guard data selects updates, is reused adaptive calibration, and must not be labeled independent heldout evidence. Training targets were independently verified as raw observed FP32 responses multiplied by the saved multipliers ONCE.

A separate evaluator used NEW PUBLIC seed941031 with2048responses and compared the committed incumbent and changed candidate on IDENTICAL points. All four group-relative errors improved, but **all four FAILED the unchanged .001quality criterion**. Maximum candidate relative error .00226807771201; maximum absolute287.313330512. No candidate promotion.

| Input scale | Incumbent relative | Candidate relative | Incumbent maximum absolute | Candidate maximum absolute |
|---:|---:|---:|---:|---:|
| 0.01 | 0.00224977274837 | 0.00202042178955 | 1.31038557963e-06 | 1.39739513015e-06 |
| 1 | 0.00236822500312 | 0.00222071903937 | 0.0230588222273 | 0.0231646002586 |
| 10 | 0.00244870303415 | 0.00226807771201 | 2.752446617 | 2.62065674662 |
| 100 | 0.00230553643473 | 0.00216153386329 | 311.792841917 | 287.313330512 |

The maximum absolute errors at scales.01and1 WORSENED despite lower group-relative errors. These tradeoffs are retained; this is not improvement of every reliability measure. Publicseed941031 is now CONSUMED and cannot be retried or reused as fresh validation. No unregistered extra sweeps follow this result.

Independent outcome audit verified12source/dependency hashes, all5primary input commitments/provenance and13oldinventoryfiles. All256initial-screen and252sequential proposals were checked from stored matrices without solving new couplings. Screening costs agreed within3.22e-20; rankings/current partials/dual guards/margins and all decisions matched. Gates remained byte-identical; couplings are finite/nonzero/rank one. Full residuals independently recomputed agreed within2.71e-13relative; both independent forward paths and all paired metrics were verified. No target query, teacher weight, old validation or private asset was read by the reviewer.

Total2048newownedPUBLICvectorcalls,0newcalibrationcalls. Controller0.700574917s; largest sampled worker-tree RSS49610752bytes (build8samples/evaluation2). Actual peak is unknown and512MiB is a polled check, not a hard reserved bound.180s overall/BLAS1. Five NEW focused tests, strict mypy and Ruff passed before fitting; no new CI, broad regression or native rebuild. All proposals, weights/residuals, incumbent/candidate and raw public validation remain outside Git at /Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-guarded-forward-convergence-v1-20261025-4da51d4/candidate/. CandidateSHA2563fa13a8f6337bfc9d317a95dde95cd0d87ab6ae04568a392bf6b227a517f40b3; publicvalidationSHA256b72eb08407a1c9789441c4f4ccaf31224ffa1d0d97f95347d5d820e7483c2540.

Next is the separate DRAFT candidate gate-gradient primitive in docs/superpowers/plans/2026-10-25-weighted-forward-gate-gradient.md. First verify weighted candidate gradients and finite differences on tiny synthetic cases, with no saved-data fitting or new queries. These are derivatives of known candidate computation, not privileged derivatives of the inaccessible target. This investigates another degree of freedom without claiming gate error is the cause. Any actual update method needs its own prospective reviewed protocol and new final evaluation.

Primary literature leads checked during this study: [Curvature Cryptanalysis of Smooth Transformer Feed-Forward Networks, v2](https://arxiv.org/html/2608.28843v2), sections3–5, analyzes isolated two-layer W2 phi(W1x+b1)+b2 branches, conditional local identifiability and shared vector-output stencil reuse. Its oracle is not a token-only whole-model API, and its plain FFN result must not be silently applied to the multiplicative GLU architecture. [Asselineau et al., IACR2026/253](https://eprint.iacr.org/2026/253) was checked at its primary abstract: it studies nonlinear activations approaching linear behavior and higher-order derivative/linear-zone methods. Its use of universal is class-conditional, not evidence for reconstructing arbitrary AI models. Neither is a newly verified ModelForge backend; fuller assumption/proof/code audits precede any new route.


Universal/full Transformer reconstruction, original stored weights, supported productionFP32, fresh hidden-teacher precision/scaling/bias/HVP and final-output-to-block reduction remain open. All prior failures are preserved and private final-model gate CLOSED.
