# Public FP32 normalized-cubic Richardson: FAILED development criterion

Source 188a03ad9d13c33a71eb24ab6b987f633d584bfc. Prospectively reviewed fixed rho=.03/.06, 16384 NEW owned NumPy FP32 calibration vector responses, four signed batches on existing4096inputs. Raw signed responses are retained and hash-bound. Normalized odd A(rho)=O(rho*x)/(.25*rho^3); R=(4A(rho)-A(2rho))/3 cancels the leading rho^2 term under smooth bias-free assumptions. Synthetic controls independently verified quintic cancellation and the septic remainder. The fit consumes an explicitly DERIVED cubic RHS through small_odd=R*(.25*.001^3); this is NOT a physical .001 target observation or FP64 target response.

Cached all-pool directions, original large/even calibration and unchanged multiscale rank/positivity/finite/rank-one guards were reused. No direction optimizers or teacher truth entered fitting; one multiscale fit. A separate process evaluated the resulting candidate on NEW PUBLIC seed941028, 2048points, four fixed scales and unchanged maxgroup relative criterion .001.

| Input scale | Relative Frobenius error | Maximum absolute error |
|---:|---:|---:|
| 0.01 | 0.00253067386392 | 2.20580656851e-06 |
| 1 | 0.00252246113273 | 0.0219901381756 |
| 10 | 0.00265891401058 | 3.24459191449 |
| 100 | 0.00265877725025 | 291.069104467 |

Actual metric flag: False; max group relative 0.00265891401058; max absolute 291.069104467. Execution completion alone is not quality success. Old all-pool result remains FAILED(.002615maxrelative,272.219maxabsolute); This run does not establish quality improvement or identify the dominant error source. Both scale and estimator changed, and the evaluation inputs differ, so comparing the two sets is not a paired causal test.

Controller 1.85303229s; sampled sequential worker-tree peak 283525120bytes; 180s overall/512MiB per worker/BLAS1, controller RSS excluded. RSS is polled rather than reserved: the fast evaluator had only one startup sample (458752bytes), so its actual peak may have been missed; the configured512MiB check is not proof of a hard whole-run bound. No evaluation is repeated to improve telemetry. 16384newcal+2048publiceval=18432new vector responses. All13oldcalibration/telemetryfiles unchanged. Seven NEW focused local tests, strict mypy and Ruff passed; no new cross-platform CI or native rebuild. Independent reviewer audited signed-array formula/bridge, commitments, raw metrics and independent prediction without teacher queries or refitting. Source review found and closed a null raw-hash bypass before any collection.

Raw signed arrays, derived RHS, candidate and evaluation arrays remain outside Git at /Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-cubic-richardson-v1-20261025-188a03a. Candidate SHA256 b2d5fdd4cf8be40954562c2ce9393cd5b9a3c3cbd3cd8cbfa9bc18728d1de41e; validation SHA256 8a32bd29dc4c70c6b2791955a1942c4afa62e268ca84cafc528ceff1e4603ccd. Public seed941028 is now CONSUMED development data and cannot count as fresh independent validation later. No private assets accessed.

This is one public owned isolated32/64/32 bias-free FP32 block, not a fresh hidden-teacher replication, original stored-weight proof, all-input equivalence, reasoning/coding benchmark, productionFP32 profile or full Transformer reconstruction. Universal reconstruction remains open; private final-model gate CLOSED.

Next: a distinct read-only calibration bottleneck diagnostic must compare the saved old normalized odd RHS and new Richardson RHS, quantify candidate cubic/even/large calibration residuals and explicitly record that preprojection coefficient arrays were not retained (no silent least-squares replay), without new queries, optimizers, teacher truth or public-evaluation reuse. The unchanged coupling tail near.00473 and orientation residual near.002284 do not establish which error source dominates. Any new direction refinement/fixed-gate polish needs separate prospective source/protocol review; blind fresh-teacher validation is premature until a public method meets its fixed criterion. Bias/metadata controls, dimension/width/HVP scaling, weaker O3 access and final-output-to-block reduction remain required before broader claims.

[Primary GLU research](https://arxiv.org/html/2608.06631v1) studies isolated blocks and explicitly leaves deriving internal block responses from final model outputs unsolved. A separate [2026 token-ranking paper](https://arxiv.org/abs/2606.04459), checked at the primary abstract, reports approximate FINAL-layer extraction from rankings with restrictive top-k caveats; it is a future weaker-access lead, not full-backbone recovery or a verified ModelForge backend. No provider endpoint was queried.
