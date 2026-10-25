# Saved FP32 calibration residuals

Reviewed source 94107291dd408f9d04c934ada63a896934fb828c; one read-only diagnostic, independently audited. The prior cubic Richardson candidate still FAILED all four .001 public quality criteria. This diagnostic changes no candidate, threshold or evaluation.

| Comparison (first estimate, second reference) | Rows | Relative Frobenius residual | Maximum absolute residual |
|---|---:|---:|---:|
| old_rhs_vs_new_rhs | 4096 | 0.000105265600307 | 0.00624977811067 |
| candidate_cubic_vs_old_rhs | 4096 | 0.00272019058296 | 0.165709880289 |
| candidate_cubic_vs_new_rhs | 4096 | 0.0027180644926 | 0.165923038899 |
| candidate_even_vs_saved_even | 528 | 0.0024748099195 | 0.000841246856636 |
| candidate_exact_odd_vs_saved_large_odd | 512 | 0.00273269763425 | 16641.9023039 |
| candidate_saturation_vs_saved_large_odd | 512 | 0.00273248035725 | 16641.902304 |
| candidate_saturation_vs_candidate_exact_odd | 512 | 1.7109113595e-05 | 296.745726176 |

Each denominator is the norm of the stated reference, not unknown teacher derivatives. The seven full records retain both reference and prediction norms. Old/new normalized odd estimators differ by .000105266, whereas this frozen candidate's cubic residual is about .00272 against either estimate. The even and large-odd residuals are similarly about .00247–.00273. Changing the small-input estimator alone was insufficient to meet the prior fixed quality criterion. These data do not identify whether direction, magnitude, coupling or projection error dominates.

The large probe inputs are 1000*large_x. Both the exact tanh odd response of the candidate and its saturation approximation are compared with the saved FP32 large_odd responses. Their mutual relative difference is .0000171091, an approximation error of the candidate itself, not a measurement of the target's saturation error. Large-scale absolute errors exceed16000 and must not be hidden; these calibration inputs differ from the prior public evaluation scales. Even probe points were reconstructed in the exact canonical e_i then e_i+e_j order used by the source-frozen collector.

The fitting-side unprojected coupling tail .004732379 is retained as a reported scalar. Preprojection least-squares coefficients were NOT saved, so the effect of rank-one projection cannot be independently recomputed here. No least-squares replay or optimizer was hidden in the diagnostic.

All seven source inputs remained byte-identical. Signed responses, Richardson RHS and its explicit nonphysical .001 bridge were recomputed. An independent einsum implementation reproduced the candidate components and all metrics without target queries, teacher weights, refitting, optimizers or consumed evaluation files. Four NEW focused tests, strict mypy and Ruff passed; no broad regression, new CI or native rebuild. Controller0.130939292s; sampled process-tree RSS40517632bytes with2samples. The512MiB bound is a polled check, not a hard reserved limit or a proof of actual peak RSS. Raw comparison arrays stay outside Git at /Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-calibration-bottleneck-v1-20261025-9410729/diagnostic/calibration-comparisons.npz, SHA2560134b14319d64c3a63ca23a28f336ad35e5bca417238e638aaa9a44eb7c788b3.

## Next discriminating method

[Primary GLU research, section3.5](https://arxiv.org/html/2608.06631v1) separates direction refinement from fixed-gate rank-one coupling polish on saved Hessian residuals. It reports limited improvements and does not guarantee repair. This motivates a separately reviewed analytic Hessian operator/adjoint/Gram and a whitened rank-one update primitive, first verified on tiny synthetic controls. No actual saved-H fitting, new target query or candidate evaluation is authorized by completion of this diagnostic alone; such a run needs its own prospective protocol, budget, source review and disjoint public evaluation points. Calibration descent is not generalization. Fixed gates cannot repair gate directions.

Universal/full Transformer reconstruction, original stored weights, supported production FP32, bias/scaling/HVP and final-output-to-block reduction remain open. Public seeds941027/941028 remain consumed. Private final-model gate CLOSED.
