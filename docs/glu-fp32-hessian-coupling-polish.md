# Saved-Hessian coupling polish: independently audited quality FAILURE

Source 0c457453aac3bef72c2a3ae53ae9994c4c79f17f. The source, protocol and synthetic controls were independently reviewed before the actual run. The independent saved-array outcome audit is now complete. Its initial usage-limit interruption was resolved without rerunning the experiment. Audit approval concerns evidence integrity only; model quality remains FAILED.

One fixed pass selected and retained units[37, 15, 6, 1, 13, 46, 2, 8], from64initial scores and8sequential local solves (72total). The saved-Hessian squared Frobenius cost fell from0.801402225348 to0.575671015623, a28.167031% decrease. Gates were fixed by construction. Complete-residual decreases do not establish forward generalization.

| Input scale | Relative Frobenius error | Maximum absolute error |
|---:|---:|---:|
| 0.01 | 0.00442867477545 | 4.10995456379e-06 |
| 1 | 0.00447106502826 | 0.082344782788 |
| 10 | 0.00439555257275 | 5.24152984197 |
| 100 | 0.00468101274842 | 965.141599535 |

All four new PUBLIC groups FAILED the prospectively unchanged .001criterion. Maximum relative error .00468101274842 and maximum absolute error965.141599535 must remain recorded. Candidate is NOT promoted. These941029points differ from prior941028points, so this is not a paired causal comparison with the previous candidate. Publicseed941029 is now CONSUMED and must never be presented as fresh validation again. No extra unregistered sweeps or repeat evaluations.

The learner used only hash-bound saved FP32 Richardson Hessians and the prior frozen candidate, no teacher weights, new calibration queries or direction updates. All64initial proposal matrices and8sequential proposal matrices, initial/final residuals and candidate are retained outside Git at /Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-hessian-coupling-polish-v1-20261025-0c45745/candidate/. An independent auditor can recompute proposal costs, order and retention without solving new couplings. Controller reports all13oldcalibration/telemetryfiles unchanged; builder reports its4inputs unchanged. The independent audit verified all11source/dependency hashes, all4primary inputs and13oldinventoryfiles. It recomputed all64proposal costs (maximum discrepancy3.11e-15), exact selection/order/retention, full residuals (relative discrepancy<=1.30e-13), byte-identical gates and finite nonzero rank-one couplings. New941029inputs/metrics and an independent einsum forward were verified (relative discrepancy4.17e-16), without fitting or teacher queries.

Elapsed1.26558913s; largest sampled worker-tree RSS209436672bytes;512MiB polled check/180s overall/BLAS1. Polling is not a hard reserved limit or proof of actual peak. New evaluation made2048ownedFP32vectorcalls,0newcalibrationcalls. Five NEW focused tests, strict mypy and Ruff passed before collection; no broad regression/newCI/native rebuild.

The outcome audit is complete. Next derive and independently test a fixed-gate full-forward linear factor/Gram/whitened rank-one primitive on tiny synthetic controls. Any actual experiment must separately freeze authorized calibration collection, scale balancing, training/guard splits, ranking, retention, resource limits and disjoint final public evaluation. Use the pre-polish frozen candidate as the explicitly declared starting point; the failed Hessian-polished candidate is not promoted. A guard used for selecting updates is calibration, not an independent final test. Fixed gates still cannot correct direction errors. Hessian descent by itself is insufficient for the fixed quality requirement. Universal reconstruction remains unresolved and private final-model gate CLOSED.
