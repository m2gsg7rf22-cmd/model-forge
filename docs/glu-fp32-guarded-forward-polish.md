# Guarded forward coupling polish: paired improvement, quality FAILED

Prospectively reviewed source db8fb33470f8ac1a9ebbf1fcb2567110e834c2bc. One bounded public development run collected1024training and1024guard responses from the disclosed owned actual NumPy FP32 fixture. Seeds882030/882031,256points each scale.01/1/10/100, raw inputs/responses and hashes retained. The learner used no teacher parameters. Starting candidate is the original pre-Hessian-polish cubic candidate, not the failed Hessian-polished candidate.

Training residual multipliers are exactly1/each training group's Frobenius reference norm, applied to physical responses ONCE. This balances the FOUR squared group-relative errors; loss weights are the squares of these multipliers. The independent audit recomputed the derived targets and weights exactly from raw arrays.

The fixed rule screened64units and made one pass over8selected units: [51, 37, 15, 39, 14, 56, 28, 8]. All8passed both complete training-cost descent and every current guard-group nonworsening test.72local rank-one solves, no gate/direction changes or extra sweeps. The training objective fell from2.64478001318e-05 to2.06931413866e-05 (21.758554% decrease). Guard data selected updates and is calibration, not independent final evaluation. Its1e-10additive tolerance is per step against the current incumbent and may accumulate to8e-10; it does not alter the final quality threshold.

A separate evaluator collected2048NEW PUBLIC responses at seed941030 and predicted BOTH the committed baseline and changed candidate on the SAME inputs. All four group-relative errors improved in this finite paired comparison, but **all four still FAILED the fixed .001quality criterion**. No candidate promotion, blind-teacher or broad generalization claim.

| Input scale | Baseline relative | Candidate relative | Baseline maximum absolute | Candidate maximum absolute |
|---:|---:|---:|---:|---:|
| 0.01 | 0.00251893698359 | 0.0022866612815 | 1.67360850299e-06 | 1.51728765328e-06 |
| 1 | 0.00263968555665 | 0.00240471160908 | 0.0289241483525 | 0.0182744333291 |
| 10 | 0.00254900402183 | 0.00235710011621 | 2.48669134334 | 2.13118888563 |
| 100 | 0.00268156038036 | 0.00241619497836 | 261.800884249 | 254.037272469 |

Maximum candidate relative error .00241619497836 and maximum absolute254.037272469 remain recorded. Publicseed941030 is now CONSUMED. It must not be used again as fresh validation, for retrying the reserved evaluation, or to adapt the completed run.

Independent outcome audit verified12source/dependency commitments,3baseline inputs, the new calibration chain and13oldinventoryfiles. All64proposal costs agreed within5.09e-20; initial ranking,8current-partial solves' saved proposals and every train/guard retention decision matched. Gates are byte-identical and all final couplings finite/nonzero/rank one. Full residuals recomputed independently agreed within2.32e-13relative; both independent einsum forward predictions agreed within4.19e-16relative. No new coupling solve, target query, teacher weight, private asset or old consumed validation file was used by the reviewer.

Total4096newownedvectorcalls=2048calibration+2048finalpublic. Controller0.453237917s; largest sampled worker-tree RSS40894464bytes,2samples in each stage. The actual peak is unknown;512MiB is a polled check, not a hard reserved bound.180s overall and one BLAS thread. Five NEW focused tests, strict mypy and Ruff passed before collection; no broad regression/newCI/native rebuild. Raw observations, all proposal/weight/residual arrays, baseline/candidate and public validation stay outside Git at /Users/adamfaruhi123/.local/share/modelforge-research/experiments/fp32-guarded-forward-polish-v1-20261025-db8fb33. CandidateSHA256b31ce967cab997e9a5fd2652ab63db28f32204638a7f3f7ca58f4c1673e4d0b2; publicvalidationSHA256f153a968e36198a26f68910c7c0f6a2204eb498b56dda92481d1f2f1389489b6.

Next is the distinct DRAFT bounded fixed-gate convergence protocol in docs/superpowers/plans/2026-10-25-guarded-forward-convergence.md: investigate up to4sweeps across64units using explicitly reused calibration, with stored proposals and stricter global guard drift bound, a new reviewed source/controller and disjoint public941031evaluation. No additional fit is authorized by this report alone before that review/commit. The current experimental candidate remains quality-failed. Fixed gates cannot repair direction errors; no assumption is made that extra coupling updates close the gap.

Universal/full Transformer reconstruction, original stored weights, productionFP32, fresh hidden-teacher precision/scaling/bias/HVP and final-output-to-block reduction remain open. All previous failures are preserved. Private final-model gate CLOSED.
