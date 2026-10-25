# All-pool public FP32 candidate: quality criterion FAILED

New reviewed method source92ea993 selected all256saved finite refined starts by loss-ordered angular deduplication, with NOresidualacceptancefilter or widestplateau confidence. Exactknownwidth64 was checked after selection; no truncation. Frozen multiscale calibration returned finite nonzero rankone64unit candidate, ranks2048/64/64 and positive magnitudes. This is a distinct public development hypothesis; the original strictFP32failure/criteria remain unchanged.

A candidate was built and a separate evaluator completed once, but **allfourgroups failed the prospectively fixed relativeFrobenius<=.001 development criterion**. Executionstatus public_candidate_evaluated is not qualitysuccess. Evaluationseed941027 is now consumed PUBLICdevelopmentdata, not an independent/sealed futuretest. No benchmarkthresholdwaschanged.

| Input scale | Relative Frobenius error | Maximum absolute error |
|---:|---:|---:|
| 0.01 | 0.00250529437835 | 1.79681040686e-06 |
| 1 | 0.00257212520553 | 0.0243142152331 |
| 10 | 0.00254897238446 | 2.87851956876 |
| 100 | 0.00261526983369 | 272.218929319 |

Maxgrouprelative.002615269834, maxabsolute272.2189293. These are finite publicvector-map measurements, not reasoning/coding/language benchmarks, originalstoredweight recovery, wholemodel equivalence or a productionFP32profile. The candidate is not promoted.

One frozenmultiscalefit;0newcalibrationqueries/0directionoptimizations;2048actualPUBLICevaluationresponses. Controllerelapsed1.707790375s, maxsampledworker-treeRSS232882176bytes(build24samples/eval2samples), overall180s/512MiB perworker, BLAS1; lightweightcontrollerRSSexcluded. All13oldcalibration/telemetryfiles remained unchanged. FiveNEWlocalwrappertests/types/lint passed; no newCI/nativebuild/privateassets. Candidate/publicvalidation arrays stay outsideGit in ~/.local/share/modelforge-research/experiments/fp32-all-pool-candidate-v1-20261025-92ea993/candidate/. CandidateSHAbaa6611a418fc2f65d1075c986034785fbac64cb25d7b45d03e09b31cf56764e; rawvalidationSHA10f6ed33bd07f032392e4617e5a54d9c911b93c3eed5cc74a7df348ce8ef45dd.

Independent outcome audit recomputed allgroupmetrics from savedinputs/reference/predictions, checked prediction using an independent einsum forward path, selection, rankone factors, commitments and oldinputs without newteacherqueries/fitting/evaluation. The evaluator's PUBLICreference responses are recorded; this is not hidden-teacher secrecy or a freshsealed replication.

## Next causal hypothesis, unverified

The preprojection couplingtail.00473715 and evenorientation calibrationresidual.00228380 suggest a calibrationconditioning issue, but do NOT prove FP32odd-part cancellation caused the error. Currentcubicfit uses O(.001*x)/(.25*.001^3). Under smooth biasfreeSwiGLU assumptions, normalizedodd expansion is cubiccoefficient+O(rho^2); subtracting rounded even-dominatedresponses can amplify output/operation error approximately as epsilon/rho. This motivates a separate numerical test, not a causal verdict.

Proposed new method: fixedrho=.03 and2rho=.06 (no validationgrid search), collect16384ownedFP32signedcalibrationresponses for existing4096small_x; retain allrawpositive/negativearrays. For A(rho)=O(rho*x)/(.25*rho^3), use (4A(rho)-A(2rho))/3 to cancel the leading rho^2 truncation. Mark this as DERIVEDcubicRHS, not physical.001observations. Fitting may bridge that RHS into the frozen linear multiscale solver with unchangedlarge/evenobservations. It needs its own source/protocol review, syntheticTaylor controls and commitment-bound collection/fit/evaluation separation BEFORE execution. Newpublicevaluationseed941028; NEVERrepeat the reserved941027evaluation or call it independentvalidation. Keep maxgrouprelative.001 unchanged, retain allerrors, and preregister freshteachers before any broader scientificclaim.

[Primary isolated-block research, section3.3](https://arxiv.org/html/2608.06631v1) explains small/large oddpart separation and scale-dependent arithmetic effects. The specific normalized-cubic Richardson test here is a proposed numerical adaptation, not a claim to have solved general model extraction. Bias/metadata/precision/scaling/HVP/O3 and final-output-to-block reduction remain open. Universalgoalactive/privatefinalgateCLOSED.
