# Public GLU precision observability development result

One fixed public diagnostic ran from committed source e6bc6f40a89cd39c4b8dedbfcd94004de8456f0c after independent source/protocol review and source-hash verification. All48registered rows and6,488,064actual full-vector evaluations completed in7.62284549998worker seconds. Existing run_json monitored the whole process tree:128samples, peak187596800bytes(178.90625MiB),512MiB/180s caps. NumPy2.5.4, macOS local host; no new cross-platform claim. No fitting, secret teachers, heldout, model-provider calls or private assets.

All64probes used the already-public seed572041 d32/width64/output32 block. The analytic reference rank128 gaps were1.193842186e14–1.244499568e14. The newly added>=100reference-gap guard was satisfied. This guards the reference only; an estimated gap is diagnostic and cannot certify recovery.

The table selects the largest *estimated* rank128 gap among each channel's fixed12options; it does not use analytic error for selection. This posthoc public comparison is not independent validation or proof that gap selection generalizes. Baseline is the existing central stencil at h=.003. Relative H errors compare full Frobenius Hessian tensors; sine is RMS principal-angle sine against the smooth analytic rank128 reference.

| Channel | Baseline H relative error | Gap-selected h/method | Selected H relative error | Selected RMS sine | Selected gap |
|---|---:|---|---:|---:|---:|
| fp64 | 2.6867229e-07 | 0.01/richardson | 4.9928829e-10 | 1.7119588e-10 | 4.7913718e+08 |
| fp64_return_fp32 | 0.3577791 | 0.3/richardson | 9.1928152e-05 | 3.2512611e-05 | 1389.8641 |
| fp64_return_fp16 | 2759.1142 | 1.0/central | 0.03866759 | 0.012143671 | 9.1176179 |
| fp32_compute | 2.1324472 | 0.3/richardson | 0.00030442182 | 0.00010461365 | 797.34824 |

At the old step, returned-FP32 and computed-FP32 curvature lose the rank separation required by the existing engine. Increasing the step trades rounding noise against truncation; Richardson at .3 gives a much cleaner subspace for these two channels. Richardson amplifies rounding noise at small steps (for example, FP32compute .003 H-relative2.132→2.841); it is not uniformly better. FP16-return's entire tested grid remains below gap100: best gap9.1176 at central h1. This negative result is retained; no out-of-grid conclusion is justified.

FP32compute means NumPyFP32 inputs, weights and arithmetic. Its analytic reference is smooth FP64 arithmetic using FP32-rounded weights at continuous points, **not a classical derivative of the piecewise-rounded runtime**. Returned-FP16 is FP64 computation with output rounding, not FP16 execution; no BF16/TF32 claim. The one publicteacher/knownprobe grid does not establish architecture-wide robustness or full-map reconstruction. Existing frozen v4 results, successful packaged GLU32 route and prior failures are unchanged.

## Next method, requiring its own review before execution

A distinct public-only full-map development pilot should test whether the observable curvature improvement survives actual gate/coupling discovery. Freeze h=.3 Richardson from this development result; collect both curvatures and **all** small/large/even observations from the same FP32 computation channel (never mix saved FP64 observations and label it FP32). This requires280608calibration evaluations:270336for two Hessian stencils plus10272multiscale. Feed only these observations into the frozen learner, preserve strict failures, and evaluate any candidate on explicitly public development points without calling them sealed heldout. No FP32 profile is added to the production IO until justified and reviewed; do not mislabel FP32 observations as the currently declared FP64 profile.

If the existing thresholds reject, record the numerical bottleneck before designing a noise-aware extension; gap alone cannot certify direction coverage or justify silently loosening criteria. Any positive development result must precede a distinct fresh-teacher preregistration with independent verification. Bias/metadata controls, dimension/width/HVP scaling, finiteO3 iteration budgets and final-text/logit-to-internal-block reduction remain separate open requirements. Universal and private final-model gates remain unmet/closed.

Primary research context: [Shi and Fu, section6.5](https://arxiv.org/html/2608.06631v1) studies output rounding separately from execution and uses step adaptation/Richardson under its stated isolated-block access. This diagnostic does not reproduce that paper's scale or recover stored weights. Evidence and source/outcome review: experiments/evidence/isolated-glu-precision-development-v1/.

Independent outcome audit verified source commitments, all48grid rows, spectral ratio consistency and resources without rerunning. Raw Hessians were not persisted in this diagnostic; therefore its reported H errors/angles cannot be recomputed from saved arrays. The review relies on committed computation and separate formula controls for those quantities. Future full-map pilots must save their calibration arrays for independent evidence inspection.
