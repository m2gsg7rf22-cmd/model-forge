# Isolated GLU pipeline development — 2026-10-24

RESEARCH IN PROGRESS. A full **isolated bias-free SwiGLU map** has now been approximated on one owned development teacher. This advances beyond the earlier even-only experiment, but does not reconstruct a decoder, arbitrary closed model or original branch weights. Private final-model gate CLOSED.

## What changed

The saved finite O3 fitting protocol failed its first sealed criterion. A separate observation regime supplies chosen continuous inputs and full vector outputs of an isolated block. Following the Hessian/subspace and antipodal decomposition in [Shi & Fu](https://arxiv.org/abs/2608.06631), the pilot collects finite-difference Hessians, searches for gate directions, fits magnitude/coupling approximations from small/large input scales, and solves orientations from the even map. This access cannot be inferred from a language-model API.

The initial Adam direction search retained62/64 directions at residual<=1e-8. An independent analysis showed why residual alone is weak: for v(t) proportional to g+t*u, the subspace already contains gg^T and gu^T+ug^T, leaving a t^2*uu^T residual and quartic squared loss. A tangent Gauss–Newton residual solve with public-loss backtracking improved all candidates. This is an independently derived local refinement, not a universal convergence theorem or a claim of unprecedented methodology. The independent Jacobian finite-difference check differs by2.13e-11.

After refinement,64 nonduplicate directions matched64/64 teacher gates one-to-one (minimum absolute cosine0.9999999975337291). This matching used truth **after** the fit; truth did not choose directions or step sizes. The public fixed teacher seed makes this a development pilot, not hidden-teacher evidence.

## Multiscale approximation

For unit direction v, gate magnitude c and coupling C, let B=c^2*C. At small rho, the odd map is approximately rho^3/4 times sum(v.x)^2*(B*x). Linear least squares from4096 probes at rho=.001 estimates B. At rho=1000, the saturation approximation gives a linear system in1/c from512 probes. Couplings are projected to rank one; canonical even queries determine orientations. These are asymptotic approximations with measured error, not exact algebraic recovery of the full map. The learner receives public arrays only; preparation owns the teacher. The process split is logical, not an adversarial OS sandbox.

The small design has rank2048 and condition54.30; magnitude and orientation systems each have rank64. The projected coupling relative singular tail was7.03e-5. New calibration cost:10272 vector queries, combined with135168 Hessian calls =145440. These are local synthetic block evaluations, not paid/provider calls or token budgets.

Development validation (256 new standard-normal inputs): relative Frobenius map error5.45662998585003e-5, maximum absolute error0.0005604833374961427. Independent reviewer fixed2048 additional inputs,512 at each scale:

| Input scale | Relative Frobenius map error | Max absolute error |
| --- | ---: | ---: |
| .04 | 5.29068328384414e-5 | 4.7250812450390367e-7 |
| 1 | 5.3174416792534015e-5 | .00048592229800847875 |
| 4 | 5.19241506479853e-5 | .008994487391338346 |
| 30 | 5.409993345545166e-5 | .6176177079814806 |

Scale30's large absolute error is preserved; a small relative metric is not an absolute-accuracy guarantee. Independent parameter diagnostics: correct64 orientations, max relative gate error7.25e-5 and coupling error7.81e-5 under one-to-one matching. U/D factors remain gauge-ambiguous.

The reviewer found an inconsistent norm in the orientation diagnostic. Original report3.3598e-5 mixed upper-triangle numerator and full-matrix denominator. Corrected consistent upper metric4.6153e-5 (Frobenius4.5574e-5). The original report is retained with explicit correction; changing this reporting metric did not change the fitted solution.

## Verification and reproducibility

Portable runnable source: examples/isolated_glu/. Three targeted integration/control tests passed on local macOS in3.94seconds, with512MiB process-tree sampled limits and180seconds per stage in the integration test. No fit peak RSS was recorded. Individual prototype times omit some preparation/SVD work and must not be presented as total pipeline time. Independent review reports memory for its own diagnostic only.

Regression rejected zero Hessians only after reproducing the old false rank128 acceptance. A second control rejects full-rank noise without spectral separation. Neither passing these checks nor a spectral gap proves the candidate subspace has exactly the target components. Finite precision, zero downstream units, collinear units, insufficient probes, width/activation misspecification and other teachers remain to be tested. Public-service input hardening is incomplete; this is a Python research pilot, not a new stable MCP full-model tool.

Public reports and frozen-source hashes: experiments/evidence/isolated-glu-development/. Private/large arrays stay in the durable application-data experiment directory. The preceding even-only sealed evidence is unchanged. Next: preregister fresh-teacher replication before teacher generation, preserve fixed thresholds and consume heldout only once. The O3 iteration-budget v2 question remains separate and unresolved.
