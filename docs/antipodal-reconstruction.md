# Antipodal GLU research, 2026-10-24

Status: bounded engineering verified; full GLU and universal model reconstruction unresolved. Private final-model gate CLOSED. No private-model creation/training or native release rebuild was performed in this checkpoint.

## Access and method

The new NumPy route accepts authorized, canonically ordered continuous forward observations of an **isolated** bias-free SwiGLU/GeGLU block. This is stronger access than the saved O3 finite activation dataset; a text or logits API does not supply it automatically. For E(x)=(f(x)+f(-x))/2 the SiLU/GELU antipodal identity yields a quadratic map. At d=32, 528 paired points (1056 signed vector evaluations) determine its symmetric coefficients by algebraic interpolation. Hidden width is not an input to interpolation.

This implements a published component of Shi & Fu, [Cryptanalytic Extraction of Isolated Bias-Free GLU Feed-Forward Blocks by Antipodal Separation](https://arxiv.org/abs/2608.06631). It is not a new universal extraction theorem. Their gate/coupling discovery and odd-component stages are not implemented here. `recover_orientations` solves only a conditional sign system given candidate gate directions and rank-one coupling matrices, with rank/conditioning/residual guards; it does not discover these candidates.

[Finite Probes Suffice](https://arxiv.org/abs/2609.33901) supplies a generic analytic-family identifiability result, not a practical arbitrary-model solver, stored-weight uniqueness or a text-interface reduction. [Carlini et al.](https://proceedings.mlr.press/v235/carlini24a.html) addresses an output-layer extraction setting, not the decoder backbone. These findings narrow the next research questions without satisfying the mission's full scope.

## Frozen experiment and independent checks

Preregistration c2333f9 preceded fresh teacher generation; implementation c46219a preceded the run. `examples/antipodal_research.py` separates preparation, learner and evaluator. The learner gets only public observations, not private seeds/weights or heldout inputs. The evaluator exclusively claims a consumed marker before private reads, verifies fixed commitments and consumes 64 heldout points once per condition. This is logical process separation on one OS account, not an adversarial OS sandbox.

| Condition | Max absolute heldout E error | Coefficient error versus bias-free formula | Preregistered component criterion |
|---|---:|---:|---|
| Bias-free owned width64/d32/output32 teacher | 2.3314683517128287e-15 | 7.632783294297951e-17 | passed |
| Gate bias 0.7, deliberate misspecification | 1.6174222522202282 | 0.07657613467564302 | failed as intended |

Each uses 1056 calibration evaluations and 128 heldout signed evaluations. Combined measured elapsed time 0.797235666992492 seconds, no external calls, 512 MiB monitored process-tree bound and 120-second initial timeout. Peak RSS was not recorded for this run. Mislabeling the bias-free student's E as full f gives heldout error 3.8982703929648945: the full model has **not** been recovered. Floating-point residuals are measurements, not exact-real certificates.

An independent reviewer reconstructed Q from both sets of public observations (zero coefficient difference), matched all commitments, and verified repeated evaluation is refused before any file read. A separate fresh fixture tested conditional signs, same-even/different-full ambiguity, rank and invalid-input controls. An agent given only the skill and public observations completed create → reconstruct → findings → export, then reloaded the artifact; calibration error 1.39e-17, with no independent heldout claim. Those pilot reports retain pre-format source hashes; the sealed report pins the final formatted source.

Public evidence: `experiments/evidence/antipodal-even-v1/`. Duplicate solution observation files are omitted; each byte-identical canonical copy remains in the condition directory. Private teacher arrays, heldout arrays and evaluator consumption markers stay outside Git. Their hashes in manifests are commitments, not exposed weights.

## Agent-facing delivery and limits

`skills/model-forge/` now routes by actual observation access and documents real CLI/MCP operations, source/runtime discovery, assumptions, scientific gaps and the closed gate. Actual operations are `engine reconstruct_antipodal` and MCP `reconstruct_antipodal`; a project export includes the public quadratic artifact and observation provenance. Metadata declaring a family does not certify the implementation, absence of biases, canonical query provenance or numeric precision. Reports default to heldout_verified=false, numeric_certificate=false, full_glu_recovered=false and original_weights_recovered=false.

26 focused tests cover algebra, controls, persistence/export, worker and real MCP invocation. The same 26 tests passed with zero failures/errors/skips on Linux, Windows and macOS in CI40104672110; scoped mypy and ruff also passed. JUnit and source pin are saved with the evidence. Existing GUI/native acceptance is not extended by this CLI/MCP experiment. Stable v0.1.0/v0.2.0 binaries remain unchanged.

## Next discriminating research

The earlier sealed SwiGLU v1 failure (16/64,55/64,52/64 within5%) remains a failure. Before more teachers, preregister v2 with iteration budgets and no concurrent fit/evaluation jobs. Separate O3 finite observations from isolated continuous-query access. Investigate gate/coupling discovery needed for the odd component, prove relevant degeneracies, and compare any algebraic constraints with the iterative baseline using disjoint validation and sealed heldout data. Attention/backbone composition and reduction from final logits/text to legal internal observations remain open. No claim of Luna equivalence, arbitrary-model reconstruction or original-weight identity follows from this milestone.
