# Reconstruction research: evidence, distinctions and unresolved obstacles

Reviewed 2026-10-20. Sources below are original research or author-maintained repositories. No third-party extraction code is vendored or executed; UMRRE implements small original controlled experiments using NumPy. No proprietary service is queried.

## Four distinct objectives

1. **Exact parameter recovery:** original numeric parameters under a fixed representation, or a clearly specified numerical tolerance. Byte identity additionally depends on serialization.
2. **Equivalent computation:** a transformed implementation with a justified algebraic relation and stated numerical precision.
3. **Functional equivalence:** equal outputs on all inputs in a specified domain; finite test agreement does not prove it.
4. **Behavioral approximation:** measured error or agreement on a finite evaluation distribution. This does not identify weights or internal architecture.

For a known affine oracle, querying zero and every basis vector recovers the map. With a rank-deficient observation matrix, adding a nullspace vector to weights preserves every observation. For a ReLU network, hidden-unit permutations and positive rescalings with compensating outgoing weights preserve the function. These elementary identities, implemented and evaluated in UMRRE, show both a solvable special case and non-uniqueness under different assumptions. They do not imply all nonlinear extraction is impossible.

## Primary literature and applicable limits

- [Carlini et al., *Cryptanalytic Extraction of Neural Network Models*, 2020](https://arxiv.org/abs/2003.04884): precision extraction for particular ReLU networks exploits their piecewise-linear structure and oracle access. This does not establish a universal method for transformers or text-only APIs. The [authors' repository](https://github.com/google-research/cryptanalytic-model-extraction) is Apache-2.0 and archived as of the review date. ModelForge uses it as literature evidence, not as an integrated extraction engine.
- [Carlini et al., *Stealing part of a production language model*, ICML 2024](https://proceedings.mlr.press/v235/carlini24a.html): extraction of an embedding projection layer up to symmetries under the investigated API conditions. Partial layer recovery and inferred hidden dimension are materially weaker claims than recovery of every transformer layer, training data, or complete computation. UMRRE does not implement or run that attack.
- [Liu Ziyin, *Symmetry Induces Structure and Constraint of Learning*, ICML 2024](https://proceedings.mlr.press/v235/ziyin24a.html): analyzes permutation, rescaling and rotation symmetries. UMRRE's explicit ReLU identity independently illustrates why a function need not determine a unique parameter vector.
- [Hu et al., *LoRA*, 2021](https://arxiv.org/abs/2106.09685): low-rank trainable updates reduce adaptation parameters under evaluated conditions. They do not imply a low-rank copy of an arbitrary pretrained model is exact.
- [Dettmers et al., *QLoRA*, 2023](https://arxiv.org/abs/2305.14314): quantized frozen bases with trainable adapters reduce memory for evaluated fine-tuning tasks. Preservation of task performance is a benchmark claim, not identical logits or lossless parameter compression.
- [Lin et al., *AWQ*, 2023/MLSys 2024](https://arxiv.org/abs/2306.00978): activation-aware low-bit weight quantization and hardware-aware inference address practical resource constraints. UMRRE's INT8 experiment is a simpler independent baseline; it does not implement AWQ kernels or inherit their speedup claims.

## Twelve-obstacle register

Classification depends on the access assumptions. “Information” means unavailable evidence; “mathematical” means a demonstrated ambiguity for a defined observation/model class; “engineering” means a concrete implementation/resource issue; “open” means no solution is demonstrated here.

| Obstacle | Classification under current assumptions | Implemented evidence or investigated technique | Hypothesis / next discriminating experiment |
|---|---|---|---|
| Missing weights | Information; open for complete unknown LLMs | Known affine oracle recovery; literature shows some recoverable special cases | Test whether richer authorized numeric outputs identify a *specified* local model family before scaling |
| Unknown architecture | Information; open | Three candidate random-feature widths selected only on validation MSE | Expand a bounded known-family hypothesis set; distinguish predictive utility from architecture identity |
| Parameter identifiability | Mathematical for deficient affine designs | Rank, nullspace perturbation, unseen witness; nine basis queries recover a known affine map | Measure singular values as queries are added; test noise sensitivity before claiming identification |
| Mathematical non-uniqueness | Mathematical even with full ReLU function access | Explicit permutation/rescaling identity and different numeric weights | Define a canonical representation or report an equivalence class; do not claim unique source weights |
| Insufficient observations | Information; mathematical counterexample | Four observations, eight dimensions admit infinitely many affine solutions | Use experiment design to increase rank; no guarantee outside the assumed family |
| Computational complexity | Engineering; open at large unknown scale | Fixed dimensions and three candidates; measured wall time | Compare query/solve growth with dimension under a strict budget rather than unbounded search |
| Training data requirements | Information and empirical engineering | 256 fit, 128 validation, 256 independent test examples from owned Gaussian teacher | Test learning curves and shifted input distributions; low in-distribution error need not transfer |
| Hardware constraints | Engineering | Fixed small arrays, sampled RSS, payload bytes; optional outer process guard | Compare measured memory and throughput for the *same* accessible model on CPU/MLX; cannot infer results from NumPy |
| Numerical precision | Engineering and observation information | Float64 errors reported, no byte-identity claim; INT8 changes outputs | Repeat with rounded/noisy oracle outputs and analyze conditioning; avoid extrapolating noiseless guarantees |
| Runtime/tokenizer differences | Engineering and missing specification | Synthetic numerical engine has no tokenizer; tokenized-language runtimes require separate tests | Compare identical tokenizer files, IDs, dtype, kernels and decoding rules; text matches alone are insufficient |
| Compression limits | Engineering; information-theoretic under arbitrary lossless claims | SVD, pruning, INT8 and KV output errors with explicit payload sizes | Evaluate task-dependent rate/distortion and speed on held-out data; MoE optimization remains unimplemented |
| Legal/technical access restrictions | Authorization and missing information | Only project-owned synthetic targets; no remote extraction or access-control bypass | Add only user-authorized local artifacts/endpoints, inspect licenses and permitted redistribution before export |

## Research outcomes and honest stopping criteria

PASS applies to measured software or to the stated elementary identities, never to universal reconstruction. A smaller test error is a useful result but not proof of original parameters. The rank-deficient counterexample rules out unique recovery *from those observations without additional assumptions*. The ReLU symmetry rules out unique raw parameter identification from the function alone for that architecture. It does not rule out recovery modulo symmetry. The literature demonstrates that specialized extraction can work, so a blanket “all reconstruction is impossible” statement would also be false.

Original next hypotheses are deliberately modest: use query-design rank/conditioning to select informative experiments; report parameter equivalence classes rather than arbitrary source identity; compare architecture candidates by an explicit held-out error/compute Pareto frontier. The first two are illustrated by implemented linear and symmetry experiments; no novel universal algorithm or scientific discovery is claimed. Evidence is produced as machine-readable JSON by `modelforge.umrre.run_research`, with seeds, costs, numerical errors and environment metadata.
