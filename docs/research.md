# Research basis and evidence boundaries

Primary sources reviewed on 2026-10-20. This document distinguishes published results from implemented features. No extraction against a third-party production model was performed for this project.

## Model extraction

[Tramèr et al., Stealing Machine Learning Models via Prediction APIs (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer) demonstrate that prediction interfaces can expose enough information to duplicate certain classifiers with high fidelity. Their experiments cover specific model classes and API properties. This does not establish a general method for recovering every large language model from chat responses. ModelForge implements authorized behavioral measurements, not this paper's extraction attacks.

[Carlini et al., Stealing Part of a Production Language Model (2024)](https://arxiv.org/abs/2403.06634) recover a transformer projection layer up to symmetries under the studied API conditions and identify hidden dimensions. Recovering a particular layer is different from recovering all layers, tokenizer behavior, service routing, training data or full weight tensors. ModelForge does not implement their extraction method and does not claim such access from ordinary text results.

## Model identification

[Oh et al., Towards Reverse-Engineering Black-Box Neural Networks (ICLR 2018)](https://arxiv.org/abs/1711.01768) study inference of model attributes using sequences of queries. Such identification is conditioned on the candidate population and observations. ModelForge's hypothesis labels remain hypotheses: similarity on selected tasks cannot certify a layer count, attention mechanism, model family or parameter count for an arbitrary closed model.

## System identification

[Han, Poggio and Cheung, System Identification of Neural Systems: If We Got It Right, Would We Know? (ICML 2023)](https://proceedings.mlr.press/v202/han23d.html) evaluate neural system identification using known ground-truth networks. They report that comparison performance depends on factors such as stimuli and that functional similarity has limits for identifying higher-level architectural motifs. The application to closed language-model services is our inference from that methodological caution: benchmark agreement should be reported with the probe distribution, conditions and uncertainty, without upgrading it to a structural proof.

## Mechanistic interpretability

[Elhage et al., A Mathematical Framework for Transformer Circuits (2021)](https://transformer-circuits.pub/2021/framework/index.html) analyze small attention-only transformers through QK/OV circuits and paths through the residual stream. The work relies on access to internals and explicitly treats a limited model setting. ModelForge's small causal attention fixture enables runtime and causal masking tests, but it does not implement circuit discovery, activation patching or full TransformerLens analysis. Token-level observations alone do not provide internal activation access.

## Architecture and the independent fixture

[Vaswani et al., Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) introduce the Transformer architecture. ModelForge independently implements the core scaled dot-product causal attention calculation in NumPy for a project-owned fixture. Its single-head tanh model omits the full paper architecture and must not be described as a general Transformer reconstruction engine.

## Fidelity policy

| Evidence | Valid result | Result not established |
|---|---|---|
| Text-only authorized API observations | Measured behavior, consistency, task scores on specified probes | Exact structure, all weights, training data |
| Declared config and tensor metadata | Observed fields and shapes; labeled architectural inferences | Verified executable graph or complete checkpoint |
| Complete authorized weights plus implemented graph/tokenizer | Candidate runnable reconstruction; numerical comparison | Universal equivalence from a finite test alone |
| Project fixture, exported and reloaded | Exact stored float32 tensors; tested numerical parity | Proprietary model identity, useful general language capability |
| Quantized fixture | Smaller lossy stored weights and measured output error | Exact float32 reconstruction or accelerated INT8 compute |

Identifiability is the central limitation. Different internal parameterizations can implement the same observations, and a service can add sampling, hidden prompts, filters and routing. A finite set of input/output pairs cannot in general distinguish every consistent implementation. This is a mathematical scope statement, not a claim that all partial model identification is impossible. All projects should retain raw evidence and label observations, inferences and unsupported claims separately.

Authorized testing does not imply permission to use service outputs for training or redistribute external assets. ModelForge provides a separate user-supplied corpus path for its own small training runtime and preserves provenance declarations; it does not automatically train on API probe results.
