# Upstream ecosystem review

Primary project pages and linked license texts reviewed 20 October 2026. These
projects informed scope and extension choices. ModelForge does not vendor their
source or claim that their capabilities are implemented merely because the
projects were researched. A model's weights, training data, and benchmark data
have licenses separate from a framework's code license.

| Project and inspected primary source | License | Actual ModelForge relationship |
| --- | --- | --- |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [license](https://github.com/TransformerLensOrg/TransformerLens/blob/main/LICENSE) | MIT | Reference only. Activation hooks and mechanistic interpretability require an accessible supported model and runtime; no activation-capture integration ships here. |
| [Transformers](https://github.com/huggingface/transformers), [configuration API](https://huggingface.co/docs/transformers/main_classes/configuration), [license](https://github.com/huggingface/transformers/blob/main/LICENSE) | Apache-2.0 | File-format interoperability: local config JSON and SafeTensors shards. ModelForge does not depend on Transformers, download models, invoke AutoModel, or execute remote model code. |
| [SafeTensors](https://github.com/safetensors/safetensors), [format and parser source](https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs), [license](https://github.com/safetensors/safetensors/blob/main/LICENSE) | Apache-2.0 | Integrated format through an original bounded Python metadata reader. No safetensors package dependency. Exact behavior is documented in [inspector sources](inspector-sources.md). |
| [MLX-LM](https://github.com/ml-explore/mlx-lm), [license](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE) | MIT | Reference only. Apple Silicon generation, low-rank/full fine-tuning, and quantization are useful future runtime backends; no MLX execution is currently implemented. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) | MIT | Integrated GGUF metadata interoperability through an original parser. No llama.cpp binary is bundled, and importing GGUF does not provide GGUF inference. |
| [Diffusers](https://github.com/huggingface/diffusers), [license](https://github.com/huggingface/diffusers/blob/main/LICENSE) | Apache-2.0 | Reference only. Pipeline abstractions informed the decision to distinguish modality adapters; diffusion image/video generation is unsupported. Compatible SafeTensors components can be inventoried individually. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), [license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | MIT | Reference only. A future graph-execution backend. Current optional metadata inspection uses the distinct [ONNX package](https://github.com/onnx/onnx) (Apache-2.0), never InferenceSession or custom-op loading. |
| [LitGPT](https://github.com/Lightning-AI/litgpt), [license](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.md) | Apache-2.0 | Reference only. Training and LoRA/QLoRA recipes were reviewed as future backend candidates. ModelForge's independent tiny training workflow does not train arbitrary LitGPT models. |
| [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [license](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE.md) | MIT | Reference only. Reproducible benchmark configuration and backend separation are useful future integration points. Current behavioral tests are ModelForge's own probes, not harness benchmark scores. |

The lightweight metadata path remains usable on CPU without Torch, Transformers,
MLX, or CUDA. Optional package imports occur only when the corresponding file
adapter is used. Runtime backends must perform independent resource checks and
must not turn a declared architecture into an assertion of exact reconstruction.
