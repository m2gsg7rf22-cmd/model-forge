# Capabilities and reconstruction boundaries — 0.2

ModelForge separates artifact evidence, accessible-weight execution and black-box behavioral research. UMRRE's name describes a research objective; it does not certify universal recovery.

| Input / operation | Implemented capability | Boundary |
| --- | --- | --- |
| SafeTensors / HF configuration | Bounded tensor/header inventory, shard validation, tokenizer metadata | Headers alone do not prove graph semantics |
| Complete local GPT-2 / Llama | Built-in installed Transformers model, fast local tokenizer, CPU or available MPS/CUDA device | Local SafeTensors only; no Hub download or remote Python; exact inventory required |
| Local Llama on Apple Silicon | Actual MLX CPU/Metal inference, 4/8-bit linear quantization, measured benchmark | GPT-2 and arbitrary MLX architectures unsupported; original local HF weights required |
| Full / LoRA training | Actual Torch full-parameter or PEFT adapter updates; immutable checkpoints; resume | Authorized corpus/model required; training loss is not held-out quality |
| QLoRA | Explicit CUDA/bitsandbytes eligibility checks and NF4 path | Not hardware-verified on this development host |
| MLX LoRA / quantized-base LoRA | Actual rank-four updates and integrity-checked adapter reload | No MLX resume or merge/export yet |
| Forward tracing / intervention | Up to 64 input tokens, 128 leaf activation summaries; named GPT-2/Llama MLP zero-ablation and numeric logits/KL | Local HF models/adapters; no semantic attribution or inaccessible-model inference |
| Reconstruction | Safe reserialization or HF LoRA merge, provenance, rights gate and finite prompt parity | Complete compatible weights required; does not recover missing weights |
| GGUF v2/v3 | Bounded metadata/tensor descriptors | Execution unsupported |
| ONNX / Core ML | Optional metadata parsers | Execution unsupported |
| Python / pickle checkpoint | Refused, including trusted files | Convert independently outside ModelForge |
| Authorized text API | Durable request/output reservation, retained responses, schema/label/context/repetition measurements | No architecture or weight recovery; no provider dollar cap |
| UMRRE | Controlled affine recovery, nonidentifiability/symmetry proofs by construction, held-out student approximation, compression | Small owned synthetic systems, not frontier models |
| Image / video / audio | Compatible artifact metadata; synthetic UMRRE tensor projections | No generative pipelines or semantic reconstruction |

[Inspector bounds](inspector-sources.md), [local runtime](local-runtime.md), [training](training.md), [MLX](mlx.md), [research lab](research-lab.md) and [UMRRE](umrre.md) provide exact technical limits.

## Evidence and exactness

Configuration is declared evidence. Tensor names support qualified structural inference. ONNX graphs describe serialized edges. Runtime hooks measure actual built-in forward execution. API outputs show only observed endpoint behavior. These are distinct evidence sources.

The metadata inspector always reports `reconstruction.exact=false`. Complete supported weights can pass finite tokenizer/logit/token comparisons; this does not prove universal input parity. Lossy INT8 or MLX 4/8-bit operations cannot recover original floating-point values. Tiny randomly initialized fixtures establish engineering behavior rather than language capability.

A forward module tree is containment rather than execution order. Activation statistics and MLP ablation quantify a particular computation/intervention; they do not identify concepts or prove mechanistic explanations.

UMRRE distinguishes cases where a known affine family is identifiable from sufficient noiseless queries, constructive nonidentifiability, functionally identical different weights and approximate students on a held-out synthetic distribution. These statements do not imply all black-box extraction is impossible or all architectures are recoverable.

## Runtime, hardware and validation

The native app remains a lightweight base. Heavy engines execute in a companion interpreter with sampled process RSS, elapsed-time and output bounds. Missing dependencies produce explicit errors. Memory checks/MLX allocation limits are estimates and monitored limits, not OS reservations or a sandbox for arbitrary code.

Torch GPT-2/Llama CPU execution, full/LoRA training, safe reconstruction, offscreen GUI operations and activation hooks have focused local tests. Tiny Llama MPS and MLX Metal inference, quantization and adapter training/reload were exercised on the available Apple M4/16 GiB host. CUDA and CUDA QLoRA were not hardware-tested. macOS Intel has no validated native package.

Windows/Linux v0.1 CI and native archives are historical evidence for that release. New v0.2 platform packages require their own acceptance record; see [validation](validation.md) and [release notes](release-notes.md). Python 3.12+ is required for source/companion environments. MLX requires macOS Apple Silicon.

The queue admits one or two workers. API output caps exclude input/provider overhead and are not monetary caps. Installed trusted adapter plugins remain an explicit trusted-code extension, never automatically loaded during inspection.

## Sharing boundaries

Attached models remain external references. Raw training checkpoints/adapters are not automatically swept into exports. HF reconstruction or adapter merge requires explicit redistribution permission and a rights statement, producing a separate exportable `hf-reconstruction` bundle. MLX adapters currently remain local. Findings may contain authorized provider text and local paths; review before sharing.
