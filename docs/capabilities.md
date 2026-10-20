# Capabilities and reconstruction boundaries

ModelForge 0.1.0 is an integrated local research workbench. Its executable model
builder supports one project-owned tiny causal architecture. Imported models are
inspected as artifacts, not automatically executed or reconstructed.

## Format matrix

| Source | Working inspection | Execution or reconstruction |
| --- | --- | --- |
| SafeTensors | Tensor names, shapes, dtypes, stored element count, validated offsets; bounded headers | Only the supported ModelForge tiny-model bundle can run |
| GGUF little-endian v2/v3 | Typed metadata, tensor descriptors, quantization types; known-type byte bounds | Unsupported |
| HF config directory | Config, SafeTensors shards and index consistency, tokenizer metadata | Arbitrary Transformers execution unsupported |
| ONNX | Optional `onnx` parser: graph operators/edges and dense initializer shapes; external data unopened | ONNX Runtime execution unsupported |
| Core ML `.mlmodel` | Optional `coremltools`: declared type and input/output interface | Tensor inventory and execution unsupported |
| Core ML `.mlpackage` | Explicit unsupported result | Unsupported |
| PyTorch/pickle checkpoints | Refused, including trusted checkpoints in this release | Unsupported; convert trusted files externally |
| Authorized text API | Actual responses, end-to-end latency, JSON validity, repeated-prompt consistency | No structural or weight recovery |

[Inspector bounds and sources](inspector-sources.md) describe accepted dtypes,
limits, and malformed-file rejection. Metadata is parsed without executing model
code. An optional dependency being absent produces an explicit unsupported result.

## Modality matrix

| Model type | Current capability | Planned extension, not implemented |
| --- | --- | --- |
| Language | Local metadata; authorized text API probes; tiny CPU fixture build/inference/output-head training | General LLM runtime adapters, LoRA/QLoRA, activation interventions |
| Multimodal | Declared nested vision/text/audio configs and compatible tensor metadata | Uniform executable multimodal adapters |
| Images / diffusion / ViT | Compatible tensor inventories, config declarations, name-based layer categories | Diffusers pipelines and image generation |
| Video / audio | Compatible artifact metadata only | Modality-specific probes, pipelines, and reconstruction |

Separate `LanguageAdapter`, `MultimodalAdapter`, `ImageAdapter`, `VideoAdapter`,
and `AudioAdapter` classes share `inspect(path)`, `infer(path, prompt)`, and a
`capabilities` dictionary. Metadata inspection delegates to the same bounded
inspector. Only `LanguageAdapter` runs the supported project-owned tiny bundle;
all other inference calls raise `UnsupportedModelOperation` with a technical
reason. Routing uses artifact declarations and remains a heuristic.

Installed `modelforge.adapters` entry points can be listed without importing their
code. Loading requires an explicit `load_plugin(name, trusted=True)` call;
inspection never loads extensions automatically. A plugin factory must return a
`ModelAdapter` and cannot replace a built-in modality. The plugin path is a
trusted-code extension interface, not a sandbox. Extension candidates and their
actual integration status appear in [ecosystem review](ecosystem.md).

## Evidence and exactness

| Available evidence | What the report establishes | What remains unknown |
| --- | --- | --- |
| Config only | What the artifact declares about layers, heads, activation, context and modality | Actual computation, parameter count, weights, tokenizer equivalence |
| Tensor headers | Stored names, shapes, dtypes, element counts | Training semantics, precise connectivity, completeness without inventory |
| Tensor names | Explicitly inferred layer categories | Verified attention mechanism or execution order |
| ONNX graph | Serialized operators and graph connectivity | External data contents, full subgraph inventory, numerical parity |
| Finite API outputs | Observed behavior for those requests and endpoint version | Architecture, weights, true context limit, universal behavior |
| Complete supported tiny bundle | Loader validates architecture, tensor mapping and tokenizer; finite numerical comparison | Universal parity on all inputs or parity with an unrelated source |

Inspection always reports `reconstruction.exact = false`. The independent seeded
fixture is not a replica of an imported model. Float32 reconstruction of the
supported bundle can show identical logits on tested prompts. INT8 storage is
lossy and cannot recover original pre-quantization weights. See
[reconstruction](reconstruction.md).

## Hardware and operating systems

| Platform | CPU metadata/CLI/MCP/tiny model | Desktop | Accelerators | Validation scope |
| --- | --- | --- | --- | --- |
| macOS Apple Silicon | Supported | PySide6 | NumPy CPU; MLX unsupported | Local development and acceptance host: Darwin arm64 / Python 3.12 |
| macOS Intel | Portable source path, dependencies permitting | PySide6 wheels permitting | CPU only | Not locally validated |
| Windows 10/11 x64 | Portable source path | PySide6 | CPU only; CUDA unsupported | CI workflow supplied; local validation not claimed |
| Linux x64 | Portable source path | PySide6 plus a working display/Qt libraries | CPU only; CUDA unsupported | CI workflow supplied; local validation not claimed |

Python 3.12+ is required. A GPU is unnecessary. Metadata parsing does not load
weight payloads. New fixture construction requires 256 MiB available RAM and
64 MiB free disk through the service. Builder memory estimates also enforce the
configured working-memory limit before allocation. These checks are estimates,
not OS memory reservations. The agent scheduler admits at most two concurrent
jobs; agent subprocesses have time, memory, and log bounds. API output-token
budgets and agent time limits are not monetary provider-account caps.

Packaging, signed distribution, host integrations and executed checks have
separate validation status in [validation](validation.md).
