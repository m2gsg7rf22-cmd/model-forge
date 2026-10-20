# Inspector format research and evidence contract

Reviewed 20 October 2026. `src/modelforge/inspector.py` is an original metadata reader;
no upstream implementation was copied or vendored. Optional third-party packages
retain their own licenses. Model files have separate provenance and licensing;
inspecting them grants no redistribution rights.

| Primary reference | Applied behavior | Upstream license |
| --- | --- | --- |
| [SafeTensors format](https://github.com/safetensors/safetensors#format) and [reference validation](https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs) | Eight-byte little-endian length, JSON header, tensor-relative offsets, contiguous complete buffer validation; duplicate keys rejected. | [Apache-2.0](https://github.com/safetensors/safetensors/blob/main/LICENSE) |
| [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | Little-endian v2/v3 typed metadata, tensor descriptors, GGML dimension order, alignment. Explicit limitation for future numeric formats whose byte extent is unknown. | [MIT](https://github.com/ggml-org/ggml/blob/master/LICENSE) |
| [Hugging Face configuration](https://huggingface.co/docs/transformers/main_classes/configuration) | Read local JSON directly; never instantiate remote classes, `AutoModel`, or `trust_remote_code`. | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/api/tokenizer) | Read local tokenizer JSON algorithm, vocabulary entry count, class and special-token metadata; no tokenizer execution or equivalence claim. | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE) |
| [ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData) | Optional ONNX protobuf parser, `load_external_data=False`, graph edges and dense initializer shapes; no execution session. | [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE) |
| [Core ML utilities](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html) | Optional `load_spec` for standalone `.mlmodel` interfaces; no `MLModel` runtime. | [BSD-3-Clause](https://github.com/apple/coremltools/blob/main/LICENSE.txt) |

## Report semantics

`inspect_model(path)` returns JSON-compatible dictionaries. `evidence` entries
record source paths, field values, and one of `observed`, `declared`, or `inferred`.
Tensor descriptors are observed file contents; model family, head counts,
activations, and other configuration values are artifact declarations. Tensor-name
layer classifications are inferences. Neither category proves numerical behavior.
Local tokenizer JSON contributes a declared algorithm, vocabulary entry count, class, and special-token summary; tokenization behavior is not reconstructed. Each tokenizer file is limited to 32 MiB and oversized files are skipped with an explicit limitation.
Every graph node links to evidence. `contains` edges describe inventory;
ONNX edges describe serialized graph connectivity. Neither proves complete
end-to-end computational equivalence. Properties retain their evidence identifiers.

`parameters.total` counts stored tensor elements (or ONNX dense initializer
elements), not independently verified trainable parameters. Buffers, tied storage,
missing model files outside a shard index, subgraphs, and sparse initializers can
make this differ from a model's advertised parameter count. Config-only counts
remain null. Directory shard indexes are checked for missing, misplaced, and
extra tensors. Directory inspection never guesses that weights are complete if
there is no authoritative inventory.

## Bounds and limitations

SafeTensors and GGUF inspect at most 32 MiB of header metadata, up to 100,000
tensors, 16 dimensions per tensor, and 2^64 elements per tensor. SafeTensors
directory headers share a 32 MiB budget and at most 1,024 shards. Weight payloads
are not read. GGUF arrays may have at most 1,000,000 entries and retain only a
64-item preview. GGUF nested arrays, big-endian files, v1, and unknown metadata
types are refused. Unknown GGML tensor types are reported honestly, with their
unvalidated byte extents disclosed. Known quantized types require row alignment
to their block size. Malformed descriptors raise `ValueError`.

ONNX and Core ML protobufs are limited to 32 MiB before their optional parsers run.
ONNX external weights remain unopened. Core ML packages and full Core ML weight
inventories are unsupported. A missing optional parser returns `unsupported` with
an actionable reason. All `.pt`, `.pth`, `.bin`, `.ckpt`, `.pkl`, and `.pickle`
files are refused, including trusted files: convert a trusted checkpoint to
SafeTensors in a separately controlled environment. ModelForge does not offer a
switch that enables arbitrary pickle execution.

The format parser never infers license rights, weight availability, or exact
reconstruction from metadata. All inspection reports set reconstruction to
`metadata_only`, `exact: false`. A separate verified reconstruction workflow must
supply and test the executable computation and weight mapping.
