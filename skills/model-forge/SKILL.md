---
name: model-forge
description: Inspect authorized AI artifacts, execute compatible local GPT-2/Llama models, run bounded reconstruction research and training, and retain evidence through ModelForge MCP.
---
Use the model-forge MCP server. Begin with hardware_status and list_projects or
create_project. For an authorized artifact use inspect_model and read_findings.
Distinguish measured metadata, configuration declarations, runtime measurements,
behavioral observations, inferred structure and unknowns. API outputs do not
identify original architecture or weights.

The original lightweight owned example uses build_model, generate and train_model
(output-head training on a NumPy byte fixture). For a real local language engine,
check local_runtime_status, then create_local_fixture (llama/gpt2) or obtain
explicit authorization for attach_local_model. Use local_generate and authorized
train_local_model for HF full/LoRA updates and resume. Apple Silicon MLX supports
local Llama inference, 4/8-bit linear quantization, benchmarks and train_mlx_model
with adapter reload. MLX resume/merge is unsupported. CUDA QLoRA is guarded but
not hardware-verified on the development host. Missing dependencies or unsupported
architectures/devices must be reported as errors, never substituted silently.

Optional engine installation requires explicit permission to download/install
libraries through install_local_runtime. It creates a dedicated companion Python
environment; it does not download model weights. Repository bootstrap instructions
are in docs/agent-install.md and use packaging/bootstrap.py. Apply host registration
only when the user has requested connection; do not replace unrelated entries.

trace_local_model measures bounded activation statistics and optionally zeroes an
exact reported MLP for a second forward pass. Do not claim neuron semantics or
hidden internals from API behavior. run_umrre executes small owned NumPy research
experiments: exact affine recovery under stated assumptions, nonidentifiability,
symmetry, held-out approximation and compression. Its name is not a universal
closed-model recovery claim.

probe_model and run_experiment require explicit endpoint testing authorization.
Use request/output-token caps; input costs and provider overhead are additional,
and these are not dollar caps. Durable experiment resume does not retry reserved
unknown/failed requests automatically. Never train on API outputs automatically;
obtain separate model/corpus rights before training.

reconstruct_local requires explicit redistribution permission and a rights
statement, including for HF adapter merge. Finite prompt parity is not universal
identity. Attached model files and raw training trees are excluded from export;
explicitly reconstructed HF bundles are exportable. MLX adapters remain local.
Review report paths/provider responses before export_project and sharing.

External Claude/Codex jobs require explicit authorization and an installed
account-authenticated CLI. Provider billing remains external. Read task_status
for failures. Do not describe fixtures, approximate students, quantized outputs or
unsupported operations as exact recovery of inaccessible weights.
