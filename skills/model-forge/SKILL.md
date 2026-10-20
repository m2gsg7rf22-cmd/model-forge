---
name: model-forge
description: Inspect authorized local AI model metadata, run bounded behavioral studies, and build an independent tiny causal model through ModelForge MCP.
---
Use the model-forge MCP tools. Start with hardware_status, then list_projects or
create_project. For an authorized local path use inspect_model and read_findings.
Keep observed metadata, config declarations, structural inference, and unknowns
separate. Evidence from outputs does not identify architecture or weights.

For a reproducible independent example: build_model, generate, train_model using
user-owned text, then export_project. This builds ModelForge's byte-token tiny
causal model; it does not rebuild arbitrary imported models. Imported model files
are never copied into exports. Review report paths and provider responses before
sharing. Do not train on proprietary API outputs without explicit rights.

probe_model requires explicit authorization and a token budget; input token costs
and provider overhead are additional. Use only permitted endpoints. No bypasses.
External agent jobs require explicit authorization and have provider billing that
cannot be monetarily capped by the CLI adapter. Read task_status for failures.
Unsupported means unsupported; do not describe generated output as exact recovery.
