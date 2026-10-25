---
name: model-forge
description: Reconstruct authorized AI model artifacts or investigate inaccessible models with ModelForge's real CLI/MCP engines, observation-specific methods, and reproducible evidence.
---

# ModelForge

Work toward the user's requested reconstruction, including unresolved cases. Determine what evidence is actually accessible and choose a method that uses only that evidence. Do not turn a successful component experiment into a claim of universal reconstruction.

## Enter the actual backend

Use ModelForge MCP when connected: `hardware_status`, `list_projects` / `create_project`, `inspect_model`, `read_findings`. Discover the available tool schemas; an older installed server may lack research operations. With source or an installed package, the same operations are available through `python -m modelforge.cli --home DATA ...`. Read [execution and observation routes](references/reconstruction.md) for commands, prerequisites and interpretation.

On resuming this repository's mission, read `AGENTS.md`, `docs/RESUME.md` and `.modelforge/research_state.json`, inspect Git status, then call `resume_research(repository_path)` or `scripts/resume_research.py`. Resume validates/recovers the queue; it does not execute arbitrary saved commands. Preserve the private final-model gate and current user authorization. Do not start the gated private model or access its assets as a side effect of reconstruction research.

## Route by what the agent can observe

- **Complete authorized weights:** inspect compatibility, then `attach_local_model`, `reconstruct_local`, reload and independently compare. This imports accessible computation; it is not recovery from a closed API.
- **Exposed hidden states/full logits:** `run_representation_research` demonstrates an owned decoder final-head recovery; it does not recover the backbone.
- **Attention inputs/full probabilities:** `run_attention_research` studies composed-score recovery and factor ambiguity.
- **Queryable isolated GLU block with full vector responses at signed continuous inputs:** read [the antipodal contract](references/antipodal.md), collect the exact documented probes, then call `reconstruct_antipodal`. It reconstructs the even component for any supported hidden width, not all GLU weights. Use independent heldout checks; input metadata is no certificate.
- **Isolated bias-free FP64 SwiGLU with input32/width64/output32 and authorized saved Hessian/multiscale observations:** read [the GLU32 contract](references/glu32.md), then `reconstruct_glu32`, `predict_glu32` and export/reload. This builds a full-block candidate under the fixed profile; generated metadata does not claim heldout verification or original stored-weight recovery.
- **Only final logits or generated text:** use explicitly authorized `probe_model` / `run_experiment`, retain observations and investigate weaker-access methods. Do not synthesize hidden states, isolate internal blocks without evidence, or claim original weights from finite matching.
- **Unknown architecture/modality/access:** retain the full objective, identify the missing observation/algorithm, search primary research, preregister a discriminating owned experiment and add a real tested backend before advertising another reconstruction route. Read [research leads](references/research.md) only when needed.

Declare computation precision and returned-output precision separately; float64 NPZ storage does not imply FP64 computation. For rounded or low-precision observations, read [precision triage](references/precision.md) before choosing or advertising a route.

Distinguish original parameters, equivalent parameterizations, defined-domain function equivalence and finite approximation. Record which was requested and which is supported. Preserve failed runs, source/artifact digests, calibration versus sealed-test boundaries, numerical error, time, memory and query cost. A verifier must not trust fitting-side success labels. Unknowns remain unknown; passing software tests is not a scientific equivalence certificate.

## Preserve existing workflows and rights

The base `build_model`/`generate`/`train_model` workflow is an owned NumPy fixture. Supported GPT-2/Llama engines provide actual inference, full/LoRA training and adapter resume; Apple Silicon MLX provides local inference, 4/8-bit quantization and saved-adapter reload. Use `local_runtime_status` first. Install optional engines only with existing authorization; model weights are a separate download. Unsupported architectures/devices and missing engines must remain explicit errors. MLX adapter resume/merge and hardware-unverified CUDA QLoRA must not be invented.

`trace_local_model` measures activation statistics and reversible ablation, not semantic neuron meaning. `run_umrre`/`run_discovery` execute owned bounded reconstruction studies. `benchmark_mlx_cache` compares actual cache paths; numerical divergence must not be hidden or defaults changed without verification.

Endpoint testing, training on provider outputs, redistribution and external Claude/Codex jobs have separate authorization/rights requirements. Request/output-token caps exclude input costs and overhead; unknown/failed reserved requests are not automatically retried. No source-inspection operation silently starts training. Keep teacher truth, proprietary weights/implementation, secrets and heldout data outside public export/Git. `export_project` exports findings and allowed generated artifacts; inspect the actual archive before sharing.
