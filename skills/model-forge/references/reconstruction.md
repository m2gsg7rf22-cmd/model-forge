# Execute the reconstruction routes

Use an installed ModelForge environment or run from its source checkout with `PYTHONPATH=src`. The optional source bootstrap is `python packaging/bootstrap.py install --client codex|claude --prefix PATH`; applying host registration is a separate user-requested connection. Existing installation guidance is `docs/agent-install.md`; preserve unrelated host configuration.

Create a persistent workspace/project through MCP `create_project(name)` or:

```sh
python -m modelforge.cli --home /absolute/research-data create 'Reconstruction'
python -m modelforge.cli --home /absolute/research-data engine local_runtime_status '{}'
```

For the following commands, replace PROJECT_ID with the returned ID and keep data paths absolute. Every `engine` argument is a JSON object matching the actual service method, not a shell program.

| Evidence / operation | MCP and CLI engine name | Required arguments | What it establishes |
| --- | --- | --- | --- |
| Exposed-head owned study | `run_representation_research` | `project_id` | Final linear head recovery under stated hidden-state/logit access |
| Attention owned study | `run_attention_research` | `project_id` | Composed score matrices and gauge controls |
| Quadratic nonlinear owned study | `run_discovery` | `project_id`, optional `seed` | Canonical polynomial coefficients with sealed controls |
| Paired isolated GLU responses | `reconstruct_antipodal` | `project_id`, `observations_path` | Quadratic even component; no gate discovery/full model |
| Repository recovery | `resume_research` | `repository_path` | Checked checkpoint/eligible jobs, report only |
| Actual full-prefix/KV comparison | `benchmark_mlx_cache` | `project_id`, optional prompt/token/memory/repeat/precision controls | Finite timing/numeric comparison; requires a compatible bound model |

```sh
python -m modelforge.cli --home /absolute/research-data engine reconstruct_antipodal '{"project_id":"PROJECT_ID","observations_path":"/absolute/observations.json"}'
python -m modelforge.cli --home /absolute/research-data findings PROJECT_ID
python -m modelforge.cli --home /absolute/research-data export PROJECT_ID /absolute/new-export.zip
```

Inspect `reconstruction` / `limitations` in artifact reports. Metadata inspection does not execute GGUF or unknown architecture code. For complete supported HF weights, `attach_local_model` binds an authorized model; `reconstruct_local` needs explicit redistribution permission and rights statement. Check the live schemas and `docs/local-runtime.md` before issuing those operations. Independently reload/compare the resulting artifact on heldout prompts; finite parity does not prove equality for every input.

The existing O3 SwiGLU optimizer is `modelforge.swiglu_identification.identify`, with iteration and wall-clock limits. It is a Python research API, not an MCP tool or a completed decoder reconstruction. Its preregistered v1 failed the per-teacher criterion; `docs/swiglu-identification.md` and the ledger retain that failure. The antipodal route adds different constraints; it does not erase that result or grant isolated-query access from passive O3 traces.
