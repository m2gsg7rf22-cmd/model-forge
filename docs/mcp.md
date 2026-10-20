# MCP server

`modelforge-mcp` runs the official MCP Python SDK `FastMCP` server over stdio.
Tools call the same persistent service as the GUI and CLI; no tool returns a
prearranged model report. Start it from the installed environment. The parent
client owns protocol stdin/stdout; do not use the server as an interactive shell.

Generic server configuration, replacing the command with an actual absolute
installed executable path:

```json
{
  "mcpServers": {
    "model-forge": {
      "command": "/absolute/path/.venv/bin/modelforge-mcp",
      "args": [],
      "env": {"MODELFORGE_HOME": "/absolute/path/research-data"}
    }
  }
}
```

On Windows the executable is normally `.venv\\Scripts\\modelforge-mcp.exe`.
Configure provider secrets through the host environment; do not commit API keys.
The server has access to local paths available to its process, so run it under the
same file permissions appropriate for the intended research workspace.

| Tool | Inputs | Result |
| --- | --- | --- |
| `create_project` | `name` | Persistent project |
| `list_projects` | none | Saved projects |
| `inspect_model` | `project_id`, `path` | Saved evidence-backed artifact report |
| `read_findings` | `project_id` | Recorded reports and experiments |
| `build_model` | `project_id` | Independent tiny fixture bundle |
| `reconstruct_model` | `project_id`, `source_path` | Supported bundle reconstruction in an empty project, plus finite logit parity |
| `quantize_model` | `project_id` | Separate lossy INT8 fixture storage; float32 CPU execution |
| `compare_experiments` | `project_id`, `left_id`, `right_id` | Saved matched-prompt behavioral comparison |
| `benchmark_experiment` | `project_id`, `finding_id`, `expected` | Saved exact-output label benchmark |
| `generate` | `project_id`, `text` | CPU output from generated fixture |
| `train_model` | `project_id`, `text`, `steps=100` | Output-head training metrics and saved bundle |
| `probe_model` | `project_id`, `endpoint`, `model`, `prompts`, `authorized=false`, `budget=1024` | Actual bounded text API observations |
| `export_project` | `project_id`, `destination` | Archive path, SHA-256, generated-file count |
| `enqueue_task` | `project_id`, `kind`, `payload` | Queued or existing deduplicated job |
| `run_tasks` | none | One batch of configured 1–2 jobs plus status ledger |
| `task_status` | none | Persistent statuses/results |
| `hardware_status` | none | Current hardware and NumPy CPU backend |

A practical sequence is create project → build fixture → inspect its generated
model directory → generate → train on authorized text → read findings → export.
For imported artifacts, create project → inspect → read findings is the general
supported path; `build_model` creates an independent fixture, not an inferred
replica. A validated supported ModelForge bundle can instead be passed to
`reconstruct_model` in an empty project; its result includes finite-prompt parity.
`quantize_model` exports lossy INT8 storage for the current fixture and does not
promise accelerated INT8 execution.

For API work, obtain user authorization before passing `authorized=true`. Read
behavioral finding IDs with `read_findings`, then compare two via
`compare_experiments`, or call `benchmark_experiment` with `expected` mapping
prompt strings to exact expected output strings. Comparisons pair repeated
samples by occurrence; benchmarks exclude and count unlabeled samples. Probe
results also rank observed JSON validity and repeat consistency descriptively;
no architecture or weight posterior is inferred.

`enqueue_task(kind="agent")` accepts a payload with `provider` (`claude` or
`codex`), `prompt`, and `authorized=true`. It requires an installed authenticated
CLI and uses bounded subprocesses. `run_tasks` waits for its claimed batch; call
again if more queued work remains. This is a persistent queue, not an always-on
scheduler daemon or a provider monetary-budget manager.

The server currently exposes tools only; no separate resource/prompt catalog,
HTTP transport, or remote authentication service is shipped. The table above lists the original tools; the v0.2 additions are listed below. Direct numerical comparison of two arbitrary
supported bundle paths remains `builder.compare_models` in the Python API;
`compare_experiments` compares stored behavioral findings instead.

See [Claude Code](claude-code.md) and [Codex](codex.md) for host setup. The
`packaging/connect.py` helper prepares isolated files and requires `--apply`
before invoking host registration. Real MCP protocol validation and host
integration limitations are tracked in [validation](validation.md).


## Local engines and research tools (0.2)

The server also registers typed Service signatures as MCP tools named
`local_runtime_status`, `install_local_runtime`, `create_local_fixture`,
`attach_local_model`, `local_generate`, `train_local_model`, `train_mlx_model`,
`reconstruct_local`, `mlx_benchmark`, `trace_local_model`, `run_umrre` and
`run_experiment`. See [API arguments](api.md#local-engine-operations-02) and inspect
the client's actual `list_tools` response for the installed version.

A supported HF sequence is `create_project` → `create_local_fixture` (or authorized
`attach_local_model`) → `local_generate` → authorized `train_local_model` →
`local_generate` → rights-gated `reconstruct_local` → `export_project`. These
operations run actual local weights, distinct from the legacy `generate` and
`train_model` byte-fixture tools. MLX generation/training requires its optional
Apple Silicon runtime. MLX adapters support local reload but no merge/export.

`trace_local_model` measures activations and optionally ablates an exact reported
MLP; it does not infer semantic neuron roles. `run_umrre` performs controlled owned
NumPy experiments without remote requests. `run_experiment` issues requests only
with explicit endpoint authorization and enforces its persisted request/output
reservation caps; input costs and provider overhead remain additional.

Installing libraries or listing tools does not demonstrate actual execution on
every hardware backend. Native desktop users can configure a separate companion
Python. Agents can install and verify an isolated environment directly from the
repository using [bootstrap](agent-install.md). Host registration is applied only
when explicitly requested; bootstrap verification calls actual MCP engine tools.
