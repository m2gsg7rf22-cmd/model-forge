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
| `generate` | `project_id`, `text` | CPU output from generated fixture |
| `train_model` | `project_id`, `text`, `steps=100` | Output-head training metrics and saved bundle |
| `probe_model` | `project_id`, `endpoint`, `model`, `prompts`, `authorized=false`, `budget=1024` | Actual bounded text API observations |
| `export_project` | `project_id`, `destination` | Archive path, SHA-256, generated-file count |
| `enqueue_task` | `project_id`, `kind`, `payload` | Queued or existing deduplicated job |
| `run_tasks` | none | One batch of up to two jobs plus status ledger |
| `task_status` | none | Persistent statuses/results |
| `hardware_status` | none | Current hardware and NumPy CPU backend |

A practical sequence is create project → build fixture → inspect its generated
model directory → generate → train on authorized text → read findings → export.
For imported artifacts, create project → inspect → read findings is the supported
path; `build_model` creates an independent fixture, not an inferred replica.
For API work, obtain user authorization before passing `authorized=true`.

`enqueue_task(kind="agent")` accepts a payload with `provider` (`claude` or
`codex`), `prompt`, and `authorized=true`. It requires an installed authenticated
CLI and uses bounded subprocesses. `run_tasks` waits for its claimed batch; call
again if more queued work remains. This is a persistent queue, not an always-on
scheduler daemon or a provider monetary-budget manager.

The server currently exposes tools only; no separate resource/prompt catalog,
HTTP transport, or remote authentication service is shipped. Quantization and
explicit tiny-bundle reconstruction/comparison remain Python APIs.

See [Claude Code](claude-code.md) and [Codex](codex.md) for host setup. The
`packaging/connect.py` helper prepares isolated files and requires `--apply`
before invoking host registration. Real MCP protocol validation and host
integration limitations are tracked in [validation](validation.md).
