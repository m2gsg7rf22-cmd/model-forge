# Codex integration

ModelForge exposes an actual stdio MCP server and a filesystem skill. The implementation follows the [official MCP documentation](https://developers.openai.com/codex/mcp/) and [skill documentation](https://developers.openai.com/codex/skills/).

Activate the environment and prepare files:

```sh
python packaging/connect.py codex --destination ./codex-model-forge
```

The prepared `config.toml` has an absolute executable path. Copy the generated `model-forge` folder into your project's `.agents/skills/` for skill discovery. No skill is silently copied into your personal global directories.

Register the server explicitly:

```sh
codex mcp add model-forge -- /absolute/path/to/environment/bin/modelforge-mcp
codex mcp list --json
```

On Windows the executable is `Scripts/modelforge-mcp.exe`. `connect.py codex ... --apply` invokes registration only when you opt in. An alternative is adding the generated `[mcp_servers.model-forge]` section to your chosen configuration while preserving existing settings. `auth_status: unsupported` is expected for a local stdio server that has no OAuth endpoint; it does not indicate a failed local tool operation.

The Skill tells the agent to separate observations, inferences and unknowns, to honor authorization and to avoid claiming arbitrary model recovery. MCP tool execution uses the same SQLite projects and engines as the desktop.

Automated tests exercise the official MCP client/server transport and Codex registration. A completed paid/authenticated Codex model turn is a separate acceptance check; see validation.md. Existing credentials and host settings are never exported to the repository.

## Read-only approval policy

Some noninteractive Codex configurations require tool approval even when the
operation is read-only. The MCP server marks project listing, findings, job status
and hardware tools read-only. A user can explicitly approve one known operation
with a scoped override such as
`-c 'mcp_servers.model-forge.tools.hardware_status.approval_mode="approve"'`.
Do not blanket-approve execution, API probes or exports unless intended. An
actual authenticated Codex turn successfully invoked hardware_status during local
acceptance; this proves a real agent-to-engine call, not arbitrary autonomous
research quality. See validation.md.
