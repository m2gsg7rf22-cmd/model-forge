# Claude Code integration

ModelForge follows the [official plugin layout](https://code.claude.com/docs/en/plugins-reference): `.claude-plugin/plugin.json`, marketplace manifest, `.mcp.json`, skills, agents and commands. No hooks are required.

After installing and activating ModelForge:

```sh
python packaging/connect.py claude --destination ./claude-model-forge
```

This makes a plugin copy whose MCP command is an absolute path into the active environment. It does not edit existing host settings. To explicitly approve registration, choose a new destination and add `--apply`. This invokes:

```sh
claude plugin marketplace add /absolute/path/to/claude-model-forge
claude plugin install model-forge@model-forge-marketplace
claude mcp list
```

In Claude Code 2.1.275+ the published marketplace can also be installed with `/plugin install model-forge --marketplace m2gsg7rf22-cmd/model-forge`; its default `modelforge-mcp` command must be on the host PATH. The absolute-path preparation above avoids desktop PATH differences.

In local Claude Code sessions launched from the desktop app, use the same local plugin install and restart the session after registration. Plugin execution requires that local session to see the executable and have project filesystem access. The remote/cloud environment does not automatically get your local Python environment or model files. Actual desktop-host session behavior beyond local CLI plugin installation is separately listed in validation.md.

Use `/model-forge:research` with an authorized path, or ask the agent to use ModelForge MCP `create_project`, `inspect_model` and `read_findings`. Agent role files guide research; they do not start ten processes automatically. Use the bounded job queue to schedule explicit work.

Before existing host settings change, `connect.py` requires `--apply`. Tests use separate `CLAUDE_CONFIG_DIR` directories so your normal settings remain untouched. Uninstall with the official plugin manager; removing the Python environment separately leaves a stale registration until removed.

Primary documentation: [marketplaces](https://code.claude.com/docs/en/plugin-marketplaces), [official plugins](https://github.com/anthropics/claude-plugins-official). Their code is not vendored. The official directory uses Apache-2.0 at repository level; individual plugin licenses must be checked separately.
