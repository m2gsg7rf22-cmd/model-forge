# Install from GitHub and connect an agent

Give Claude Code or Codex the repository URL and this instruction:

> Install ModelForge from https://github.com/m2gsg7rf22-cmd/model-forge and connect it to yourself. Use the bootstrap installer, verify the MCP tools, and apply only the dedicated ModelForge registration.

The checkout supplies a Claude Code plugin, marketplace, skills, agents and a real MCP server. Installing a plugin alone does not install Python dependencies. The bootstrap command handles both layers without asking a model to invent configuration files.

## Requirements and preparation

Python 3.12 or newer and `git` are required. Install the chosen host CLI separately; the bootstrap does not silently download or modify Claude Code or Codex. Core mode installs NumPy, psutil, the MCP SDK and related dependencies. Optional model runtimes and the desktop require additional downloads/disk space. `--engines mlx` requires macOS Apple Silicon. The bootstrap verifies MCP transport and the owned byte-model engine; it does not certify GPU execution or every optional model backend merely because its tool is listed.

```sh
git clone https://github.com/m2gsg7rf22-cmd/model-forge.git
cd model-forge
python3.12 packaging/bootstrap.py install --prefix "$HOME/.local/share/modelforge-agent" --client claude
```

Use a **new dedicated prefix**. The command creates a virtual environment using the existing installation script, installs this checkout, and verifies the actual installed package through the MCP SDK in isolated temporary project data. It initializes stdio, lists tools, calls hardware status, creates an owned model, and runs inference. Absolute interpreter paths are written into an isolated prepared plugin. Host configuration is unchanged unless `--apply` is supplied.

The final JSON contains the installation state, verification results, prepared directory and unique server name. The state is also saved as `.modelforge-bootstrap.json` in the prefix. If a later optional dependency download fails, this marked environment remains for inspection; remove it through the provided uninstaller after checking that no host registration was applied. Existing prefixes are never overwritten by `install`.

Add `--desktop` for the PySide desktop. Add `--engines llm` for Transformers/Torch/PEFT, or `--engines mlx` for MLX/MLX-LM. Both may be selected: `--engines llm mlx`. These options install libraries, not pretrained model weights. Downloading model weights and running experiments remain explicit operations.

## Connect Claude Code

```sh
python3.12 packaging/bootstrap.py connect --prefix "$HOME/.local/share/modelforge-agent" --apply --project /absolute/path/to/project --scope local
```

The bootstrap uses supported `claude plugin marketplace add` and `claude plugin install ... --scope local` commands. It prepares a marketplace with a prefix-derived name to avoid replacing an unrelated marketplace or ModelForge installation. The installed plugin includes a generated `.mcp.json` invoking the dedicated Python environment. `--scope project` or `--scope user` may be chosen explicitly. The host still controls whether MCP execution requires trust/approval in a session. Restart or reload the host session to discover a newly installed plugin. An authenticated Claude conversation is not automatically created or tested by the installer.

For a CLI outside PATH, pass `--client-executable /absolute/path/to/claude`. If the host executable reports a broken or missing native binary, repair the host installation; ModelForge reports the command failure rather than claiming the plugin connected.

## Connect Codex

Choose `--client codex` when installing the dedicated environment, then use the same `connect --apply` command. Codex registration uses supported `codex mcp add <unique-name> -- <absolute-python> -m modelforge.mcp_server`. Codex MCP registration is **user-level**; the Claude scope option does not change that. The prepared directory also contains a `model-forge` skill and a TOML snippet. To enable project-local skill discovery, copy that skill to your project's `.agents/skills/model-forge` only if that location is unused. The bootstrap does not replace existing project skills.

You can combine `install` and `connect` by adding `--apply` and the desired project/scope options to `install`. Explicitly requesting installation and connection authorizes this dedicated registration; it does not authorize changes to other tools or server entries.

## Verify, update and uninstall

```sh
python3.12 packaging/bootstrap.py verify --prefix "$HOME/.local/share/modelforge-agent"
git pull --ff-only
python3.12 packaging/bootstrap.py update --prefix "$HOME/.local/share/modelforge-agent"
python3.12 packaging/bootstrap.py uninstall --prefix "$HOME/.local/share/modelforge-agent" --apply
```

`update` upgrades the Python package and selected extras from the current checkout, then repeats actual MCP tool calls. The registered interpreter path remains stable. **Prepared/cached plugin skills and agent definitions remain pinned**; to refresh those, uninstall/detach and reinstall into a new dedicated prefix. The engine update path does not silently modify host plugin settings.

`uninstall` accepts only a matching dedicated marker and bootstrap state. For a connected installation, `--apply` authorizes removal of precisely its plugin/marketplace or MCP server using the host CLI, before the dedicated environment is removed. If detachment fails, the environment is retained. Research projects in the normal ModelForge data directory are retained. Unregistered environments can be removed without `--apply`. The process does not uninstall Python, the host CLI, or other plugins.

Supported host methods: [Claude plugin management](https://code.claude.com/docs/en/discover-plugins), [Claude manifest reference](https://code.claude.com/docs/en/plugins-reference), and [Codex MCP configuration](https://developers.openai.com/codex/mcp). Unit tests prepare isolated Claude/Codex integrations and verify ownership boundaries without touching unrelated host settings. Release acceptance records separately identify which host versions and real installation paths were exercised.

## Verification performed for this implementation

A fresh dedicated installation from the development checkout built and installed `modelforge-research 0.2.0` in `/private/tmp/modelforge-bootstrap-verified`. The installed environment's SDK initialized stdio, discovered 28 tools, and successfully invoked hardware status, project creation, byte-model construction, and inference. A generated Claude plugin was then installed using an isolated `CLAUDE_CONFIG_DIR` and an isolated project; `claude plugin list --json` reported it enabled, and `claude mcp list` reported **Connected**. An engine `update` repeated the same installed-package verification successfully. Evidence is retained in the release acceptance records; this does not claim an authenticated Claude conversation or GPU execution. Eight targeted bootstrap tests also pass, including rollback of a partially installed marketplace. A clone of the newly published GitHub revision must be checked separately after publication; the pre-publication verification uses the local checkout of that code.

Finally, `uninstall --apply` successfully removed that isolated plugin and its marketplace at local scope, removed the dedicated virtual environment, and left the isolated settings files with empty ModelForge registration maps. The existing user configuration was never selected.

Windows SDK compatibility: the isolated verifier uses the official MCP v1 transport with a narrow shutdown guard for its [known closed-stream race](https://github.com/modelcontextprotocol/python-sdk/issues/1564). Only closed-stream exceptions after every requested operation has returned are tolerated, with a warning. Initialization, operation failures, timeouts and other exceptions remain failures. No server response is fabricated or skipped.
