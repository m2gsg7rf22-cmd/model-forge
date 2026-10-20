# User guide

Install using [installation instructions](installation.md). Launch `modelforge gui`
from the activated installation environment. The CLI and MCP server can run
without a desktop display. Data defaults to `~/.modelforge`; use
`MODELFORGE_HOME` or CLI `--home /path/to/workspace` to choose a separate workspace.

1. **Create a project.** Enter its name above the pages and click **Create project**.
   Select existing projects from the workspace dropdown. Projects, findings and
   task states persist in SQLite.
2. **Inspect a local artifact.** In **Explorer**, browse to a file, click **Choose model folder…**, or type a model
   directory path, then click **Inspect artifact**. Read the returned status,
   limitations and evidence. Pickle/PyTorch files are refused.
3. **Explore architecture.** Open **Architecture**. Drag to pan, scroll to zoom,
   and hover over nodes or edges for their descriptors. The scene displays up to
   256 nodes; the JSON report retains the complete inventory. `contains` edges
   mean artifact membership; they do not assert computation order. **Load project
   findings** restores saved inspection results.
4. **Run authorized probes.** In **Research**, enter an OpenAI-compatible API base
   URL (for example a local `/v1` endpoint), model identifier, and one prompt per
   line. Repeat a prompt to measure exact-output repeat consistency. Confirm
   testing permission and run. Use HTTPS for remote services. Set
   `MODELFORGE_API_KEY` before launch when needed. Token limits cap requested
   output, while input/provider overhead remains billable. Results are saved,
   never automatically reused as training data.
5. **Build or reconstruct the supported fixture.** In **Reconstruction**, click
   **Build independent tiny fixture**, then enter a prompt and **Run local
   inference**. This creates a small seeded byte-token model. To reconstruct an
   existing supported ModelForge bundle, create an empty project, choose its
   source directory with **Choose source folder…**, and click **Reconstruct
   supported bundle**. The saved report compares logits on finite test prompts.
   Other imported architectures remain inspection-only. **Quantize current
   fixture** creates separate lossy INT8 storage; execution still uses float32 CPU
   arithmetic. The current model is not replaced by its quantized export.
6. **Train its output head.** In **Training**, paste your own authorized text or
   load a UTF-8 file, choose 1–1,000 steps, and click **Train baseline**. Only the
   output projection is trained. Attention and embeddings remain fixed.
7. **Queue work.** In **Agents**, choose a task kind and enter a JSON payload.
   Examples: `{"path":"/models/config.json"}` for inspection,
   `{"text":"hello"}` for inference, or
   `{"provider":"codex","prompt":"Review this project"}` for an external agent.
   External agents require the authorization checkbox and an installed,
   authenticated provider CLI. **Queue job** persists it; **Run pending jobs**
   starts up to the configured one or two jobs. Repeated identical pending work is deduplicated.
8. **Export.** In **Export**, choose a new ZIP path and click **Export project**.
   Findings and project-generated artifacts are included; imported model files
   are not copied. Reports can contain local paths and API responses, so review
   the archive before sharing.

**Dashboard** refreshes actual hardware and tasks. **Settings** accepts JSON
containing `api_endpoint`, `api_model`, `max_workers`, `agent_timeout_seconds`, and
`agent_memory_mb`. The queue has a hard two-worker ceiling;
`max_workers` selects one or two active workers. API endpoint/model preferences
are stored; Research fields must currently be entered explicitly. Secret fields
are rejected; keep API keys in the environment. Agent limits must be integers in the ranges
1–300 seconds and 64–4,096 MiB; invalid values are rejected.

Operations run in background workers and failures appear in their page. Wait for
active operations to finish before closing. Closing while work is active is
blocked to preserve worker state. The CLI provides the same persisted projects:

```sh
modelforge --home ./research-data create "Local model research"
modelforge --home ./research-data projects
modelforge --home ./research-data inspect PROJECT_ID /path/to/config.json
modelforge --home ./research-data findings PROJECT_ID
```

Replace `PROJECT_ID` with the returned ID. For a complete generated-model workflow,
run `modelforge --home ./example-data example --destination ./example-export.zip`.
Use a new destination because exports refuse to overwrite existing files.

For saved API experiment evaluation, use the CLI or MCP. Obtain behavioral finding
IDs from the project's findings, then run:

```sh
modelforge --home ./research-data compare-experiments PROJECT_ID LEFT_ID RIGHT_ID
modelforge --home ./research-data benchmark PROJECT_ID FINDING_ID '{"prompt":"expected output"}'
```

Comparisons align identical prompt text and repeated-sample occurrence, report
unmatched samples, and retain output-change/JSON-rate/latency summaries. Benchmarks
measure exact text matches only against your supplied labels. Probe results
include descriptive rankings of observed JSON validity and repeat consistency;
these do not establish broad correctness, architecture, weights, or statistical
independence. Both evaluation operations save findings without issuing new API
requests or using outputs for training.
