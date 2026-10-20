# Python and CLI API

The 0.1 API uses JSON-compatible dictionaries rather than a stable versioned
Python schema. Import from the installed `modelforge` package. Failures raise
exceptions in Python; the desktop displays them and the MCP SDK reports tool
errors. The CLI currently prints uncaught error tracebacks.

```python
from modelforge.service import Service

service = Service("./research-data")
try:
    project = service.create_project("Accessible model")
    report = service.inspect(project["id"], "/path/to/config.json")
    assert report["reconstruction"]["exact"] is False
finally:
    service.close()
```

| Service operation | Arguments / behavior |
| --- | --- |
| `create_project(name)` / `projects()` | New ID, name, timestamp / persisted list |
| `inspect(project_id, path)` | Run metadata engine and save an inspection finding |
| `findings(project_id)` | Saved records including parsed `data` |
| `build(project_id)` | Create one independent tiny-model bundle; refuse replacement |
| `infer(project_id, text)` | Generated model only; prompt <=16,000 UTF-8 bytes; up to 32 new tokens |
| `train(project_id, text, steps=100)` | Output head only; text 2–64,000 bytes; 1–1,000 steps |
| `probe(project_id, endpoint, model, prompts, authorized, budget=1024)` | Authorized text API observations; see bounds below |
| `export(project_id, dest)` | New ZIP of findings and generated artifacts; return path/hash/count |
| `enqueue(project_id, kind, payload)` | Deduplicated persisted job; kind inspect/build/train/infer/agent |
| `run_pending(wait=True)` / `tasks()` | Claim up to two jobs / return statuses and results |
| `hardware()` | Current platform, CPU, memory, disk, actual inference backend |
| `settings()` / `save_settings(values)` | Persist allowlisted non-secret JSON preferences |
| `close()` | Wait for worker completion and close the executor |

`modelforge.inspector.inspect_model(path)` returns `schema_version`, `source`,
`format`, `status`, `metadata`, `tensors`, `parameters`, `evidence`, `architecture`,
`limitations`, and `reconstruction`. Evidence IDs link graph nodes and properties
to their source. `parameters.total` can be null; counts reflect stored elements,
not verified trainable parameter counts. Malformed/unsafe inputs raise
`ValueError`; missing paths raise `FileNotFoundError`. See [format bounds](inspector-sources.md).

Builder functions operate on the supported generated bundle:

| Function | Scope |
| --- | --- |
| `create_model(output_dir, seed=7, d_model=32, context_length=64, memory_limit_mb=256, rights=...)` | Keyword-only options; seeded independent initialization |
| `load_model(path, memory_limit_mb=256)` | Validate config/tokenizer/fixed tensor mapping and available bundle hashes |
| `train_model(path, text, steps=100, learning_rate=0.2, output_dir=None, memory_limit_mb=256)` | Keyword-only training options, output-head optimization |
| `quantize_model(path, output_dir)` | Separate lossy INT8 bundle |
| `reconstruct_model(path, output_dir, memory_limit_mb=256)` | Separate float32 bundle plus finite-prompt comparison |
| `compare_models(path_a, path_b, prompts=None)` | Logit error and greedy-token equality for tested prompts |

`research.probe(endpoint, model, prompts, authorized, budget=1024)` accepts HTTPS
or loopback HTTP base URLs and appends `/chat/completions`. Supply 1–20 prompts,
each <=16,000 characters; `len(prompts) <= budget <= 8192`. Each request receives
`max_tokens = budget // len(prompts)`. Responses are limited to 2 MB and requests
time out after 45 seconds. Redirects are refused. The bearer token comes from
`MODELFORGE_API_KEY`. The cap excludes input and provider overhead and is not a
monetary budget. No outputs automatically enter training.

Run `modelforge --help` for CLI commands. All commands share optional `--home`
before the subcommand. `example` is the integrated generated-model workflow;
`enqueue` takes a JSON object string and `run-tasks` executes one pending batch.
Probe, quantization, reconstruction and model comparison are Python/MCP-specific
where exposed; they do not all have dedicated CLI subcommands. See [MCP](mcp.md)
for the exact tool list.
