# Authorized API research lab

`modelforge.lab.run_experiment` runs explicitly authorized OpenAI-compatible chat requests. Supply a project ID, endpoint, model, and cases with unique `id`, `kind`, and `prompt`. Case kinds are `capability` (requires an explicit `expected` text label), `structured_output` (requires JSON `schema`), `context_retrieval` (requires `expected` and exact `tested_characters`), and `repeat`. Repeat a prompt under different case IDs to collect repeat observations. Label scoring uses exact Unicode equality. Schema validation uses the installed `jsonschema` package locally, without promising provider-native constrained decoding.

The SQLite ledger in the supplied data directory stores experiment configurations and request outcomes. Requests are durably reserved before HTTP. `max_requests` is 1–20 and caps reservations per experiment; the aggregate requested output-token cap is at most 8192. Each case receives floor(total cap / case count) requested tokens. These caps are **not** a monetary budget or a guarantee of provider billing behavior. Input tokens, overhead, and failed calls may cost money. Secrets are read from the named environment variable and are not saved in the ledger. Prompts and outputs are saved locally.

Save the returned `experiment_id`; pass it as `resume_id` with the identical configuration to resume. Completed requests reuse saved results. Failed requests and interrupted reservations are consumed and never retried automatically, because the provider may already have processed or billed them. An interrupted reservation stays `reserved_transport_unknown`. Inspect the ledger and deliberately start a new experiment if a fresh call is needed. Budgets apply per experiment, not globally across unrelated experiment IDs. Retention, encryption and data-directory permissions are the operator's responsibility.

`context_cases([128, 1024, 16000])` creates sentinel retrieval requests at those exact prompt character lengths. Results report observed success at tested character lengths and preserve provider-reported token usage; these are neither exact tokenizer counts nor a true maximum context window. Prompt size includes the retrieval instruction and sentinel, with a 16000-character transport ceiling.

Reports include local schema results, explicit-label exact matches, repeat observations through the existing behavioral evaluator, and transport outcomes. `compare_experiments(left, right)` uses the existing exact-prompt/occurrence comparison for repeated runs or explicitly chosen model versions. Routing, stochastic output, and timing can confound version comparisons. No internal architecture, weight extraction, causal version effect, or general capability reconstruction is established.

Tests use a clearly local HTTP fixture, including response errors and a simulated interruption. No real provider calls or charges are involved.

## Experimental reconstruction service operations (`discovery-lab` source branch)

| Service / MCP operation | Arguments and scope |
| --- | --- |
| `run_discovery` | `project_id`, `seed=17`: owned quadratic suite, public observations and sealed one-time evaluator |
| `run_attention_research` | `project_id`: known single-head composed-score recovery and ambiguity controls |
| `run_representation_research` | `project_id`: owned tiny Llama output head from exposed hidden states/full logits; optional CPU LLM engine required |
| `resume_research` | `repository_path`: validate/recover mission checkpoint; report queue only, no task execution |
| `benchmark_mlx_cache` | `project_id`, `prompt`, `max_new_tokens`, `memory_limit_mb`, `repeats`, `quantize_bits`, `precision`, `compare_logits`: compare fresh KV and full-prefix paths; active adapter must be cleared/merged |

The project ID comes from `modelforge --home ./research-data create NAME`. Invoke the same typed API through `modelforge --home ./research-data engine run_discovery '{"project_id":"ID","seed":17}'` (replace ID), or the named MCP tools. GUI actions are on Research; the cache comparison is on Reconstruction. Reports persist as findings and public research files. Original teacher weights, seeds and heldout inputs persist separately under the data root and are excluded from project ZIP exports. Process separation is not a sandbox against the same OS user. Resource guards sample RSS and enforce time limits; they are not a kernel memory reservation.

These operations do not create the gated private final model. See the individual protocols for observation requirements, fixed thresholds, code/artifact commitments, numerical controls and limits on conclusions.
