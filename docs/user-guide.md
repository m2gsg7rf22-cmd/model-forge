# User guide

Launch `modelforge gui` from an installed environment, or open the native application. Data defaults to `~/.modelforge`; override with `MODELFORGE_HOME` or CLI `--home /path`. Create/select a project above the nine pages. Results and errors remain visible below each page's controls. Long forms scroll; engine panels are tabs.

## Inspect and visualize

In **Explorer**, select a local artifact or directory and click **Inspect artifact**. Read evidence and limitations. Inspection does not execute model code; pickle/Python model files are refused. **Architecture** displays discovered nodes: pan, zoom and hover for evidence. Containment is not execution order.

After attaching a supported local HF model, **Architecture** can measure real activations for a prompt of at most 64 tokens. Run without an MLP name to obtain `ablation_modules`; optionally enter one exact name and rerun for a reversible zero-output intervention. The report measures numeric changes, not neuron meaning. Up to 128 leaf modules are summarized.

## Enable optional engines

Open **Settings → Optional local engines** and inspect runtime status. The native base does not include all Torch/MLX dependencies. Select `llm` or `mlx`, tick explicit download/install consent and click **Install optional engine**. Installation requires `uv` or Python 3.12+, network access and at least 3 GiB free disk. The dedicated environment is not silently overwritten. `mlx` requires Apple Silicon and also supplies the common validation dependencies.

Alternatively, install ModelForge with `.[llm]` or `.[llm,mlx]` into an existing dedicated Python environment and save its absolute interpreter path. An installed library or listed tool does not establish that every device/backend works. Unsupported devices or missing dependencies produce visible errors.

## Run an accessible language model

1. Open **Reconstruction → Local GPT-2 / Llama / MLX**.
2. For an offline engineering example choose `llama` or `gpt2` and **Create owned offline language-model fixture**. These random tiny weights have no claim to language quality.
3. For your own model choose a complete local GPT-2/Llama SafeTensors folder, confirm authorization and **Validate and attach local model**. External source weights stay in place. Local tokenizer files and exact built-in tensor inventory are required; custom Python, pickle, missing shards and remote mappings are rejected.
4. Select the Torch device and memory budget, enter a prompt and generate. Choose backend `mlx` for supported Apple Silicon Llama execution; select 4 or 8 bits for actual in-memory linear quantization. Use 0 for original precision. GPT-2 uses backend `hf`.
5. **Run MLX GPU benchmark** records actual CPU/GPU and quantized runs. Tiny models may run faster on CPU; interpret the measured report rather than assuming a speedup.
6. To produce an exportable HF checkpoint, supply a rights statement and redistribution permission, then **Reconstruct and check prompt parity**. HF LoRA is merged into a separate bundle. MLX adapter merge/export is currently unsupported.

The **Baseline fixture** tab retains the original NumPy byte model, source reconstruction, lossy INT8 storage and inference. That INT8 path uses dequantized float32 computation; it is distinct from MLX quantized linear execution.

## Train local weights or adapters

Open **Training → Full / LoRA training**. Paste authorized text and enter a model/dataset rights statement. Select full or LoRA training, total target steps, learning rate and sequence limit. The device/memory settings come from Reconstruction. Confirm training permission and click **Train attached model**.

HF checkpoints are immutable per step. To continue, choose **Resume the saved training checkpoint**, retain the same source, corpus and immutable hyperparameters, and increase total target steps. An integrity mismatch fails explicitly. Trained HF models/adapters become active for local inference. QLoRA requires CUDA and bitsandbytes; it has not been hardware-verified on the development machine.

**Train MLX LoRA on Apple GPU** uses the same corpus, rights, step/rate and selected quantization bits. It performs actual adapter updates and supports a quantized base. It does not support Resume or merging; selecting Resume produces an explicit error. The saved MLX adapter can be reloaded for local MLX inference.

Losses describe the supplied training corpus, not held-out quality. API outputs do not enter any training corpus automatically. The original **Baseline fixture** tab still trains only the NumPy model's output projection.

## Research an endpoint or run UMRRE

In **Research**, enter a permitted OpenAI-compatible endpoint, its model ID and output-token budget. Confirm ownership/testing permission. A local endpoint may use loopback HTTP; remote endpoints require HTTPS. Set `MODELFORGE_API_KEY` before launch when necessary. Secrets are never saved in settings.

The first tab supports line-separated probes. **Durable experiments / UMRRE** accepts a JSON array of cases and a maximum request count. It uses the endpoint, model, authorization and token cap from the first tab. An optional experiment ID resumes only unreserved requests. Reserved/failed/unknown requests remain charged to the ledger; they are not silently retried. Token limits exclude input/provider overhead and are not dollar caps. JSON Schema cases reject external references. See [research lab](research-lab.md) for case examples.

**Run local UMRRE research** runs bounded owned synthetic affine, symmetry, approximation and compression experiments. It uses NumPy, makes no API calls and needs no optional heavyweight runtime. Read the explicit assumptions and counterexamples in [UMRRE](umrre.md).

## Queue and export

**Agents** persists jobs and runs up to the configured one or two workers. External Claude/Codex jobs require explicit authorization and an installed authenticated provider CLI. Billing remains with the provider. Duplicate pending work is deduplicated; interrupted states are retained.

**Export** writes to a new ZIP path. It includes findings, original generated assets, UMRRE artifacts and explicitly authorized `hf-reconstruction` files. It does not copy attached source directories or raw training-checkpoint trees. Reports can contain local paths/provider outputs. Review before sharing. Existing ZIP files are not overwritten.

**Settings** accepts allowlisted JSON keys `api_endpoint`, `api_model`, `max_workers`, `agent_timeout_seconds`, `agent_memory_mb` and `runtime_python`. API secrets are rejected. Closing during active operations is blocked to preserve worker state.

## CLI examples

```sh
modelforge --home ./research-data create "Accessible local model"
modelforge --home ./research-data engine create_local_fixture '{"project_id":"PROJECT_ID","model_type":"llama"}'
modelforge --home ./research-data engine local_generate '{"project_id":"PROJECT_ID","text":"hello world","max_new_tokens":4,"device":"cpu"}'
modelforge --home ./research-data engine run_umrre '{"project_id":"PROJECT_ID","seed":7}'
modelforge --home ./research-data findings PROJECT_ID
```

Replace `PROJECT_ID` with the ID returned by `create`. The complete owned Llama workflow is `local-example --destination ./new-local-example.zip` with the `llm` extra. The original lightweight workflow is `example --destination ./new-byte-example.zip`. See [API](api.md), [MCP](mcp.md) and [agent bootstrap](agent-install.md).
