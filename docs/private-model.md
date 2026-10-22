# Private local model pipeline

ModelForge builds and runs a private chat model entirely on the local Mac (MLX/Metal). The public repository
contains only the engine: runtime, evaluation, data pipeline, training wrapper, CLI/MCP/API/GUI integration.
Weights, adapters, datasets and evaluation transcripts live outside Git in `MODELFORGE_PRIVATE_HOME`
(default `~/.local/share/modelforge-private`).

## Target identity

"Luna 5.6" was resolved to OpenAI **GPT-5.6 Luna**, a closed hosted model. No authorized API access was available
in the build session, and its outputs are not used for training. ModelForge therefore measures against a
provisional threshold profile (`capability_eval.TARGET_THRESHOLDS`); meeting it is **not** evidence of equivalence to
GPT-5.6 Luna. A real comparison requires running the same procedural suite against the target with authorized
access.

## Route

Training a frontier-class model from scratch is not possible on a 16 GB MacBook Air within hours. The most
capable independent route is: an openly licensed base (Qwen3-4B-Instruct-2507, Apache-2.0, MLX 4-bit, about
2.1 GB), adapted locally by QLoRA on licensed and self-verified data, fused into a standalone private bundle with
its own model card, and served by ModelForge's own runtime. The result is a fine-tuned derivative, not a new
pretrained model, and it is labelled as such.

## Evaluation (`capability_eval.py`)

Ten categories: coding (generated code is executed against hidden tests in an isolated child process), math,
multi-step reasoning, Hebrew, English, instruction following (verifiable constraints), tool use (parsed
`<tool_call>` JSON), long conversations (fact recall across 18 turns), factual accuracy, general assistance.
Items are generated procedurally from seeds; finite pools (facts, coding specs, topics, cities, Hebrew capitals)
are split into disjoint thirds for train/validation/test. Validation is used for keep/revert decisions; test is
reserved for final reporting. Disjointness is enforced by `tests/test_private_model.py`.

## Data (`private_build.py`)

1. Wikidata (CC0) Hebrew country/capital/continent labels via SPARQL, excluding every Hebrew evaluation country.
2. Rejection sampling: the base model answers procedural *train*-split tasks; only checker-verified answers are kept.

No hosted-provider outputs, no unlicensed corpora.

## Usage

```bash
modelforge private list
```

```bash
modelforge private chat --model <name>
```

```bash
modelforge private serve --model <name> --port 8765
```

```bash
modelforge private eval --model <name> --split test --per-category 12 --output report.json
```

The GUI has a **Private Chat** page (load/unload, streaming multi-turn chat, max tokens, temperature, memory and
tokens/s). MCP tools: `private_model_status`, `private_model_load`, `private_model_unload`, `private_chat`. The
local API binds to loopback only and exposes `/v1/chat/completions`, `/v1/models`, `/status`.

Measured results are recorded in docs/RESEARCH_LOG.md and the ledger.

## Current private model (2026-10-22)

`modelforge-private-4b-v1`: QLoRA (16 layers, rank 8 default, 400 steps, batch 1, 768 tokens, peak 3.6 GB) on 306
examples, fused into 4-bit weights (2.26 GB). Sealed test, 12 items per category:

| category | base | private v1 | provisional threshold |
| --- | --- | --- | --- |
| coding | 1.00 | 1.00 | 0.90 |
| math | 0.67 | 0.67 | 0.90 |
| reasoning | 0.92 | 0.83 | 0.90 |
| hebrew | 0.58 | 0.67 | 0.90 |
| english | 1.00 | 1.00 | 0.95 |
| instruction following | 1.00 | 1.00 | 0.90 |
| tool use | 1.00 | 1.00 | 0.95 |
| long conversation | 1.00 | 1.00 | 0.95 |
| factual | 1.00 | 1.00 | 0.90 |
| general assistance | 0.75 | 0.75 | 0.90 |
| mean | 0.892 | 0.892 | |

The fine-tune did not improve the mean; with 12 items per category, single-item differences are not significant.
These easy procedural tasks saturate in several categories and do not measure frontier capability.
Runtime: load 0.4 s, 40–50 generated tokens/s, peak 2.5 GB on M4.

## Install and start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]' mlx-lm
```

Copy or build a private bundle into `~/.local/share/modelforge-private/models/<name>` (never commit it), then:

```bash
.venv/bin/modelforge private chat --model modelforge-private-4b-v1
```

```bash
.venv/bin/modelforge gui
```
