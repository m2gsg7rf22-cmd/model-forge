# Local engine development — 0.2

The `local-model-runtime` development line extends the published v0.1 code. These
features are integrated in the Service, CLI, MCP and PySide desktop; their own
release artifacts and acceptance record must be checked separately from v0.1.

Implemented and exercised on owned small models:

- Complete local GPT-2/Llama SafeTensors validation, actual token generation,
  rights-gated reconstruction, tokenizer/logit checks and provenance.
- Actual full/PEFT LoRA updates, immutable checkpoints, deterministic HF resume,
  adapter reload/merge; guarded CUDA QLoRA remains hardware-unverified.
- Actual Llama MPS/MLX Metal inference and 4/8-bit MLX linear quantization on the
  available Apple Silicon host; CPU/GPU benchmark measurements.
- MLX LoRA and quantized-base LoRA updates with saved-adapter integrity/reload;
  no MLX resume/merge/export.
- Bounded forward-hook activation statistics and reversible MLP ablation on real
  GPT-2/Llama fixtures, including unchanged-weight and hook-cleanup tests.
- Durable authorized API request/output reservation ledgers, schema checks,
  context retrieval and explicit labels.
- Runnable owned UMRRE affine identification, nonidentifiability/symmetry
  constructions, held-out approximation and compression experiments.
- Optional companion interpreter with bounded subprocesses; desktop controls
  use workers, missing dependencies/errors are visible and native base stays
  independent of heavyweight engines.
- Dedicated bootstrap installation from the current checkout, real MCP SDK
  initialization/tool calls and isolated Claude plugin connection were exercised.
  Cloning newly changed code from GitHub requires those changes to be published.

Focused engine/GUI/hook/bootstrap tests are real development evidence, not final
cross-platform package acceptance or language-quality evaluation. In particular,
small random fixtures do not establish practical model quality or large-model
memory capacity. Consult [validation](validation.md), [release notes](release-notes.md)
and the release's evidence artifacts for complete executed results. Historical
v0.1 records are retained. No new all-platform CI result is asserted here.

Detailed scope: [capabilities](capabilities.md), [MLX](mlx.md), [UMRRE](umrre.md),
[training](training.md), [research ledger](research-lab.md) and
[agent installation](agent-install.md).
