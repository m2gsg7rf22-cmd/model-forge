# Local Hugging Face causal language models

ModelForge executes **complete, local GPT-2 and Llama causal language model bundles** using installed PyTorch and Transformers 4.x. This is a real forward pass and greedy token generation. Other architectures and modalities remain metadata-only unless another documented runtime supports them.

No model downloads, network requests or custom model code are allowed. The directory must contain `config.json`, `tokenizer_config.json`, a fast `tokenizer.json`, and either `model.safetensors` or a complete `model.safetensors.index.json` with every referenced shard. The loader rejects Python/pickle files, `auto_map`, quantized configurations, missing weights, mismatched tensor shapes, out-of-directory shards and symlinks. Every Hugging Face loader uses `local_files_only=True` and `trust_remote_code=False`; the model additionally requires `use_safetensors=True`.

The preflight reads weight headers, verifies tensor sizes/offsets, checks every expected built-in model tensor using a metadata-only Torch model, validates the local tokenizer, and estimates memory before allocating weights. The float32 RAM estimate is three times the stored parameter count times four bytes, plus a 256 MiB inference reserve. It refuses models exceeding the caller's configured budget. This is a conservative estimate, not a hard operating-system memory limit; execution context is capped at 4096 tokens and generation at 256 new tokens. CPU, available Apple MPS and available CUDA devices are supported. Reconstruction parity uses CPU float32 for reproducibility.

```python
from modelforge.runtime_hf import create_owned_fixture, generate, reconstruct

create_owned_fixture("./owned-gpt2", model_type="gpt2")
print(generate("./owned-gpt2", "hello world", device="cpu"))
reconstruct("./owned-gpt2", "./export", ["hello world"],
            redistributable=True, rights="Project-owned fixture; MIT")
```

Fixtures are small randomly initialized real models and real WordLevel tokenizers, generated entirely offline with a seed. They verify the pipeline; they are not useful pretrained assistants. Reconstruction saves weights with safe serialization plus model configuration and tokenizer. It requires an explicit redistribution authorization and rights statement, verifies exact logit equality on the supplied prompts, and records SHA-256 hashes for source and export files. This establishes weight serialization and tested-prompt numerical parity. It does not recover proprietary source code, hidden training data, original training procedures, or establish general behavioral equivalence from a few prompts.

Apple Silicon also has a separate optional [MLX Llama runtime](mlx.md) with real CPU/Metal inference, 4/8-bit linear quantization and LoRA/quantized-base LoRA training with saved-adapter reload. Torch CPU/MPS and MLX Metal were exercised on tiny owned fixtures; CUDA remains hardware-unverified. Selecting an unavailable accelerator fails explicitly. The native desktop runs these heavy engines in an optional companion interpreter; see [installation](installation.md).

References: [Transformers security policy](https://github.com/huggingface/transformers/blob/main/SECURITY.md), [Transformers model loading](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model), [PyTorch meta device](https://docs.pytorch.org/docs/stable/meta.html).
