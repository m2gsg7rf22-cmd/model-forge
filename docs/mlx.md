# MLX on Apple Silicon

The optional MLX runtime executes actual local Llama forward passes using the
built-in `mlx_lm.models.llama.Model`. It supports CPU and Metal GPU execution,
greedy bounded generation, and real 4/8-bit `mlx.nn.QuantizedLinear` layers
with group size 32. It does not load arbitrary Python from model folders or use
the Hub model resolver. GPT-2 is supported by the HF runtime, not this MLX path.

Install the optional engine through the application runtime installer. Source
development uses `pip install '.[llm,mlx]'`. Tested dependency versions are MLX
0.32.4, MLX-LM 0.29.1, Transformers 4.57.6 and Torch 2.14.1. Keep MLX-LM below
0.30 for this Transformers 4.x integration.

Inputs must be complete, local unquantized HF Llama SafeTensors bundles with a
local fast tokenizer. The common HF preflight verifies exact model tensor names
and shapes, rejects custom mappings, Python/pickle files, missing shards and
symlinks, and checks a conservative memory estimate before loading weights.
Only the built-in installed architecture implementation is instantiated.

`runtime_mlx.generate(path, prompt, max_new_tokens=32, memory_limit_mb=2048,
quantize_bits=0, device="gpu")` returns measured elapsed decode time, generated
token IDs, MLX peak allocator bytes, device identity and validation evidence.
Quantization is performed in memory after the validated checkpoint is loaded.
It reduces the linear weight representation used in computation. Embeddings
and other modules remain floating-point; this is not a claim that every model
tensor uses four bits. Generation uses bounded full-prefix decoding rather than
an unbounded KV cache. The requested token count is a maximum; configured EOS IDs
stop generation earlier. Token IDs retain the terminator as evidence, while decoded
text omits it and other special tokens. Tokenizer special-token insertion matches
the validated local tokenizer, including its BOS processing.

`runtime_mlx.benchmark(path, memory_limit_mb=2048)` measures one warm eight-token
decode for CPU float32, GPU float32, GPU 4-bit and GPU 8-bit on the same prompt.
These are measurements for that model and run. A tiny model can run faster on CPU
because GPU dispatch overhead dominates. No universal speedup is inferred.

`runtime_mlx.train_lora(path, destination, text, steps=5, learning_rate=0.01,
memory_limit_mb=2048, authorized=False, rights="", quantize_bits=0)` performs
actual MLX autodifferentiation and SGD updates on rank-four query/value LoRA
adapters. A quantize_bits value of four trains adapters over quantized linear
base weights. It requires explicit authorization and a rights statement, accepts
2–64 local tokens, caps updates at 100, and writes SafeTensors adapters plus a
provenance JSON with training-text hash, base hashes and adapter hash. Raw text
is not written. These artifacts are adapters, not standalone full checkpoints.
This MLX training path does not support resume or merging yet; use HF training
for deterministic resume and full checkpoint exports.

`runtime_mlx.generate_adapter(path, adapter_path, prompt, max_new_tokens=32,
memory_limit_mb=2048)` verifies the adapter file inventory, hashes, expected LoRA
configuration and tensor shapes before loading a saved adapter for actual GPU
inference. The matching base bundle is still required.

MLX allocation limits are set between 256 and 2048 MiB, cache is capped, and prior
settings are restored on success and failure. The service additionally monitors
worker RSS and elapsed time. MLX limits cover MLX allocations rather than Python,
Torch preflight or all native allocations; RSS monitoring samples periodically.
Neither mechanism reserves memory or provides a kernel-enforced hard quota.

## Measured verification

On the available Apple M4 / 16 GiB machine, a project-owned randomly initialized
19,616-parameter Llama fixture completed actual Metal inference, 4/8-bit inference,
CPU/GPU greedy token parity, LoRA training and integrity-checked adapter reload.
The Torch CPU and MPS paths also produced the same tested token sequence.
The machine-readable release evidence records actual timings and training loss.
No external model weights were downloaded. Training-set loss improvement is not
held-out evaluation and does not establish language quality.

Implementation follows the official [MLX-LM Python and quantization APIs](https://github.com/ml-explore/mlx-lm)
and [MLX memory-management API](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).
