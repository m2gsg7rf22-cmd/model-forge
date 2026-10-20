# Local language-model training

`modelforge.training.train_local` implements real full-parameter and PEFT LoRA
fine-tuning on complete local GPT-2/Llama causal-language-model bundles. It uses
only SafeTensors, built-in architectures, local files, and
`trust_remote_code=False`. It does not download models or train on API responses.
Install the optional LLM dependencies in the runtime environment first.

```python
from modelforge.training import train_local, merge_adapter

report = train_local(
    '/models/my-base', '/models/my-adaptation',
    'Text I own or have permission to use for this training.',
    method='lora', steps=10, device='cpu',
    authorized=True, rights='I own this text and have rights to train this base.',
)
# report['artifact_path'] names the final immutable checkpoint directory.
continued = train_local(
    '/models/my-base', '/models/my-adaptation',
    'Text I own or have permission to use for this training.',
    method='lora', steps=10, device='cpu', resume=True,
    authorized=True, rights='I own this text and have rights to train this base.',
)
# A standalone merged export requires a separate redistribution declaration.
merge_adapter('/models/my-base', continued['artifact_path'], '/models/my-merged',
              redistributable=True, rights='I have rights to redistribute this export.')
```

The source and output must be separate non-nested directories. New runs refuse
existing destinations. Training accepts at most 1 MB of explicit user-provided
text, 1–1000 updates per call, and a sequence length between 2 and 4096 within
the model's context. Only the first sequence is used; this is a bounded local
research adaptation workflow, not a corpus trainer. At least two tokens are
required. Training records a data digest, not the raw text. There is no automatic
API-output ingestion or provider distillation.

LoRA uses rank 4, alpha 8, all supported linear projections, and freezes base
parameters. Full training updates all parameters. Both use SGD without momentum
or weight decay, gradient clipping at norm 1, and two-step linear warmup. Dropout
is disabled for repeatable single-sequence updates. This intentionally small
optimizer has no hidden tensor state: checkpoint weights, the update cursor,
and the constant schedule suffice to resume. Resume counts additional updates
and rejects changes to the source digest, dataset digest, method, learning rate,
sequence length, or exact rights string. Each completed update writes a new
checkpoint; free disk is checked for the planned checkpoint series; the atomically replaced JSON cursor identifies the latest completed
one. Checkpoint integrity is verified before resume. If a write is interrupted,
an incomplete orphan directory may need removal before retrying that update.
There is no pickle optimizer deserialization.

JSON provenance and a model card retain the rights declaration verbatim. They
record actual initial/final training loss, parameters, runtime device and memory
estimate. Loss on this same short sequence is not held-out quality evidence;
users must evaluate independent validation data before deployment. The package
makes no training-data recovery or arbitrary-model reconstruction claim.

`memory_limit_mb` is an admission estimate, **not an OS-enforced memory cap**.
Model preflight occurs before allocation; training also estimates gradients and
activation allowance. Large architectures may exceed these estimates. Use a
small model and short sequence first. Training is synchronous and bounded by
update count, rather than a wall-clock timeout. Host applications can isolate
it in their worker process.

QLoRA has an NF4 bitsandbytes path gated to CUDA plus installed bitsandbytes.
ModelForge explicitly reports Unsupported on CPU and Apple MPS. CUDA QLoRA is
hardware-conditional and is not verified on this Apple Silicon development host.
CPU training is tested using a project-created offline tiny GPT-2 checkpoint;
MPS and arbitrary pretrained checkpoints require separate hardware/model
validation. Successful toy adaptation does not establish useful language quality.

Implementation references: [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora),
[Transformers bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes),
and [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html).

Queue privacy: queued task payloads, including explicitly supplied training text, are persisted in the local SQLite task ledger for execution and recovery. The model artifact/provenance stores a text digest rather than the corpus. Project findings and queued payloads are local research data; review them before sharing.
