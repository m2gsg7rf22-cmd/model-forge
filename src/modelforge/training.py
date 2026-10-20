"""Offline, bounded local language-model adaptation with explicit data rights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .adapters import UnsupportedModelOperation


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Symlinks are not permitted in model bundles")
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _destination(source: Path, destination: Path, *, resume: bool = False) -> None:
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError(
            "Source and destination must be separate, non-nested directories"
        )
    if destination.exists() and not resume:
        raise FileExistsError("Destination already exists; no files were overwritten")
    if resume and (not destination.is_dir() or destination.is_symlink()):
        raise ValueError("Resume requires an existing training directory")


def _card(destination: Path, state: dict[str, Any]) -> None:
    text = (
        "# ModelForge local adaptation\n\nResearch training artifact; no quality or capability guarantee.\n\n"
        f"Method: {state['method']}\n\nSource SHA-256: {state['source_sha256']}\n\n"
        "Training data: explicit user-provided text only; the text is not redistributed.\n\n"
        "Rights declaration (verbatim JSON string):\n\n```json\n"
        + json.dumps(state["rights"], ensure_ascii=False)
        + "\n```\n\n"
        "Redistribution is not granted by this tool. Review base-model and dataset rights.\n"
    )
    (destination / "README.md").write_text(text, encoding="utf-8")


def train_local(
    source: str,
    destination: str,
    text: str,
    *,
    method: str = "lora",
    steps: int = 10,
    learning_rate: float = 0.01,
    max_sequence_length: int = 64,
    memory_limit_mb: int = 2048,
    device: str = "auto",
    authorized: bool = False,
    rights: str = "",
    resume: bool = False,
) -> dict[str, Any]:
    """Train using user-owned text; ``steps`` is additional updates on resume.

    SGD without momentum has no hidden optimizer tensors. Immutable SafeTensors
    checkpoints plus the global step reconstruct its exact schedule. Evaluation
    mode disables stochastic dropout while autograd remains enabled.
    """
    if not authorized or not rights.strip():
        raise ValueError(
            "Training requires explicit authorization and a nonempty rights declaration"
        )
    if method not in {"full", "lora", "qlora"}:
        raise ValueError("Method must be full, lora, or qlora")
    if not 1 <= steps <= 1000 or not 2 <= max_sequence_length <= 4096:
        raise ValueError("Require 1..1000 steps and 2..4096 sequence length")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("Learning rate must be finite and in (0, 1]")
    if not text.strip() or len(text.encode("utf-8")) > 1_000_000:
        raise ValueError("Provide 1..1000000 bytes of user-owned training text")
    from .runtime_hf import load_bundle
    import torch

    if Path(destination).is_symlink():
        raise ValueError("Destination may not be a symlink")

    source_path, target = Path(source).resolve(), Path(destination).resolve()
    _destination(source_path, target, resume=resume)
    fingerprint = _digest(source_path)
    data_hash = hashlib.sha256(text.encode()).hexdigest()
    settings: dict[str, Any] = dict(
        source_sha256=fingerprint,
        text_sha256=data_hash,
        method=method,
        learning_rate=learning_rate,
        max_sequence_length=max_sequence_length,
        rights=rights,
        optimizer="SGD(momentum=0,weight_decay=0)",
        warmup_steps=2,
    )
    previous: dict[str, Any] = {}
    if resume:
        _digest(target)  # Reject symlinks throughout existing training output.
        previous = json.loads((target / "training-state.json").read_text(encoding="utf-8"))
        saved_path = target / previous["checkpoint"]
        if not saved_path.resolve().is_relative_to(target):
            raise ValueError("Invalid resume checkpoint")
        if _digest(saved_path) != previous.get("checkpoint_sha256"):
            raise ValueError("Resume checkpoint integrity mismatch")
        if any(previous.get(k) != v for k, v in settings.items()):
            raise ValueError(
                "Resume settings, text, source, and rights must match original run"
            )
    if method == "qlora":
        if (
            device not in {"auto", "cuda", "cuda:0"}
            or not torch.cuda.is_available()
            or importlib.util.find_spec("bitsandbytes") is None
        ):
            raise UnsupportedModelOperation(
                "QLoRA requires supported CUDA hardware and bitsandbytes; CPU and Apple MPS are unsupported"
            )
        from .runtime_hf import preflight
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        preflight(source, memory_limit_mb=memory_limit_mb, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            device_map={"": "cuda:0"},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
            ),
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    else:
        load_path = (
            str(target / previous["checkpoint"])
            if resume and method == "full"
            else source
        )
        model, tokenizer, _runtime = load_bundle(
            load_path, memory_limit_mb=memory_limit_mb, device=device
        )
    if method != "full":
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if resume:
            checkpoint = target / previous["checkpoint"]
            if (
                not checkpoint.resolve().is_relative_to(target)
                or checkpoint.is_symlink()
            ):
                raise ValueError("Invalid resume checkpoint")
            if not (checkpoint / "adapter_model.safetensors").is_file():
                raise ValueError("Resume adapter must use SafeTensors")
            model = PeftModel.from_pretrained(
                model, str(checkpoint), is_trainable=True, local_files_only=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=4,
                    lora_alpha=8,
                    lora_dropout=0.0,
                    target_modules="all-linear",
                    bias="none",
                ),
            )
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_sequence_length
    )
    if encoded["input_ids"].shape[-1] < 2:
        raise ValueError("Training text must contain at least two tokenizer tokens")
    context = getattr(model.config, "max_position_embeddings", max_sequence_length)
    if encoded["input_ids"].shape[-1] > context:
        raise ValueError("Sequence exceeds model context length")
    batch = {
        key: value.to(model.device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Conservative activation allowance; this is admission estimation, not an OS memory cap.
    estimate = (
        parameter_bytes
        + sum(p.numel() * p.element_size() for p in parameters) * 2
        + max_sequence_length**2
        * int(
            getattr(
                model.config, "n_layer", getattr(model.config, "num_hidden_layers", 1)
            )
        )
        * 64
    )
    if estimate > memory_limit_mb * 1024**2:
        raise MemoryError("Estimated training memory exceeds budget")
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    checkpoint_bytes = (
        parameter_bytes
        if method == "full"
        else sum(p.numel() * p.element_size() for p in parameters)
    )
    disk_required = steps * (checkpoint_bytes + 8 * 1024**2)
    if shutil.disk_usage(parent).free < disk_required:
        raise OSError("Insufficient free disk space for immutable training checkpoints")
    optimizer = torch.optim.SGD(
        parameters, lr=learning_rate, momentum=0, weight_decay=0
    )
    model.eval()
    model.config.use_cache = False
    with torch.no_grad():
        initial_loss = float(model(**batch, labels=batch["input_ids"]).loss.item())
    start = int(previous.get("completed_steps", 0))
    losses = list(previous.get("losses", []))
    target.mkdir(parents=True, exist_ok=resume)
    (target / "checkpoints").mkdir(exist_ok=True)
    state: dict[str, Any] = {}
    for step in range(start, start + steps):
        optimizer.param_groups[0]["lr"] = learning_rate * min(1.0, (step + 1) / 2)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=batch["input_ids"]).loss
        if not bool(torch.isfinite(loss)):
            raise ValueError(
                "Nonfinite training loss; latest completed checkpoint remains resumable"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.item()))
        checkpoint = target / "checkpoints" / f"step-{step + 1:06d}"
        checkpoint.mkdir(exist_ok=False)
        model.save_pretrained(str(checkpoint), safe_serialization=True)
        tokenizer.save_pretrained(str(checkpoint))
        _json(
            checkpoint / "modelforge-training.json",
            {
                **settings,
                "completed_steps": step + 1,
                "adapter_weights_sha256": (
                    hashlib.sha256(
                        (checkpoint / "adapter_model.safetensors").read_bytes()
                    ).hexdigest()
                    if method != "full"
                    else None
                ),
            },
        )
        state = {
            **settings,
            "completed_steps": step + 1,
            "losses": losses,
            "checkpoint": str(checkpoint.relative_to(target)),
            "checkpoint_sha256": _digest(checkpoint),
            "device": str(model.device),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "estimated_training_bytes": estimate,
        }
        _json(target / "training-state.json", state)
    with torch.no_grad():
        final_loss = float(model(**batch, labels=batch["input_ids"]).loss.item())
    state.update(
        initial_loss=initial_loss,
        final_loss=final_loss,
        artifact_path=str(target / state["checkpoint"]),
        destination=str(target),
    )
    _json(target / "training-state.json", state)
    _card(target, state)
    return state


def load_adapter_bundle(
    source: str, adapter: str, *, memory_limit_mb: int = 2048, device: str = "cpu"
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a project-provenanced LoRA on its matching, complete safe base."""
    from .runtime_hf import load_bundle
    from peft import PeftModel

    adapter_path = Path(adapter)
    if adapter_path.is_symlink():
        raise ValueError("Adapter directory may not be a symlink")
    adapter_path = adapter_path.resolve()
    fingerprint = _digest(adapter_path)
    if any(
        p.suffix in {".bin", ".pt", ".pth", ".pkl", ".pickle", ".py"}
        for p in adapter_path.rglob("*")
    ):
        raise ValueError("Only data-only SafeTensors adapters are supported")
    weights = adapter_path / "adapter_model.safetensors"
    config_path = adapter_path / "adapter_config.json"
    provenance_path = adapter_path / "modelforge-training.json"
    if (
        not weights.is_file()
        or not config_path.is_file()
        or not provenance_path.is_file()
    ):
        raise ValueError(
            "Adapter requires SafeTensors, configuration, and ModelForge provenance"
        )
    if (
        config_path.stat().st_size > 1024**2
        or provenance_path.stat().st_size > 2 * 1024**2
    ):
        raise ValueError("Oversized adapter metadata")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        config.get("peft_type") != "LORA"
        or config.get("task_type") != "CAUSAL_LM"
        or config.get("auto_mapping")
    ):
        raise ValueError(
            "Only built-in causal LoRA adapters without custom mappings are supported"
        )
    if config.get("init_lora_weights", True) is not True:
        raise ValueError("Custom adapter initialization is unsupported")
    if provenance.get("method") not in {"lora", "qlora"} or provenance.get(
        "source_sha256"
    ) != _digest(Path(source).resolve()):
        raise ValueError("Adapter provenance does not match this base model")
    if (
        provenance.get("adapter_weights_sha256")
        != hashlib.sha256(weights.read_bytes()).hexdigest()
    ):
        raise ValueError("Adapter weight integrity mismatch")
    if weights.stat().st_size > memory_limit_mb * 1024**2 // 2:
        raise MemoryError("Adapter exceeds memory admission budget")
    model, tokenizer, report = load_bundle(
        source, memory_limit_mb=memory_limit_mb, device=device
    )
    model = PeftModel.from_pretrained(
        model, str(adapter_path), local_files_only=True, is_trainable=False
    )
    model.eval()
    report.update(
        adapter_path=str(adapter_path),
        adapter_sha256=fingerprint,
        adapter_source_sha256=provenance["source_sha256"],
    )
    return model, tokenizer, report


def generate_adapter(
    source: str,
    adapter: str,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    memory_limit_mb: int = 2048,
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate real tokens using a verified local LoRA and its matching base."""
    import time
    import torch

    if not prompt.strip() or len(prompt.encode()) > 16000:
        raise ValueError("Prompt must be nonempty and at most 16000 UTF-8 bytes")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256")
    model, tokenizer, report = load_adapter_bundle(
        source, adapter, memory_limit_mb=memory_limit_mb, device=device
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs["input_ids"].shape[-1] + max_new_tokens > min(
        4096, getattr(model.config, "max_position_embeddings", 2048)
    ):
        raise ValueError("Prompt and generation exceed the supported context budget")
    batch = {
        k: v.to(model.device)
        for k, v in inputs.items()
        if k in {"input_ids", "attention_mask"}
    }
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    ids = output[0].tolist()
    return {
        "output": tokenizer.decode(ids, skip_special_tokens=True),
        "token_ids": ids,
        "new_tokens": len(ids) - inputs["input_ids"].shape[-1],
        "elapsed_seconds": time.perf_counter() - start,
        "preflight": report,
        "scope": "local causal language model with verified PEFT LoRA",
    }


def merge_adapter(
    base: str,
    adapter: str,
    destination: str,
    *,
    redistributable: bool = False,
    rights: str = "",
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Export adapter plus its exact base into a new standalone SafeTensors bundle."""
    if not redistributable or not rights.strip():
        raise ValueError(
            "Merging requires explicit redistribution rights and declaration"
        )
    base_path, adapter_path, target = map(
        lambda p: Path(p).resolve(), (base, adapter, destination)
    )
    _destination(base_path, target)
    _destination(adapter_path, target)
    _digest(adapter_path)  # Reject linked payloads before PEFT deserialization.
    if any(
        p.suffix in {".bin", ".pt", ".pkl", ".pickle", ".py"}
        for p in adapter_path.rglob("*")
    ):
        raise ValueError("Only data-only SafeTensors adapters are supported")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError("Adapter must contain adapter_model.safetensors")
    # Require project provenance, rather than silently pairing an unrelated base.
    state_file = adapter_path / "modelforge-training.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state["source_sha256"] != _digest(base_path) or state["method"] not in {
        "lora",
        "qlora",
    }:
        raise ValueError("Adapter provenance does not match this base model")
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("auto_mapping"):
        raise ValueError(
            "Only built-in LoRA adapters without custom mappings are supported"
        )
    model, tokenizer, _ = load_adapter_bundle(base, adapter, device="cpu")
    from .runtime_hf import _inputs
    import torch

    parity_prompts = prompts if prompts is not None else ["hello world"]
    if not 1 <= len(parity_prompts) <= 32:
        raise ValueError("Provide 1–32 merge parity prompts")
    with torch.inference_mode():
        references = [
            model(**_inputs(model, tokenizer, text)).logits.detach().cpu().clone()
            for text in parity_prompts
        ]
    merged = model.merge_and_unload(safe_merge=True)
    comparisons = []
    with torch.inference_mode():
        for text, reference in zip(parity_prompts, references, strict=True):
            actual = merged(**_inputs(merged, tokenizer, text)).logits.detach().cpu()
            error = float((reference - actual).abs().max().item())
            matched = bool(torch.allclose(reference, actual, atol=1e-5, rtol=1e-4))
            if not matched or not math.isfinite(error):
                raise ValueError("Merged adapter failed tested-prompt numerical parity")
            comparisons.append(
                dict(
                    prompt=text,
                    max_absolute_logit_error=error,
                    within_tolerance=matched,
                )
            )
    target.mkdir(parents=True)
    merged.save_pretrained(str(target), safe_serialization=True)
    tokenizer.save_pretrained(str(target))
    provenance = {
        **state,
        "rights": rights,
        "training_rights": state["rights"],
        "adapter_sha256": _digest(adapter_path),
        "operation": "merge",
        "parity": comparisons,
        "parity_scope": "Tested-prompt float32 numerical agreement within atol=1e-5, rtol=1e-4; not exact universal equivalence",
        "destination": str(target),
    }
    _json(target / "provenance.json", provenance)
    _card(target, provenance)
    return provenance
