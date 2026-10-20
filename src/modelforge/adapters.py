"""Uniform modality adapters with explicit unsupported execution boundaries.

Third-party entry points are listed as metadata; their code is loaded only by
an explicit trusted-plugin call. Built-in inspection never loads plugins.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from .inspector import inspect_model


class UnsupportedModelOperation(RuntimeError):
    """The adapter has no compatible executable computation for this model."""


class ModelAdapter:
    """Common metadata interface; subclasses declare available execution scope."""

    modality = "unknown"

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "metadata_inspection": True,
            "inference": False,
            "inference_scope": "unsupported",
            "exact_reconstruction": False,
        }

    def inspect(self, path: str) -> dict[str, Any]:
        """Return an actual artifact report with this adapter's capability scope."""
        report = inspect_model(path)
        report["adapter"] = self.capabilities
        return report

    def infer(self, path: str, prompt: str) -> dict[str, Any]:
        """Refuse unsupported inference without opening a runtime or weights."""
        raise UnsupportedModelOperation(
            f"{self.modality} inference is unsupported: no compatible runtime is implemented. "
            "Metadata inspection does not establish an executable architecture."
        )


class LanguageAdapter(ModelAdapter):
    """Language artifacts, with inference only for ModelForge's tiny format."""

    modality = "language"

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            **super().capabilities,
            "inference": True,
            "inference_scope": "ModelForge project-owned TinyCausalLM bundles only; arbitrary imported LLMs unsupported",
        }

    def infer(self, path: str, prompt: str) -> dict[str, Any]:
        """Execute only the strictly validated project fixture architecture."""
        from .builder import load_model

        if not Path(path).is_dir():
            raise UnsupportedModelOperation("Language inference requires a supported TinyCausalLM bundle directory")
        if len(prompt.encode("utf-8")) > 16000:
            raise ValueError("Prompt exceeds 16000 UTF-8 bytes")
        report = inspect_model(path)
        if report["metadata"].get("model_type") != "modelforge_tiny_causal":
            # Use the builder's public format constant to avoid drifting format names.
            from .builder import FORMAT

            if report["metadata"].get("model_type") != FORMAT:
                raise UnsupportedModelOperation("Imported language architecture has no runtime adapter; only ModelForge TinyCausalLM is supported")
        model = load_model(path)
        return {"output": model.generate(prompt, max_new_tokens=32),
                "scope": "project-owned tiny causal model", "modality": self.modality}


class MultimodalAdapter(ModelAdapter):
    """Inspect multimodal components without claiming cross-modal execution."""

    modality = "multimodal"


class ImageAdapter(ModelAdapter):
    """Inspect image, ViT, and diffusion artifact metadata."""

    modality = "image"


class VideoAdapter(ModelAdapter):
    """Inspect video artifact metadata; no video generation runtime."""

    modality = "video"


class AudioAdapter(ModelAdapter):
    """Inspect audio artifact metadata; no audio execution runtime."""

    modality = "audio"


class AdapterRegistry:
    """Built-in registry and opt-in trusted entry-point extension loading."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {
            item.modality: item for item in (
                LanguageAdapter(), MultimodalAdapter(), ImageAdapter(),
                VideoAdapter(), AudioAdapter(), ModelAdapter(),
            )
        }

    def get(self, modality: str) -> ModelAdapter:
        """Return a named adapter; never silently select a runtime."""
        if modality not in self._adapters:
            raise ValueError(f"Unknown modality adapter: {modality}")
        return self._adapters[modality]

    def list_plugins(self) -> list[dict[str, str]]:
        """List installed entry-point declarations without importing plugin code."""
        return [{"name": point.name, "value": point.value,
                 "group": point.group} for point in metadata.entry_points(group="modelforge.adapters")]

    def load_plugin(self, name: str, *, trusted: bool = False) -> ModelAdapter:
        """Explicitly import a user-trusted adapter factory; never called on inspection.

        Installed distributions can execute arbitrary Python on load. Callers must
        obtain authorization for the named installed extension before opting in.
        """
        if not trusted:
            raise PermissionError("Adapter plugins execute Python code; explicitly trust this installed plugin before loading")
        matches = [point for point in metadata.entry_points(group="modelforge.adapters") if point.name == name]
        if len(matches) != 1:
            raise ValueError("Plugin must identify exactly one installed entry point")
        factory = matches[0].load()
        candidate = factory()
        if not isinstance(candidate, ModelAdapter):
            raise TypeError("Plugin factory must return a ModelAdapter instance")
        if candidate.modality in self._adapters:
            raise ValueError("Plugin cannot replace a built-in modality adapter")
        self._adapters[candidate.modality] = candidate
        return candidate


def detect_adapter(report: dict[str, Any], registry: AdapterRegistry | None = None) -> ModelAdapter:
    """Select a built-in from declared config; unknown files remain unknown.

    This is a routing heuristic over artifact declarations, not architecture
    verification. It never performs a second inspection or loads plugin code.
    """
    selected = registry or AdapterRegistry()
    config = report.get("metadata", {})
    if not isinstance(config, dict):
        return selected.get("unknown")
    declared = " ".join(str(config.get(key, "")) for key in (
        "model_type", "general.architecture", "_class_name", "architectures", "pipeline_tag"
    )).lower()
    if "video" in declared or any(x in declared for x in ("wan", "hunyuanvideo", "cogvideo")):
        return selected.get("video")
    if (("vision_config" in config or "audio_config" in config) and "text_config" in config
            or any(x in declared for x in ("llava", "qwen2_vl", "qwen2_5_vl", "multimodal", "idefics", "paligemma"))):
        return selected.get("multimodal")
    if any(x in declared for x in ("whisper", "wav2vec", "audio", "speech", "clap")):
        return selected.get("audio")
    if any(x in declared for x in ("diffusion", "unet", "vit", "vision", "resnet", "flux", "image", "vae")):
        return selected.get("image")
    if any(x in declared for x in ("llama", "gpt", "mistral", "qwen", "gemma", "phi", "bert", "causal", "falcon", "bloom")):
        return selected.get("language")
    return selected.get("unknown")
