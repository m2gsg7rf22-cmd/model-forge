"""Local runtime for private ModelForge chat models on Apple Silicon (MLX).

Models live outside Git under MODELFORGE_PRIVATE_HOME (default
~/.local/share/modelforge-private). Only local SafeTensors MLX bundles of
built-in mlx-lm architectures are loaded; no remote code, no hosted inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

DEFAULT_MEMORY_LIMIT_MB = 8192
MAX_MEMORY_LIMIT_MB = 12288
SUPPORTED_MODEL_TYPES = {"llama", "qwen2", "qwen3", "gemma3_text", "mistral", "phi3"}


def private_home() -> Path:
    return Path(os.environ.get("MODELFORGE_PRIVATE_HOME", "~/.local/share/modelforge-private")).expanduser()


def list_private_models(home: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(home).expanduser() if home else private_home() / "models"
    found = []
    for config in sorted(root.glob("*/config.json")) if root.is_dir() else []:
        bundle = config.parent
        try:
            meta = json.loads(config.read_text(encoding="utf-8"))
        except ValueError:
            continue
        card = bundle / "modelforge.json"
        found.append({
            "name": bundle.name, "path": str(bundle), "model_type": meta.get("model_type"),
            "quantization": meta.get("quantization"),
            "size_bytes": sum(p.stat().st_size for p in bundle.glob("*.safetensors")),
            "card": json.loads(card.read_text(encoding="utf-8")) if card.is_file() else None,
        })
    return found


def validate_bundle(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("Model bundle must contain config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {config.get('model_type')!r}")
    if any(root.glob("*.py")) or config.get("auto_map"):
        raise ValueError("Bundles with custom code are refused")
    if not list(root.glob("*.safetensors")) or any(root.glob("*.bin")) or any(root.glob("*.pt")):
        raise ValueError("Only SafeTensors weights are accepted")
    return root


def bundle_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(path).glob("*")):
        if p.is_file() and p.suffix in {".safetensors", ".json", ".jinja", ".txt"}:
            h.update(p.name.encode())
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


class PrivateChatModel:
    """One loaded model; thread-safe generation with explicit load/unload.

    MLX binds GPU streams to the thread that created them, so every MLX call
    for this model runs on one dedicated worker thread; callers on any thread
    (GUI workers, HTTP handlers, MCP) receive streamed chunks through a queue.
    """

    def __init__(self, path: str | Path, memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
                 adapter_path: str | Path | None = None) -> None:
        from concurrent.futures import ThreadPoolExecutor

        if not 1024 <= memory_limit_mb <= MAX_MEMORY_LIMIT_MB:
            raise ValueError(f"memory_limit_mb must be within 1024..{MAX_MEMORY_LIMIT_MB}")
        self.path = validate_bundle(path)
        self.adapter_path = str(Path(adapter_path).expanduser()) if adapter_path else None
        self.memory_limit_mb = memory_limit_mb
        self.model: Any = None
        self.tokenizer: Any = None
        self.load_seconds = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelforge-mlx")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> dict[str, Any]:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        def work() -> None:
            import mlx.core as mx
            from mlx_lm import load

            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal GPU is unavailable")
            mx.set_memory_limit(self.memory_limit_mb * 1024**2)
            start = time.perf_counter()
            self.model, self.tokenizer = load(str(self.path), adapter_path=self.adapter_path)
            self.load_seconds = time.perf_counter() - start

        self._executor.submit(work).result()
        return self.status()

    def unload(self) -> None:
        def work() -> None:
            import gc

            import mlx.core as mx

            self.model = self.tokenizer = None
            gc.collect()
            mx.clear_cache()

        self._executor.submit(work).result()

    def status(self) -> dict[str, Any]:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else {}
        return {
            "path": str(self.path), "adapter": self.adapter_path, "loaded": self.loaded,
            "load_seconds": self.load_seconds, "active_memory_bytes": mx.get_active_memory(),
            "peak_memory_bytes": mx.get_peak_memory(), "memory_limit_bytes": self.memory_limit_mb * 1024**2,
            "device": info.get("device_name", "Apple GPU"), "unified_memory_bytes": info.get("memory_size"),
        }

    def prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        if not messages or any(m.get("role") not in {"system", "user", "assistant", "tool"} for m in messages):
            raise ValueError("messages must be a nonempty list of system/user/assistant/tool turns")
        kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if tools:
            kwargs["tools"] = tools
        return str(self.tokenizer.apply_chat_template(messages, **kwargs))

    def stream(self, messages: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float = 0.0,
               top_p: float = 1.0, tools: list[dict[str, Any]] | None = None,
               stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
        import queue

        if not 1 <= max_tokens <= 8192 or not 0 <= temperature <= 2 or not 0 < top_p <= 1:
            raise ValueError("Invalid generation settings")
        text = self.prompt(messages, tools)
        chunks: queue.Queue[Any] = queue.Queue()
        done = object()

        def work() -> None:
            try:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_sampler

                for response in stream_generate(self.model, self.tokenizer, text, max_tokens=max_tokens,
                                                 sampler=make_sampler(temp=temperature, top_p=top_p)):
                    chunks.put({"text": response.text, "finish_reason": response.finish_reason,
                                "prompt_tokens": response.prompt_tokens,
                                "generation_tokens": response.generation_tokens,
                                "prompt_tps": response.prompt_tps, "generation_tps": response.generation_tps,
                                "peak_memory_gb": response.peak_memory})
                    if stop and stop():
                        break
            except BaseException as exc:  # re-raised in the consumer thread
                chunks.put(exc)
            finally:
                chunks.put(done)

        self._executor.submit(work)
        while (item := chunks.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
        start, parts, last = time.perf_counter(), [], {}
        for chunk in self.stream(messages, **settings):
            parts.append(chunk["text"])
            last = chunk
        reply = "".join(parts)
        from .capability_eval import parse_tool_calls

        return {"reply": reply, "tool_calls": parse_tool_calls(reply), "seconds": time.perf_counter() - start,
                **{k: v for k, v in last.items() if k != "text"}}


class PrivateModelManager:
    """Process-wide holder of at most one loaded private model (GUI/CLI/MCP/API)."""

    def __init__(self) -> None:
        self.current: PrivateChatModel | None = None
        self._lock = threading.Lock()

    def resolve(self, name_or_path: str) -> Path:
        candidate = Path(name_or_path).expanduser()
        if not candidate.is_dir():
            candidate = private_home() / "models" / name_or_path
        return validate_bundle(candidate)

    def load(self, name_or_path: str, memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB) -> dict[str, Any]:
        path = self.resolve(name_or_path)
        with self._lock:
            if self.current and self.current.path == path and self.current.loaded:
                return self.current.status()
            if self.current:
                self.current.unload()
            self.current = PrivateChatModel(path, memory_limit_mb)
            return self.current.load()

    def unload(self) -> dict[str, Any]:
        with self._lock:
            if self.current:
                self.current.unload()
            self.current = None
        return {"loaded": False}

    def status(self) -> dict[str, Any]:
        if self.current is None:
            return {"loaded": False, "models": list_private_models()}
        return {**self.current.status(), "models": list_private_models()}

    def chat(self, messages: list[dict[str, Any]], model: str | None = None, **settings: Any) -> dict[str, Any]:
        if model:
            self.load(model)
        if self.current is None or not self.current.loaded:
            raise RuntimeError("No private model is loaded; load one first")
        return self.current.chat(messages, **settings)


MANAGER = PrivateModelManager()


def serve_api(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Minimal local OpenAI-style /v1/chat/completions endpoint (loopback only)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The private API binds to loopback only")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": m["name"], "object": "model"} for m in list_private_models()]})
            elif self.path == "/status":
                self._send(200, MANAGER.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Request body too large or empty")
                req = json.loads(self.rfile.read(length))
                result = MANAGER.chat(req["messages"], model=req.get("model"),
                                      max_tokens=int(req.get("max_tokens", 512)),
                                      temperature=float(req.get("temperature", 0.0)), tools=req.get("tools"))
            except Exception as exc:  # reported to the client, server stays up
                self._send(400, {"error": str(exc)})
                return
            self._send(200, {"object": "chat.completion", "model": str(MANAGER.current.path.name if MANAGER.current else ""),
                             "choices": [{"index": 0, "finish_reason": result.get("finish_reason"),
                                          "message": {"role": "assistant", "content": result["reply"],
                                                      "tool_calls": result["tool_calls"]}}],
                             "usage": {"prompt_tokens": result.get("prompt_tokens"),
                                       "completion_tokens": result.get("generation_tokens")},
                             "metrics": {"generation_tps": result.get("generation_tps"),
                                         "peak_memory_gb": result.get("peak_memory_gb"), "seconds": result["seconds"]}})

        def log_message(self, *args: Any) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
