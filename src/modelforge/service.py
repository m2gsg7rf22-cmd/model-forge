"""Persistent project service shared by desktop, CLI and MCP."""

from __future__ import annotations
import concurrent.futures
from contextlib import contextmanager
from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import threading
import time
from typing import Any
import uuid
import zipfile
import psutil
from . import research


class Service:
    """SQLite backed research workspace, bounded to two local worker threads."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.root = Path(
            data_dir or os.environ.get("MODELFORGE_HOME") or Path.home() / ".modelforge"
        ).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "projects.sqlite"
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,name TEXT NOT NULL,created REAL);
            CREATE TABLE IF NOT EXISTS findings(id TEXT PRIMARY KEY,project_id TEXT,kind TEXT,data TEXT,created REAL);
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,project_id TEXT,kind TEXT,payload TEXT,status TEXT,result TEXT,created REAL,pid INTEGER);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS local_models(project_id TEXT PRIMARY KEY,data TEXT NOT NULL);
            """)
            # Only requeue interrupted processes; active workers in another process retain ownership.
            for row in db.execute(
                "SELECT id,pid FROM tasks WHERE status='running'"
            ).fetchall():
                if row[1] is None or not psutil.pid_exists(row[1]):
                    db.execute(
                        "UPDATE tasks SET status='queued',pid=NULL WHERE id=?",
                        (row[0],),
                    )

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_project(self, name: str) -> dict[str, Any]:
        if not name.strip() or len(name) > 200:
            raise ValueError("Project name must have 1–200 characters")
        item: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "name": name.strip(),
            "created": time.time(),
        }
        with self._db() as db:
            db.execute("INSERT INTO projects VALUES(:id,:name,:created)", item)
        (self.root / item["id"]).mkdir()
        return item

    def projects(self) -> list[dict[str, Any]]:
        with self._db() as db:
            return [
                dict(r)
                for r in db.execute("SELECT * FROM projects ORDER BY created DESC")
            ]

    def _project(self, project_id: str) -> Path:
        with self._db() as db:
            if not db.execute(
                "SELECT 1 FROM projects WHERE id=?", (project_id,)
            ).fetchone():
                raise ValueError("Unknown project")
        return self.root / project_id

    def _record(
        self, project_id: str, kind: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        self._project(project_id)
        with self._db() as db:
            db.execute(
                "INSERT INTO findings VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, project_id, kind, json.dumps(data), time.time()),
            )
        return data

    def findings(self, project_id: str) -> list[dict[str, Any]]:
        self._project(project_id)
        with self._db() as db:
            return [
                {**dict(r), "data": json.loads(r["data"])}
                for r in db.execute(
                    "SELECT * FROM findings WHERE project_id=? ORDER BY created DESC",
                    (project_id,),
                )
            ]

    def inspect(self, project_id: str, path: str) -> dict[str, Any]:
        from .inspector import inspect_model

        self._project(project_id)
        from .adapters import detect_adapter

        report = inspect_model(path)
        report["adapter"] = detect_adapter(report).capabilities
        report["adapter"]["inference_supported_for_artifact"] = (
            report.get("metadata", {}).get("model_type") == "modelforge-tiny-causal-v1"
        )
        return self._record(project_id, "inspection", report)

    def build(self, project_id: str) -> dict[str, Any]:
        from .builder import create_model

        dest = self._project(project_id) / "model"
        if dest.exists():
            raise ValueError(
                "Project already has a model; create a new project to build another"
            )
        if (
            psutil.virtual_memory().available < 256 * 1024**2
            or shutil.disk_usage(self.root).free < 64 * 1024**2
        ):
            raise MemoryError("Need 256MB available memory and 64MB free disk")
        return self._record(project_id, "build", create_model(str(dest)))

    def reconstruct(self, project_id: str, source_path: str) -> dict[str, Any]:
        """Reconstruct only a validated supported fixture, preserving accessible tensors."""
        from .builder import reconstruct_model

        destination = self._project(project_id) / "model"
        if destination.exists():
            raise ValueError("Use an empty project for reconstruction")
        with self._lock:
            result = reconstruct_model(source_path, str(destination))
        return self._record(project_id, "reconstruction", result)

    def quantize(self, project_id: str) -> dict[str, Any]:
        """Create lossy INT8 fixture storage; CPU computation remains float32."""
        from .builder import quantize_model

        project = self._project(project_id)
        with self._lock:
            result = quantize_model(str(project / "model"), str(project / "quantized"))
        return self._record(project_id, "quantization", result)

    def compare_experiments(
        self, project_id: str, left_id: str, right_id: str
    ) -> dict[str, Any]:
        """Compare two stored behavioral reports by exact prompt matches."""
        from .evaluation import compare_reports

        reports = {
            row["id"]: row["data"]
            for row in self.findings(project_id)
            if row["kind"] == "behavior"
        }
        if left_id not in reports or right_id not in reports:
            raise ValueError("Choose two behavioral finding IDs in this project")
        return self._record(
            project_id,
            "comparison",
            compare_reports(reports[left_id], reports[right_id]),
        )

    def benchmark(
        self, project_id: str, finding_id: str, expected: dict[str, str]
    ) -> dict[str, Any]:
        """Score a stored report using explicit user-supplied exact text labels."""
        from .evaluation import benchmark_report

        reports = {
            row["id"]: row["data"]
            for row in self.findings(project_id)
            if row["kind"] == "behavior"
        }
        if finding_id not in reports:
            raise ValueError("Choose a behavioral finding ID in this project")
        return self._record(
            project_id, "benchmark", benchmark_report(reports[finding_id], expected)
        )

    def infer(self, project_id: str, text: str) -> dict[str, Any]:
        from .builder import load_model

        if len(text.encode()) > 16000:
            raise ValueError("Prompt too long")
        model = load_model(str(self._project(project_id) / "model"))
        return {
            "output": model.generate(text, max_new_tokens=32),
            "scope": "project-owned tiny causal model",
        }

    def train(self, project_id: str, text: str, steps: int = 100) -> dict[str, Any]:
        from .builder import train_model

        if not 1 <= steps <= 1000 or not 2 <= len(text.encode()) <= 64000:
            raise ValueError("Use 1–1000 steps and 2–64000 bytes of authorized text")
        with self._lock:
            result = train_model(
                str(self._project(project_id) / "model"), text, steps=steps
            )
        return self._record(project_id, "training", result)

    def probe(
        self,
        project_id: str,
        endpoint: str,
        model: str,
        prompts: list[str],
        authorized: bool,
        budget: int = 1024,
    ) -> dict[str, Any]:
        self._project(project_id)
        return self._record(
            project_id,
            "behavior",
            research.probe(endpoint, model, prompts, authorized, budget),
        )

    def export(self, project_id: str, dest: str) -> dict[str, Any]:
        project = self._project(project_id)
        target = Path(dest).expanduser().resolve()
        if target.exists():
            raise ValueError("Export destination already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Only this application's generated model is copied. Imported files stay external.
        files = [
            p
            for name in (
                "model",
                "quantized",
                "hf-reconstruction",
                "umrre",
                "discovery",
                "attention-research",
                "representation-research",
                "antipodal-reconstruction",
            )
            if not (project / name).is_symlink()
            for p in (project / name).rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and p.resolve().is_relative_to(project.resolve())
        ]
        from .glu32_io import export_files as glu32_export_files
        files.extend(glu32_export_files(project / "glu32-reconstruction", self._accepted_glu32(project_id)))
        findings = self.findings(project_id)
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("research.json", json.dumps(findings, indent=2))
            for file in files:
                archive.write(file, file.relative_to(project))
            archive.writestr(
                "NOTICE.txt",
                "External model paths are never copied automatically. Authorized reconstruction artifacts may contain accessible weights under the recorded redistribution declaration. Reports may contain local paths and provider outputs. Review rights and privacy before sharing.\n",
            )
        return {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "generated_model_files": len(files),
        }

    def _engine(
        self,
        operation: str,
        arguments: dict[str, Any],
        memory_limit_mb: int = 2048,
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        """Execute an optional backend outside the desktop process."""
        from .runtime_manager import engine_call

        if type(memory_limit_mb) is not int or not 128 <= memory_limit_mb <= 65536:
            raise ValueError("Memory budget must be 128–65536 MiB")
        if memory_limit_mb * 1024**2 > psutil.virtual_memory().available:
            raise MemoryError("Requested engine memory exceeds currently available RAM")
        with self._lock:
            return engine_call(
                self.root,
                operation,
                arguments,
                python=self.settings().get("runtime_python", ""),
                memory_limit_mb=memory_limit_mb,
                timeout_seconds=timeout_seconds,
            )

    def local_runtime_status(self) -> dict[str, Any]:
        """Check actual installed libraries in the configured companion interpreter."""
        return self._engine("status", {})

    def install_local_runtime(
        self, backend: str = "llm", authorized: bool = False
    ) -> dict[str, Any]:
        """Install a new dedicated optional environment after explicit consent."""
        from .runtime_manager import install_runtime

        with self._lock:
            result = install_runtime(self.root, backend, authorized)
            self.save_settings({"runtime_python": result["python"]})
            return result

    def _binding(self, project_id: str) -> dict[str, Any]:
        self._project(project_id)
        with self._db() as db:
            row = db.execute(
                "SELECT data FROM local_models WHERE project_id=?", (project_id,)
            ).fetchone()
        if row is None:
            raise ValueError(
                "Attach a compatible local model or create an owned fixture first"
            )
        value: dict[str, Any] = json.loads(row[0])
        return value

    def _save_binding(self, project_id: str, data: dict[str, Any]) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO local_models VALUES(?,?)",
                (project_id, json.dumps(data)),
            )

    def create_local_fixture(
        self, project_id: str, model_type: str = "llama"
    ) -> dict[str, Any]:
        """Build real randomly initialized GPT-2/Llama weights, without downloads."""
        destination = self._project(project_id) / "local-model"
        with self._lock:
            result = self._engine(
                "hf_fixture",
                {"destination": str(destination), "model_type": model_type},
            )
            self._save_binding(
                project_id,
                {
                    "base_path": str(destination),
                    "active_path": str(destination),
                    "adapter_path": "",
                    "adapter_backend": "hf",
                    "owned": True,
                },
            )
        return self._record(project_id, "local-build", result)

    def attach_local_model(
        self,
        project_id: str,
        source_path: str,
        authorized: bool = False,
        memory_limit_mb: int = 2048,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Validate a local, authorized SafeTensors bundle without copying its weights."""
        self._project(project_id)
        if authorized is not True:
            raise PermissionError(
                "Explicit authorization to access these model files is required"
            )
        source = str(Path(source_path).expanduser().resolve())
        with self._lock:
            result = self._engine(
                "hf_preflight",
                {"path": source, "memory_limit_mb": memory_limit_mb, "device": device},
                memory_limit_mb,
            )
            self._save_binding(
                project_id,
                {
                    "base_path": source,
                    "active_path": source,
                    "adapter_path": "",
                    "adapter_backend": "hf",
                    "owned": False,
                },
            )
        return self._record(project_id, "local-model", result)

    def local_generate(
        self,
        project_id: str,
        text: str,
        max_new_tokens: int = 32,
        memory_limit_mb: int = 2048,
        device: str = "auto",
        backend: str = "hf",
        quantize_bits: int = 0,
    ) -> dict[str, Any]:
        """Run genuine model inference; imported files require prior authorized binding."""
        binding = self._binding(project_id)
        if backend not in {"hf", "mlx"}:
            raise ValueError("Backend must be hf or mlx")
        if backend == "hf" and quantize_bits:
            raise ValueError(
                "Use the MLX backend for quantized language-model inference"
            )
        args: dict[str, Any] = {
            "prompt": text,
            "max_new_tokens": max_new_tokens,
            "memory_limit_mb": memory_limit_mb,
        }
        if backend == "mlx":
            if device not in {"cpu", "auto", "mps", "gpu"}:
                raise ValueError("MLX supports CPU or Apple GPU, not CUDA")
            if binding["adapter_path"] and device == "cpu":
                raise ValueError(
                    "MLX adapter inference currently requires Apple GPU; choose auto or mps"
                )
            args["device"] = "cpu" if device == "cpu" else "gpu"
            if (
                binding["adapter_path"]
                and binding.get("adapter_backend", "hf") != "mlx"
            ):
                raise ValueError("Merge the HF adapter before loading it with MLX")
            args.update(path=binding["active_path"], quantize_bits=quantize_bits)
            if binding["adapter_path"]:
                args["adapter_path"] = binding["adapter_path"]
                args.pop("quantize_bits", None)
                args.pop("device", None)
                operation = "mlx_generate_adapter"
            else:
                operation = "mlx_generate"
        elif binding["adapter_path"]:
            if binding.get("adapter_backend", "hf") == "mlx":
                raise ValueError("Use the MLX backend for this MLX adapter")
            args.update(
                source=binding["base_path"],
                adapter=binding["adapter_path"],
                device=device,
            )
            operation = "generate_adapter"
        else:
            args.update(path=binding["active_path"], device=device)
            operation = "hf_generate"
        result = self._engine(operation, args, memory_limit_mb)
        return self._record(project_id, "local-inference", result)

    def train_local_model(
        self,
        project_id: str,
        text: str,
        method: str = "lora",
        steps: int = 10,
        learning_rate: float = 0.01,
        max_sequence_length: int = 64,
        memory_limit_mb: int = 2048,
        device: str = "cpu",
        authorized: bool = False,
        rights: str = "",
        resume: bool = False,
    ) -> dict[str, Any]:
        """Train actual parameters with integrity-checked immutable checkpoints."""
        binding = self._binding(project_id)
        if binding["adapter_path"] and not resume:
            raise ValueError(
                "An adapter is active; merge it or use a new project before starting another training method"
            )
        sources = binding.setdefault("training_sources", {})
        source = (
            sources.get(method, binding["base_path"])
            if resume
            else binding["active_path"]
        )
        destination = self._project(project_id) / "local-training" / method
        with self._lock:
            result = self._engine(
                "train_local",
                {
                    "source": source,
                    "destination": str(destination),
                    "text": text,
                    "method": method,
                    "steps": steps,
                    "learning_rate": learning_rate,
                    "max_sequence_length": max_sequence_length,
                    "memory_limit_mb": memory_limit_mb,
                    "device": device,
                    "authorized": authorized,
                    "rights": rights,
                    "resume": resume,
                },
                memory_limit_mb,
                1800,
            )
            sources[method] = source
            if method == "full":
                binding.update(active_path=result["artifact_path"], adapter_path="")
            else:
                binding.update(
                    base_path=source,
                    active_path=source,
                    adapter_path=result["artifact_path"],
                )
            binding["adapter_backend"] = "hf"
            self._save_binding(project_id, binding)
        return self._record(project_id, "local-training", result)

    def reconstruct_local(
        self,
        project_id: str,
        prompts: list[str],
        redistributable: bool = False,
        rights: str = "",
    ) -> dict[str, Any]:
        """Copy/merge accessible weights only with explicit redistribution rights."""
        binding = self._binding(project_id)
        destination = self._project(project_id) / "hf-reconstruction"
        if binding["adapter_path"] and binding.get("adapter_backend", "hf") == "mlx":
            raise ValueError(
                "MLX adapters currently support local reload/inference; HF merge/export is unsupported"
            )
        if binding["adapter_path"]:
            operation = "merge_adapter"
            args = {
                "base": binding["base_path"],
                "adapter": binding["adapter_path"],
                "destination": str(destination),
                "redistributable": redistributable,
                "rights": rights,
            }
        else:
            operation = "hf_reconstruct"
            args = {
                "source": binding["active_path"],
                "destination": str(destination),
                "prompts": prompts,
                "redistributable": redistributable,
                "rights": rights,
            }
        if operation == "merge_adapter":
            args["prompts"] = prompts
        result = self._engine(operation, args, 2048, 600)
        binding.update(
            base_path=str(destination),
            active_path=str(destination),
            adapter_path="",
            adapter_backend="hf",
        )
        self._save_binding(project_id, binding)
        return self._record(project_id, "local-reconstruction", result)

    def mlx_benchmark(self, project_id: str) -> dict[str, Any]:
        """Measure real CPU/GPU MLX execution on the bound model."""
        binding = self._binding(project_id)
        result = self._engine(
            "mlx_benchmark", {"path": binding["active_path"], "memory_limit_mb": 2048}
        )
        result["benchmark_scope"] = (
            "base_without_active_adapter" if binding["adapter_path"] else "active_base"
        )
        return self._record(project_id, "mlx-benchmark", result)

    def benchmark_mlx_cache(
        self,
        project_id: str,
        prompt: str = "hello world",
        max_new_tokens: int = 16,
        memory_limit_mb: int = 2048,
        repeats: int = 3,
        quantize_bits: int = 0,
        precision: str = "native",
        compare_logits: bool = False,
    ) -> dict[str, Any]:
        """Measure full-prefix versus fresh KV-cache inference on the active base."""
        binding = self._binding(project_id)
        if binding["adapter_path"]:
            raise ValueError(
                "Merge or clear the active adapter before a base cache benchmark"
            )
        result = self._engine(
            "mlx_cache_benchmark",
            {
                "path": binding["active_path"],
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "memory_limit_mb": memory_limit_mb,
                "repeats": repeats,
                "quantize_bits": quantize_bits,
                "precision": precision,
                "compare_logits": compare_logits,
            },
            memory_limit_mb,
            timeout_seconds=180,
        )
        return self._record(project_id, "mlx-cache-benchmark", result)

    def run_discovery(self, project_id: str, seed: int = 17) -> dict[str, Any]:
        """Run owned hidden-teacher experiments with a separately sealed evaluator."""
        destination = self._project(project_id) / "discovery" / uuid.uuid4().hex
        started = time.monotonic()
        try:
            result = self._engine(
                "discovery",
                {
                    "destination": str(destination),
                    "seed": seed,
                    "private_storage": str(self.root / "private-research"),
                },
                memory_limit_mb=512,
                timeout_seconds=90,
            )
        except Exception as exc:
            self._record(
                project_id,
                "discovery-failed",
                {
                    "seed": seed,
                    "completed": False,
                    "destination": str(destination),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "scope": "Failed controlled research run; no positive scientific result",
                },
            )
            raise
        return self._record(project_id, "discovery", result)

    def run_attention_research(self, project_id: str) -> dict[str, Any]:
        """Study score-matrix observability and Q/K ambiguity on owned attention."""
        run_id = uuid.uuid4().hex
        destination = self._project(project_id) / "attention-research" / run_id
        started = time.monotonic()
        try:
            result = self._engine(
                "attention_research",
                {
                    "destination": str(destination),
                    "private_storage": str(self.root / "private-attention" / run_id),
                },
                memory_limit_mb=512,
                timeout_seconds=90,
            )
        except Exception as exc:
            self._record(
                project_id,
                "attention-research-failed",
                {
                    "completed": False,
                    "destination": str(destination),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                },
            )
            raise
        return self._record(project_id, "attention-research", result)

    def run_representation_research(self, project_id: str) -> dict[str, Any]:
        """Recover an owned tiny decoder output head from exposed hidden states/logits."""
        run_id = uuid.uuid4().hex
        destination = self._project(project_id) / "representation-research" / run_id
        started = time.monotonic()
        try:
            result = self._engine(
                "representation_research",
                {
                    "destination": str(destination),
                    "private_storage": str(
                        self.root / "private-representation" / run_id
                    ),
                },
                memory_limit_mb=2048,
                timeout_seconds=150,
            )
        except Exception as exc:
            self._record(
                project_id,
                "representation-research-failed",
                {
                    "completed": False,
                    "destination": str(destination),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                },
            )
            raise
        return self._record(project_id, "representation-research", result)

    def _accepted_glu32(self, project_id: str) -> dict[str, str]:
        return {str(f["data"]["artifact_id"]): str(f["data"]["manifest_sha256"]) for f in self.findings(project_id)
                if f["kind"] == "glu32-reconstruction" and f["data"].get("candidate_built") is True
                and f["data"].get("process_monitor_scope") == "process_and_children"
                and isinstance(f["data"].get("artifact_id"), str)
                and isinstance(f["data"].get("manifest_sha256"), str)}

    def reconstruct_glu32(self, project_id: str, observations_path: str) -> dict[str, Any]:
        """Build a fixed isolated bias-free FP64 SwiGLU32/64/32 candidate from authorized saved observations."""
        artifact_id = uuid.uuid4().hex
        destination = self._project(project_id) / "glu32-reconstruction" / artifact_id
        try:
            result = self._engine("glu32_reconstruct", {"observations_path": observations_path, "destination": str(destination)},
                                  memory_limit_mb=512, timeout_seconds=180)
            if result.get("process_monitor_scope") != "process_and_children":
                raise ValueError("Incomplete GLU32 process-tree monitoring")
        except Exception as exc:
            self._record(project_id, "glu32-reconstruction-failed", {"completed":False,"error_type":type(exc).__name__,"error":str(exc)[:2000]})
            raise
        return self._record(project_id, "glu32-reconstruction", {**result,"artifact_id":artifact_id})

    def predict_glu32(self, project_id: str, artifact_id: str, inputs: list[list[float]]) -> dict[str, Any]:
        """Reload a generated GLU32 candidate and predict locally; no reference-model query."""
        if len(artifact_id)!=32 or any(c not in "0123456789abcdef" for c in artifact_id):
            raise ValueError("Invalid GLU32 artifact id")
        accepted = self._accepted_glu32(project_id)
        if artifact_id not in accepted:
            raise ValueError("GLU32 candidate was not accepted by a completed bounded operation")
        directory = self._project(project_id) / "glu32-reconstruction" / artifact_id
        from .glu32_io import require_commitment
        require_commitment(directory, accepted[artifact_id])
        result = self._engine("glu32_predict", {"artifact_directory":str(directory),"inputs":inputs,"manifest_sha256":accepted[artifact_id]},memory_limit_mb=512,timeout_seconds=30)
        return self._record(project_id,"glu32-prediction",result)

    def reconstruct_antipodal(self, project_id: str, observations_path: str) -> dict[str, Any]:
        """Reconstruct an isolated GLU even component from authorized paired responses."""
        destination = self._project(project_id) / "antipodal-reconstruction" / uuid.uuid4().hex
        try:
            result = self._engine(
                "antipodal_reconstruct",
                {"observations_path": observations_path, "destination": str(destination)},
                memory_limit_mb=512,
                timeout_seconds=60,
            )
        except Exception as exc:
            self._record(project_id, "antipodal-reconstruction-failed", {
                "completed": False, "error_type": type(exc).__name__, "error": str(exc)[:2000],
            })
            raise
        return self._record(project_id, "antipodal-reconstruction", result)

    def resume_research(self, repository_path: str) -> dict[str, Any]:
        """Validate repository checkpoint/ledger and report recoverable work; execute no tasks."""
        from .research_memory import ResearchMemory

        return ResearchMemory(Path(repository_path)).resume()

    def run_umrre(self, project_id: str, seed: int = 7) -> dict[str, Any]:
        """Run bounded owned-ground-truth experiments using the bundled NumPy engine."""
        from .umrre import run_research

        destination = self._project(project_id) / "umrre" / uuid.uuid4().hex
        with self._lock:
            result = run_research(destination=destination, seed=seed)
        return self._record(project_id, "umrre", result)

    def train_mlx_model(
        self,
        project_id: str,
        text: str,
        steps: int = 5,
        learning_rate: float = 0.01,
        memory_limit_mb: int = 2048,
        authorized: bool = False,
        rights: str = "",
        quantize_bits: int = 0,
    ) -> dict[str, Any]:
        """Train real Apple GPU LoRA, retaining a fingerprinted local adapter."""
        binding = self._binding(project_id)
        if binding["adapter_path"]:
            raise ValueError(
                "An adapter is already active; use a new project or merge the HF adapter first"
            )
        destination = self._project(project_id) / "mlx-adapter"
        with self._lock:
            result = self._engine(
                "mlx_train_lora",
                dict(
                    path=binding["active_path"],
                    destination=str(destination),
                    text=text,
                    steps=steps,
                    learning_rate=learning_rate,
                    memory_limit_mb=memory_limit_mb,
                    authorized=authorized,
                    rights=rights,
                    quantize_bits=quantize_bits,
                ),
                memory_limit_mb,
                1800,
            )
            binding.update(
                base_path=binding["active_path"],
                adapter_path=str(destination),
                adapter_backend="mlx",
            )
            self._save_binding(project_id, binding)
        return self._record(project_id, "mlx-training", result)

    def trace_local_model(
        self,
        project_id: str,
        text: str,
        ablate_module: str = "",
        memory_limit_mb: int = 2048,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Measure bounded activation traces and optional local MLP interventions."""
        binding = self._binding(project_id)
        if binding.get("adapter_backend", "hf") == "mlx" and binding["adapter_path"]:
            raise ValueError("HF tracing cannot interpret an MLX adapter")
        result = self._engine(
            "trace_local",
            dict(
                path=binding["active_path"],
                prompt=text,
                ablate_module=ablate_module,
                memory_limit_mb=memory_limit_mb,
                device=device,
                adapter=binding["adapter_path"],
            ),
            memory_limit_mb,
        )
        return self._record(project_id, "activation-trace", result)

    def run_experiment(
        self,
        project_id: str,
        endpoint: str,
        model: str,
        cases: list[dict[str, Any]],
        authorized: bool = False,
        max_requests: int = 10,
        output_token_budget: int = 1024,
        resume_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a durable per-experiment request/token ledger, without dollar guarantees."""
        from .lab import run_experiment

        self._project(project_id)
        result = run_experiment(
            str(self.root),
            project_id,
            endpoint,
            model,
            cases,
            authorized=authorized,
            max_requests=max_requests,
            output_token_budget=output_token_budget,
            resume_id=resume_id,
        )
        return self._record(project_id, "experiment", result)

    def hardware(self) -> dict[str, Any]:
        mem = psutil.virtual_memory()
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory_total": mem.total,
            "memory_available": mem.available,
            "disk_free": shutil.disk_usage(self.root).free,
            "inference_backend": "NumPy CPU",
            "mlx": "optional companion engine on macOS Apple Silicon; run local_runtime_status to check installation",
            "cuda": "optional Torch device; availability checked during model preflight",
        }

    def settings(self) -> dict[str, Any]:
        with self._db() as db:
            return {
                r["key"]: json.loads(r["value"])
                for r in db.execute("SELECT * FROM settings")
            }

    def save_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "api_endpoint",
            "api_model",
            "max_workers",
            "agent_timeout_seconds",
            "agent_memory_mb",
            "runtime_python",
        }
        if set(values) - allowed:
            raise ValueError(
                "Unknown settings or secret fields; API keys belong in MODELFORGE_API_KEY"
            )
        if "runtime_python" in values:
            path = Path(values["runtime_python"]).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ValueError(
                    "Runtime Python must be an existing absolute executable path"
                )
            values["runtime_python"] = str(path)
        limits = {
            "max_workers": (1, 2),
            "agent_timeout_seconds": (1, 300),
            "agent_memory_mb": (64, 4096),
        }
        for key, (low, high) in limits.items():
            if key in values and (
                type(values[key]) is not int or not low <= values[key] <= high
            ):
                raise ValueError(f"{key} must be an integer between {low} and {high}")
        for key in ("api_endpoint", "api_model"):
            if key in values and not isinstance(values[key], str):
                raise ValueError(f"{key} must be a string")
        with self._db() as db:
            for key, value in values.items():
                db.execute(
                    "INSERT OR REPLACE INTO settings VALUES(?,?)",
                    (key, json.dumps(value)),
                )
        return self.settings()

    def tasks(self) -> list[dict[str, Any]]:
        with self._db() as db:
            return [
                {
                    **dict(r),
                    "payload": json.loads(r["payload"]),
                    "result": json.loads(r["result"]) if r["result"] else None,
                }
                for r in db.execute("SELECT * FROM tasks ORDER BY created DESC")
            ]

    def enqueue(
        self, project_id: str, kind: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self._project(project_id)
        if kind not in {
            "inspect",
            "build",
            "train",
            "infer",
            "agent",
            "create_local_fixture",
            "attach_local_model",
            "local_generate",
            "train_local_model",
            "train_mlx_model",
            "trace_local_model",
            "reconstruct_local",
            "mlx_benchmark",
            "run_umrre",
            "run_experiment",
        }:
            raise ValueError("Unsupported task kind")
        serialized = json.dumps(payload, sort_keys=True)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute(
                "SELECT id FROM tasks WHERE project_id=? AND kind=? AND payload=? AND status IN ('queued','running')",
                (project_id, kind, serialized),
            ).fetchone()
            if duplicate:
                return {"id": duplicate[0], "status": "existing"}
            task_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    project_id,
                    kind,
                    serialized,
                    "queued",
                    None,
                    time.time(),
                    None,
                ),
            )
        return {"id": task_id, "status": "queued"}

    def run_pending(self, wait: bool = True) -> dict[str, Any]:
        """Claim at most two jobs atomically across service processes."""
        futures = []
        worker_limit = int(self.settings().get("max_workers", 2))
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            running = db.execute(
                "SELECT count(*) FROM tasks WHERE status='running'"
            ).fetchone()[0]
            rows = db.execute(
                "SELECT * FROM tasks WHERE status='queued' ORDER BY created LIMIT ?",
                (max(0, worker_limit - running),),
            ).fetchall()
            for row in rows:
                db.execute(
                    "UPDATE tasks SET status='running',pid=? WHERE id=?",
                    (os.getpid(), row["id"]),
                )
        for row in rows:
            futures.append(self._pool.submit(self._run_task, dict(row)))
        if wait:
            for future in futures:
                future.result()
        return {"started": len(rows), "tasks": self.tasks()}

    def _run_task(self, row: dict[str, Any]) -> None:
        try:
            payload = json.loads(row["payload"])
            if row["kind"] == "agent":
                result = self._agent(row["project_id"], **payload)
            else:
                result = getattr(self, row["kind"])(row["project_id"], **payload)
            status = "completed"
        except Exception as exc:
            result, status = {"error": str(exc)}, "failed"
        with self._db() as db:
            db.execute(
                "UPDATE tasks SET status=?,result=?,pid=NULL WHERE id=?",
                (status, json.dumps(result), row["id"]),
            )

    def _agent(
        self, project_id: str, provider: str, prompt: str, authorized: bool = False
    ) -> dict[str, Any]:
        if not authorized:
            raise PermissionError(
                "External agent execution needs explicit authorization; provider billing applies"
            )
        if (
            provider not in {"claude", "codex"}
            or not prompt.strip()
            or len(prompt) > 16000
        ):
            raise ValueError("Use claude or codex and a bounded prompt")
        executable = shutil.which(provider)
        if not executable:
            raise RuntimeError(f"{provider} executable unavailable")
        args = (
            [executable, "-p", "--tools", "", "--max-turns", "1", prompt]
            if provider == "claude"
            else [
                executable,
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                prompt,
            ]
        )
        log = self._project(project_id) / ("agent-" + uuid.uuid4().hex + ".log")
        settings = self.settings()
        timeout = min(300, max(1, int(settings.get("agent_timeout_seconds", 120))))
        memory = (
            min(4096, max(64, int(settings.get("agent_memory_mb", 1024)))) * 1024**2
        )
        with log.open("wb") as output:
            process = subprocess.Popen(
                args,
                cwd=self._project(project_id),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            start = time.monotonic()
            try:
                while process.poll() is None:
                    rss = psutil.Process(process.pid).memory_info().rss
                    rss += sum(
                        p.memory_info().rss
                        for p in psutil.Process(process.pid).children(recursive=True)
                        if p.is_running()
                    )
                    if (
                        time.monotonic() - start > timeout
                        or rss > memory
                        or log.stat().st_size > 2_000_000
                    ):
                        raise RuntimeError("Agent exceeded time, memory or log budget")
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    try:
                        for child in psutil.Process(process.pid).children(
                            recursive=True
                        ):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                    except (psutil.Error, PermissionError):
                        pass
                    finally:
                        process.kill()
                process.wait()
        result: dict[str, Any] = {
            "provider": provider,
            "returncode": process.returncode,
            "output": log.read_text(errors="replace")[:2_000_000],
            "billing": "Provider account billed; monetary cap not supported by CLI adapter",
        }
        if process.returncode:
            raise RuntimeError(result["output"][:2000])
        return self._record(project_id, "agent", result)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
