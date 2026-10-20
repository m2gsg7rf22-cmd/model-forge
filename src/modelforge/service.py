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
        return self._record(project_id, "inspection", inspect_model(path))

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
        model_root = project / "model"
        files = (
            []
            if model_root.is_symlink()
            else [
                p
                for p in model_root.rglob("*")
                if p.is_file()
                and not p.is_symlink()
                and p.resolve().is_relative_to(project.resolve())
            ]
        )
        findings = self.findings(project_id)
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("research.json", json.dumps(findings, indent=2))
            for file in files:
                archive.write(file, file.relative_to(project))
            archive.writestr(
                "NOTICE.txt",
                "Imported model files are not redistributed. Reports may contain local paths and provider outputs. Review before sharing.\n",
            )
        return {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "generated_model_files": len(files),
        }

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
            "mlx": "not implemented",
            "cuda": "not implemented",
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
        }
        if set(values) - allowed:
            raise ValueError(
                "Unknown settings or secret fields; API keys belong in MODELFORGE_API_KEY"
            )
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
        if kind not in {"inspect", "build", "train", "infer", "agent"}:
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
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            running = db.execute(
                "SELECT count(*) FROM tasks WHERE status='running'"
            ).fetchone()[0]
            rows = db.execute(
                "SELECT * FROM tasks WHERE status='queued' ORDER BY created LIMIT ?",
                (max(0, 2 - running),),
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
