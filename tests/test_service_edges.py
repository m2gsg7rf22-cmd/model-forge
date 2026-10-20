"""Boundary and concurrency checks using local HTTP and executable fixtures.

The executable fixtures are adapter tests, never authenticated provider runs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.error import HTTPError
import zipfile

import pytest

from modelforge.research import probe
from modelforge.service import Service
import modelforge.service as module


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Service]:
    instance = Service(tmp_path / "workspace")
    yield instance
    instance.close()


@pytest.fixture
def project(service: Service) -> str:
    return str(service.create_project("Edge cases")["id"])


def executable_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, body: str
) -> Path:
    """Install a temporary provider-named fixture, without contacting providers."""
    if sys.platform == "win32":
        pytest.skip("POSIX executable fixture; Windows provider integration unverified")
    bin_dir = tmp_path / "fixture-bin"
    bin_dir.mkdir(exist_ok=True)
    executable = bin_dir / provider
    executable.write_text(f"#!{sys.executable}\n" + body)
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    return executable


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_external_adapter_executes_fixture_and_records_output(
    service: Service,
    project: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    executable_fixture(
        tmp_path,
        monkeypatch,
        provider,
        'import json, os, sys, time\ntime.sleep(.2)\nprint(json.dumps({"fixture":True,"argv":sys.argv[1:],"cwd":os.getcwd()}))\n',
    )
    job = service.enqueue(
        project,
        "agent",
        {"provider": provider, "prompt": "inspect fixture", "authorized": True},
    )
    result = service.run_pending()["tasks"][0]
    assert result["id"] == job["id"] and result["status"] == "completed"
    output = json.loads(result["result"]["output"])
    assert output["fixture"] is True
    assert Path(output["cwd"]) == service.root / project
    if provider == "claude":
        assert output["argv"] == [
            "-p",
            "--tools",
            "",
            "--max-turns",
            "1",
            "inspect fixture",
        ]
    else:
        assert output["argv"] == [
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "inspect fixture",
        ]
    assert service.findings(project)[0]["kind"] == "agent"


def test_external_adapter_nonzero_and_missing_provider(
    service: Service, project: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="unavailable"):
        service._agent(project, "codex", "fixture", True)
    with pytest.raises(ValueError, match="bounded prompt"):
        service._agent(project, "invalid", "fixture", True)
    executable_fixture(
        tmp_path,
        monkeypatch,
        "codex",
        'import sys, time\ntime.sleep(.2)\nprint("fixture failure")\nsys.exit(3)\n',
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        service._agent(project, "codex", "fixture", True)
    assert service.findings(project) == []


@pytest.mark.parametrize("limit", ["timeout", "memory", "log"])
def test_external_adapter_resource_limits(
    service: Service,
    project: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    body = "import time\n"
    settings = {
        "agent_timeout_seconds": 1 if limit == "timeout" else 10,
        "agent_memory_mb": 64 if limit == "memory" else 1024,
    }
    if limit == "memory":
        body += "allocation = bytearray(100 * 1024**2)\n"
    if limit == "log":
        body += 'print("x" * 2_100_000, flush=True)\n'
    body += "time.sleep(20)\n"
    executable_fixture(tmp_path, monkeypatch, "codex", body)
    service.save_settings(settings)
    with pytest.raises(RuntimeError, match="exceeded"):
        service._agent(project, "codex", "fixture limits", True)
    assert service.findings(project) == []


def test_atomic_cross_service_dedup_and_claim(
    service: Service, project: str, tmp_path: Path
) -> None:
    second = Service(service.root)
    corpus = tmp_path / "config.json"
    corpus.write_text('{"model_type":"fixture"}')
    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            results = list(
                threads.map(
                    lambda current: current.enqueue(
                        project, "inspect", {"path": str(corpus)}
                    ),
                    [service, second],
                )
            )
        assert results[0]["id"] == results[1]["id"]
        assert {r["status"] for r in results} == {"queued", "existing"}
        # Three distinct jobs; a held pair ensures another service sees the global cap.
        for suffix in ["a", "b"]:
            other = tmp_path / f"{suffix}.json"
            other.write_text('{"model_type":"fixture"}')
            service.enqueue(project, "inspect", {"path": str(other)})
        release = threading.Event()
        entered = threading.Event()
        original = service.inspect

        def held_inspection(project_id: str, path: str) -> dict[str, Any]:
            entered.set()
            assert release.wait(5)
            return original(project_id, path)

        service.inspect = held_inspection  # type: ignore[method-assign]
        try:
            assert service.run_pending(wait=False)["started"] == 2
            assert entered.wait(2)
            assert second.run_pending(wait=False)["started"] == 0
            assert [t["status"] for t in service.tasks()].count("running") == 2
        finally:
            release.set()
            service._pool.shutdown(wait=True)
        assert second.run_pending()["started"] == 1
        assert all(t["status"] == "completed" for t in service.tasks())
        assert len(service.findings(project)) == 3
    finally:
        second.close()


def test_interrupted_job_recovery_preserves_live_owner(
    service: Service, project: str
) -> None:
    jobs = [service.enqueue(project, "infer", {"text": str(i)})["id"] for i in range(3)]
    with service._db() as db:
        for task, owner in zip(jobs, [None, 2_000_000_000, os.getpid()]):
            db.execute(
                "UPDATE tasks SET status='running',pid=? WHERE id=?", (owner, task)
            )
    resumed = Service(service.root)
    try:
        rows = {t["id"]: t for t in resumed.tasks()}
        assert rows[jobs[0]]["status"] == "queued" and rows[jobs[0]]["pid"] is None
        assert rows[jobs[1]]["status"] == "queued" and rows[jobs[1]]["pid"] is None
        assert (
            rows[jobs[2]]["status"] == "running" and rows[jobs[2]]["pid"] == os.getpid()
        )
    finally:
        resumed.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlink privilege unavailable on standard Windows runners",
)
def test_export_overwrite_guard_symlinks_and_hash(
    service: Service, project: str, tmp_path: Path
) -> None:
    model = service.root / project / "model"
    model.mkdir()
    (model / "owned.txt").write_text("owned model fixture")
    outside = tmp_path / "private.txt"
    outside.write_text("must not export")
    (model / "linked.txt").symlink_to(outside)
    target = tmp_path / "nested" / "archive.zip"
    exported = service.export(project, str(target))
    assert exported["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert "model/owned.txt" in archive.namelist()
        assert "model/linked.txt" not in archive.namelist()
        assert b"must not export" not in b"".join(
            archive.read(n) for n in archive.namelist()
        )
    before = target.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        service.export(project, str(target))
    assert target.read_bytes() == before


def test_settings_resource_guards_and_unknown_projects(
    service: Service, project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        service.save_settings(
            {"api_endpoint": "http://127.0.0.1:8000/v1", "max_workers": 2}
        )["max_workers"]
        == 2
    )
    with pytest.raises(ValueError, match="secret fields"):
        service.save_settings({"api_key": "fixture-not-a-secret"})
    assert "api_key" not in service.settings()
    with pytest.raises(ValueError, match="1–200"):
        service.create_project("x" * 201)
    with pytest.raises(ValueError, match="Unknown project"):
        service.export("missing", "/unused.zip")
    monkeypatch.setattr(
        module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=512 * 1024**2),
    )
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(MemoryError, match="free disk"):
        service.build(project)
    with pytest.raises(ValueError, match="authorized text"):
        service.train(project, "x", 1)


@contextmanager
def http_fixture(mode: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                {"body": payload, "authorization": self.headers.get("Authorization")}
            )
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "/stolen")
                self.end_headers()
                return
            content: Any = (
                7
                if mode == "nontext"
                else ("not json" if mode == "plain" else '{"ok":true}')
            )
            body = (
                b"x" * 2_000_001
                if mode == "large"
                else json.dumps(
                    {"choices": [{"message": {"content": content}}]}
                ).encode()
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_probe_records_plain_text_and_environment_auth(
    service: Service, project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODELFORGE_API_KEY", "local-fixture-key")
    with http_fixture("plain") as (endpoint, requests):
        result = service.probe(
            project, endpoint, "local-http-fixture", ["hello"], True, 12
        )
    assert result["results"][0]["valid_json"] is False
    assert result["results"][0]["output"] == "not json"
    assert requests[0]["authorization"] == "Bearer local-fixture-key"
    assert requests[0]["body"]["max_tokens"] == 12
    assert service.findings(project)[0]["kind"] == "behavior"
    assert "local-fixture-key" not in json.dumps(service.findings(project))


@pytest.mark.parametrize(
    ("mode", "exception", "match"),
    [
        ("nontext", ValueError, "text chat"),
        ("large", ValueError, "2MB"),
        ("redirect", HTTPError, "302"),
    ],
)
def test_probe_rejects_bad_or_redirected_http(
    mode: str, exception: type[Exception], match: str
) -> None:
    with http_fixture(mode) as (endpoint, requests):
        with pytest.raises(exception, match=match):
            probe(endpoint, "local-http-fixture", ["hello"], True, 10)
        assert len(requests) == 1


@pytest.mark.parametrize(
    ("endpoint", "model", "prompts", "budget"),
    [
        ("https://example.com?key=x", "m", ["x"], 1),
        ("https://example.com/#fragment", "m", ["x"], 1),
        ("https://example.com", "", ["x"], 1),
        ("https://example.com", "m", ["x"] * 21, 100),
        ("https://example.com", "m", ["x" * 16001], 10),
        ("https://example.com", "m", ["x"], 8193),
    ],
)
def test_probe_validates_before_network(
    endpoint: str, model: str, prompts: list[str], budget: int
) -> None:
    with pytest.raises(ValueError):
        probe(endpoint, model, prompts, True, budget)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlink privilege unavailable on standard Windows runners",
)
def test_export_does_not_follow_symlinked_model_root(
    service: Service, project: str, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (outside / "private.txt").write_text("external private data")
    (service.root / project / "model").symlink_to(outside, target_is_directory=True)
    target = tmp_path / "symlink-root.zip"
    try:
        service.export(project, str(target))
    except ValueError:
        assert not target.exists()
    else:
        with zipfile.ZipFile(target) as archive:
            assert "model/private.txt" not in archive.namelist()


def test_agent_monitor_failure_still_reaps_process(
    service: Service, project: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable_fixture(tmp_path, monkeypatch, "codex", "import time\ntime.sleep(20)\n")
    original_popen = module.subprocess.Popen
    spawned: list[Any] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("fixture process tree unavailable")

    monkeypatch.setattr(module.subprocess, "Popen", capture)
    monkeypatch.setattr(module.psutil.Process, "children", denied)
    try:
        with pytest.raises((PermissionError, RuntimeError)):
            service._agent(project, "codex", "monitor fixture", True)
        assert (
            spawned and spawned[0].poll() is not None
        ), "Monitoring error leaked the spawned process"
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_build_rejects_existing_artifacts_without_overwrite(
    service: Service, project: str
) -> None:
    existing = service.root / project / "model"
    existing.mkdir()
    marker = existing / "owned.txt"
    marker.write_text("preserve existing model")
    with pytest.raises(ValueError, match="already has a model"):
        service.build(project)
    assert marker.read_text() == "preserve existing model"
