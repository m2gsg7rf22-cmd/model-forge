import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from modelforge.service import Service
from modelforge.research import probe


def test_project_end_to_end(tmp_path):
    service = Service(tmp_path / "data")
    project = service.create_project("Test")["id"]
    assert service.hardware()["memory_available"] > 0
    service.build(project)
    report = service.inspect(project, str(tmp_path / "data" / project / "model"))
    assert report["tensors"]
    assert isinstance(service.infer(project, "hi")["output"], str)
    training = service.train(project, "hello world " * 8, 5)
    assert training
    output = service.export(project, str(tmp_path / "out.zip"))
    assert len(output["sha256"]) == 64
    assert Service(tmp_path / "data").projects()[0]["id"] == project
    assert len(service.findings(project)) == 3
    service.close()


def test_queue_persistence_dedup_failure(tmp_path):
    service = Service(tmp_path)
    project = service.create_project("queue")["id"]
    task = service.enqueue(project, "build", {})
    assert service.enqueue(project, "build", {})["id"] == task["id"]
    assert service.run_pending()["tasks"][0]["status"] == "completed"
    service.enqueue(project, "inspect", {"path": "/does/not/exist"})
    assert service.run_pending()["tasks"][0]["status"] == "failed"
    with pytest.raises(ValueError):
        service.enqueue(project, "shell", {})
    with pytest.raises(ValueError):
        service.save_settings({"api_key": "not-a-key"})
    service.save_settings({"api_model": "test"})
    assert service.settings()["api_model"] == "test"
    service.close()


def test_guards(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import modelforge.service as module

    s = Service(tmp_path)
    pid = s.create_project("guards")["id"]
    with pytest.raises(ValueError):
        s.create_project("")
    with pytest.raises(ValueError):
        s.findings("../bad")
    monkeypatch.setattr(
        module.psutil, "virtual_memory", lambda: SimpleNamespace(available=1)
    )
    with pytest.raises(MemoryError):
        s.build(pid)
    with pytest.raises(PermissionError):
        s._agent(pid, "codex", "hello")
    with pytest.raises(ValueError):
        s.train(pid, "text", 100000)
    with pytest.raises(ValueError):
        s.infer(pid, "a" * 20000)
    s.close()


def test_real_loopback_probe():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["max_tokens"] == 10
            body = json.dumps(
                {
                    "choices": [{"message": {"content": '{"ok":true}'}}],
                    "usage": {"completion_tokens": 4},
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = probe(
            f"http://127.0.0.1:{server.server_port}/v1",
            "owned-fixture",
            ["x", "x"],
            True,
            20,
        )
        assert report["repeat_consistency"]["x"]
        assert report["results"][0]["valid_json"]
        assert report["hypotheses"][0]["status"] == "not_identifiable"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_probe_guards():
    with pytest.raises(PermissionError):
        probe("https://example.com", "x", ["x"], False)
    with pytest.raises(ValueError):
        probe("http://example.com", "x", ["x"], True)
    with pytest.raises(ValueError):
        probe("https://user:pass@example.com", "x", ["x"], True)
    with pytest.raises(ValueError):
        probe("https://example.com", "x", [], True)
    with pytest.raises(ValueError):
        probe("https://example.com", "x", ["x"], True, 0)
