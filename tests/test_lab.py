"""Local HTTP fixture tests; no commercial provider is queried."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from modelforge.lab import context_cases, run_experiment, schema_evaluation


@pytest.fixture
def provider() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            prompt = payload["messages"][0]["content"]
            if prompt == "fail":
                self.send_error(503)
                return
            output = '{"answer": 42}' if prompt == "schema" else "OK"
            body = json.dumps(
                {
                    "choices": [{"message": {"content": output}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def cases() -> list[dict[str, Any]]:
    return [
        {"id": "a", "kind": "capability", "prompt": "hello", "expected": "OK"},
        {
            "id": "b",
            "kind": "structured_output",
            "prompt": "schema",
            "schema": {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
    ]


def test_http_schema_budget_and_resume(
    tmp_path: Path, provider: tuple[str, list[dict[str, Any]]]
) -> None:
    endpoint, calls = provider
    report = run_experiment(tmp_path, "p", endpoint, "fixture", cases(), True, 2, 15)
    assert len(calls) == 2
    assert sum(call["max_tokens"] for call in calls) == 14
    assert report["results"][1]["schema_evaluation"]["schema_compliant"]
    resumed = run_experiment(
        tmp_path,
        "p",
        endpoint,
        "fixture",
        cases(),
        True,
        2,
        15,
        report["experiment_id"],
    )
    assert resumed["results"] == report["results"]
    assert len(calls) == 2
    with pytest.raises(ValueError, match="configuration"):
        run_experiment(
            tmp_path,
            "p",
            endpoint,
            "changed-version",
            cases(),
            True,
            2,
            15,
            report["experiment_id"],
        )


def test_failed_calls_not_retried(
    tmp_path: Path, provider: tuple[str, list[dict[str, Any]]]
) -> None:
    endpoint, calls = provider
    rows = [{"id": "fail", "kind": "repeat", "prompt": "fail"}]
    report = run_experiment(tmp_path, "p", endpoint, "fixture", rows, True, 1, 8)
    assert report["ledger"][0]["status"] == "failed_transport_or_response"
    run_experiment(
        tmp_path, "p", endpoint, "fixture", rows, True, 1, 8, report["experiment_id"]
    )
    assert len(calls) == 1
    assert report["reserved_output_token_cap"] == 8


def test_interruption_reserves_before_http(
    tmp_path: Path, provider: tuple[str, list[dict[str, Any]]]
) -> None:
    import sqlite3
    from modelforge.research import probe

    endpoint, calls = provider
    count = 0

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt
        return probe(*args, **kwargs)

    rows = cases() + [{"id": "c", "kind": "repeat", "prompt": "third"}]
    with patch("modelforge.lab.probe", interrupt), pytest.raises(KeyboardInterrupt):
        run_experiment(tmp_path, "p", endpoint, "fixture", rows, True, 3, 18)
    with sqlite3.connect(tmp_path / "experiment-ledger.sqlite3") as db:
        experiment_id = db.execute("SELECT id FROM experiments").fetchone()[0]
    report = run_experiment(
        tmp_path, "p", endpoint, "fixture", rows, True, 3, 18, experiment_id
    )
    assert len(calls) == 2
    assert len(report["results"]) == 2
    assert report["ledger"][1]["status"] == "reserved_transport_unknown"


@pytest.mark.parametrize(
    "kwargs", [{"max_requests": 1}, {"output_token_budget": 1}, {"authorized": False}]
)
def test_limits_before_network(
    tmp_path: Path, provider: tuple[str, list[dict[str, Any]]], kwargs: dict[str, Any]
) -> None:
    endpoint, calls = provider
    config: dict[str, Any] = {"authorized": True}
    config.update(kwargs)
    with pytest.raises((ValueError, PermissionError)):
        run_experiment(tmp_path, "p", endpoint, "fixture", cases(), **config)
    assert calls == []


def test_schema_failure_and_context_lengths() -> None:
    schema = {"type": "integer"}
    assert not schema_evaluation('"42"', schema)["schema_compliant"]
    assert not schema_evaluation("NaN", schema)["valid_json"]
    rows = context_cases([128, 1024, 16000])
    assert [len(row["prompt"]) for row in rows] == [128, 1024, 16000]
    with pytest.raises(ValueError):
        context_cases([16001])


def test_remote_schema_refs_rejected_before_transport(
    tmp_path: Path, provider: tuple[str, list[dict[str, Any]]]
) -> None:
    endpoint, calls = provider
    rows = [
        {
            "id": "remote",
            "kind": "structured_output",
            "prompt": "schema",
            "schema": {"$ref": "https://example.com/schema"},
        }
    ]
    with pytest.raises(ValueError, match="remote retrieval"):
        run_experiment(tmp_path, "p", endpoint, "fixture", rows, True)
    assert calls == []
