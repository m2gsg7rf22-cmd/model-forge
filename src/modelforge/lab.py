"""Persistent, bounded authorized API experiments with conservative resume semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from .research import probe
from .evaluation import benchmark_report, compare_reports, rank_behavioral_hypotheses


def context_cases(
    lengths: list[int], sentinel: str = "MF_SENTINEL_7C42"
) -> list[dict[str, Any]]:
    """Build exact character-length retrieval prompts, not context-window estimates."""
    if not sentinel or len(sentinel) > 100:
        raise ValueError("Sentinel must contain 1–100 characters")
    prefix = f"Remember this marker: {sentinel}\n"
    suffix = "\nReturn only the marker given at the beginning."
    cases = []
    for length in lengths:
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not len(prefix + suffix) <= length <= 16000
        ):
            raise ValueError(
                "Context lengths must fit the sentinel and be <=16000 characters"
            )
        cases.append(
            {
                "id": f"context-{length}",
                "kind": "context_retrieval",
                "prompt": prefix + "x" * (length - len(prefix + suffix)) + suffix,
                "expected": sentinel,
                "tested_characters": length,
            }
        )
    return cases


def schema_evaluation(output: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Evaluate RFC JSON against an explicitly supplied JSON Schema locally."""
    import jsonschema

    # Refuse remote references: schema evaluation must not initiate extra HTTP.
    pending: list[Any] = [schema]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if isinstance(node, (dict, list)):
            if id(node) in seen:
                continue
            seen.add(id(node))
        if isinstance(node, dict):
            for key, value in node.items():
                if (
                    key in {"$ref", "$dynamicRef", "$recursiveRef"}
                    and isinstance(value, str)
                    and value
                    and not value.startswith("#")
                ):
                    raise ValueError(
                        "Schema references must be local fragments; remote retrieval is disabled"
                    )
                pending.append(value)
        elif isinstance(node, list):
            pending.extend(node)
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    try:

        def reject(value: str) -> None:
            raise ValueError(f"Non-JSON constant: {value}")

        value = json.loads(output, parse_constant=reject)
    except (ValueError, RecursionError):
        return {
            "valid_json": False,
            "schema_compliant": False,
            "errors": ["Output is not RFC JSON"],
        }
    errors = list(validator_class(schema).iter_errors(value))
    return {
        "valid_json": True,
        "schema_compliant": not errors,
        "errors": [error.message[:500] for error in errors[:20]],
    }


def _validate(
    cases: list[dict[str, Any]], max_requests: int, output_token_budget: int
) -> None:
    if (
        isinstance(max_requests, bool)
        or not isinstance(max_requests, int)
        or not 1 <= max_requests <= 20
    ):
        raise ValueError("max_requests must be 1–20")
    if not isinstance(cases, list) or not 1 <= len(cases) <= max_requests:
        raise ValueError("Cases must fit the request limit")
    if (
        isinstance(output_token_budget, bool)
        or not isinstance(output_token_budget, int)
        or not len(cases) <= output_token_budget <= 8192
    ):
        raise ValueError("Output token budget must cover cases and be <=8192")
    identifiers: set[str] = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in identifiers
        ):
            raise ValueError("Cases need unique nonempty string IDs")
        identifiers.add(case["id"])
        if not isinstance(case.get("prompt"), str) or len(case["prompt"]) > 16000:
            raise ValueError("Prompts must be strings of at most 16000 characters")
        if case.get("kind") not in {
            "capability",
            "structured_output",
            "context_retrieval",
            "repeat",
        }:
            raise ValueError("Unknown case kind")
        if "expected" in case and not isinstance(case["expected"], str):
            raise ValueError("Expected labels must be strings")
        if (
            case["kind"] in {"capability", "context_retrieval"}
            and "expected" not in case
        ):
            raise ValueError(
                "Capability and context cases require explicit expected labels"
            )
        if case["kind"] == "context_retrieval" and case.get("tested_characters") != len(
            case["prompt"]
        ):
            raise ValueError("Context tested_characters must equal prompt length")
        if case["kind"] == "structured_output":
            if not isinstance(case.get("schema"), dict):
                raise ValueError("Structured output cases require a JSON Schema object")
            schema_evaluation("null", case["schema"])


def run_experiment(
    data_dir: str | Path,
    project_id: str,
    endpoint: str,
    model: str,
    cases: list[dict[str, Any]],
    authorized: bool = False,
    max_requests: int = 20,
    output_token_budget: int = 1024,
    resume_id: str | None = None,
    key_env: str = "MODELFORGE_API_KEY",
) -> dict[str, Any]:
    """Reserve each call durably before HTTP; never automatically retry any reservation.

    An interrupted reservation is transport-unknown and remains consumed. Resume
    reuses saved results and sends only cases that have never been reserved.
    """
    if not authorized:
        raise PermissionError("Confirm permission to query this endpoint first")
    _validate(cases, max_requests, output_token_budget)
    # Validate transport parameters before creating an experiment or issuing HTTP.
    import urllib.parse

    parsed = urllib.parse.urlsplit(endpoint)
    if (
        (
            parsed.scheme != "https"
            and not (
                parsed.scheme == "http"
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.hostname
        or not model.strip()
    ):
        raise ValueError(
            "Supply a model and HTTPS (or loopback HTTP) endpoint without credentials, query or fragment"
        )
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id must be nonempty")
    config = {
        "project_id": project_id,
        "endpoint": endpoint,
        "model": model,
        "cases": cases,
        "max_requests": max_requests,
        "output_token_budget": output_token_budget,
        "key_env": key_env,
    }
    encoded = json.dumps(config, sort_keys=True, allow_nan=False)
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiment_id = resume_id or uuid.uuid4().hex
    cap = output_token_budget // len(cases)
    with sqlite3.connect(root / "experiment-ledger.sqlite3", timeout=30) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, config TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS requests (experiment_id TEXT, case_id TEXT, status TEXT NOT NULL, token_cap INTEGER NOT NULL, result TEXT, error TEXT, PRIMARY KEY(experiment_id, case_id))"
        )
        existing = db.execute(
            "SELECT fingerprint FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
        if resume_id and existing is None:
            raise ValueError("Unknown experiment ID")
        if existing and existing[0] != fingerprint:
            raise ValueError(
                "Resume configuration differs from the original experiment"
            )
        if not existing:
            db.execute(
                "INSERT INTO experiments VALUES (?, ?, ?)",
                (experiment_id, fingerprint, encoded),
            )
        db.commit()
        for case in cases:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM requests WHERE experiment_id=? AND case_id=?",
                (experiment_id, case["id"]),
            ).fetchone()
            if row:
                db.commit()
                continue
            count, tokens = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(token_cap),0) FROM requests WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if count >= max_requests or tokens + cap > output_token_budget:
                db.rollback()
                raise ValueError("Persistent experiment budget exhausted")
            db.execute(
                "INSERT INTO requests VALUES (?, ?, 'reserved_transport_unknown', ?, NULL, NULL)",
                (experiment_id, case["id"], cap),
            )
            db.commit()
            start = time.monotonic()
            try:
                result = probe(
                    endpoint,
                    model,
                    [case["prompt"]],
                    authorized=True,
                    budget=cap,
                    key_env=key_env,
                )["results"][0]
                result.update({"case_id": case["id"], "kind": case["kind"]})
                if "expected" in case:
                    result["exact_match"] = result["output"] == case["expected"]
                if case["kind"] == "structured_output":
                    result["schema_evaluation"] = schema_evaluation(
                        result["output"], case["schema"]
                    )
                if case["kind"] == "context_retrieval":
                    result["tested_characters"] = len(case["prompt"])
                db.execute(
                    "UPDATE requests SET status='completed', result=? WHERE experiment_id=? AND case_id=?",
                    (json.dumps(result), experiment_id, case["id"]),
                )
            except Exception as exc:
                # Failure can still have incurred charges. No retry, including on resume.
                error = json.dumps(
                    {
                        "type": type(exc).__name__,
                        "latency_seconds": time.monotonic() - start,
                        "http_status": getattr(exc, "code", None),
                    }
                )
                db.execute(
                    "UPDATE requests SET status='failed_transport_or_response', error=? WHERE experiment_id=? AND case_id=?",
                    (error, experiment_id, case["id"]),
                )
            db.commit()
        rows = db.execute(
            "SELECT case_id, status, token_cap, result, error FROM requests WHERE experiment_id=? ORDER BY rowid",
            (experiment_id,),
        ).fetchall()
    results = [json.loads(row[3]) for row in rows if row[1] == "completed"]
    report: dict[str, Any] = {
        "kind": "authorized_api_experiment",
        "experiment_id": experiment_id,
        "project_id": project_id,
        "endpoint": endpoint,
        "model": model,
        "results": results,
        "ledger": [
            {
                "case_id": row[0],
                "status": row[1],
                "requested_output_token_cap": row[2],
                "error": json.loads(row[4]) if row[4] else None,
            }
            for row in rows
        ],
        "reserved_requests": len(rows),
        "max_requests": max_requests,
        "reserved_output_token_cap": sum(row[2] for row in rows),
        "requested_output_token_cap": output_token_budget,
        "context_observations": [
            r for r in results if r["kind"] == "context_retrieval"
        ],
        "limitations": [
            "Request and requested output-token caps are enforced locally; provider billing and provider compliance are not guaranteed. Input tokens, overhead and failed calls may be billed.",
            "Interrupted reservations are transport-unknown and consumed; resuming never retries them.",
            "Context observations describe tested prompt characters and provider-reported usage only, never a true maximum window or tokenizer count.",
            "Schema scores are local validation of observed outputs; requests do not require provider-native JSON Schema support.",
            "No weights, architecture or broad capability reconstruction is established.",
        ],
    }
    report["behavioral_hypothesis_ranking"] = rank_behavioral_hypotheses(report)
    # Explicit cases may label identical prompts differently; score per case above,
    # and use existing benchmark only for unambiguous prompt labels.
    labels: dict[str, set[str]] = {}
    for case in cases:
        if "expected" in case:
            labels.setdefault(case["prompt"], set()).add(case["expected"])
    report["benchmark"] = benchmark_report(
        report,
        {
            prompt: next(iter(values))
            for prompt, values in labels.items()
            if len(values) == 1
        },
    )
    return report


def compare_experiments(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare exact matched prompts using existing descriptive evaluation."""
    return compare_reports(left, right)
