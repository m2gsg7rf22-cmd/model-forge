"""Bounded authorized OpenAI-compatible behavioral experiments."""

from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.parse
from typing import Any


def probe(
    endpoint: str,
    model: str,
    prompts: list[str],
    authorized: bool,
    budget: int = 1024,
    key_env: str = "MODELFORGE_API_KEY",
) -> dict[str, Any]:
    """Measure observed outputs without claiming architectural identification.

    Budget is a hard requested output-token cap, not a provider dollar cap.
    No outputs are used as training data. Redirects are refused to protect keys.
    """
    if not authorized:
        raise PermissionError("Confirm permission to query this endpoint first")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Use HTTPS or a loopback HTTP test endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, query or fragment")
    if (
        not model.strip()
        or not 1 <= len(prompts) <= 20
        or not all(isinstance(p, str) and len(p) <= 16000 for p in prompts)
    ):
        raise ValueError("Supply a model and 1–20 prompts of at most 16000 characters")
    if not len(prompts) <= budget <= 8192:
        raise ValueError("Output token budget must cover each prompt and be <=8192")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
        ) -> None:
            return None

    opener = urllib.request.build_opener(NoRedirect)
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": budget // len(prompts),
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get(key_env):
            headers["Authorization"] = "Bearer " + os.environ[key_env]
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            json.dumps(payload).encode(),
            headers,
        )
        start = time.monotonic()
        with opener.open(req, timeout=45) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("Response exceeded 2MB safety bound")
        obj = json.loads(data)
        output = obj["choices"][0]["message"]["content"]
        if not isinstance(output, str):
            raise ValueError("Only text chat completions are supported")
        try:
            json.loads(output)
            valid_json = True
        except ValueError:
            valid_json = False
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "latency_seconds": time.monotonic() - start,
                "valid_json": valid_json,
                "usage": obj.get("usage", {}),
            }
        )
    groups: dict[str, list[str]] = {}
    for result in results:
        groups.setdefault(result["prompt"], []).append(result["output"])
    consistency = {p: len(set(out)) == 1 for p, out in groups.items() if len(out) > 1}
    from .evaluation import rank_behavioral_hypotheses

    return {
        "behavioral_hypothesis_ranking": rank_behavioral_hypotheses(
            {"results": results}
        ),
        "kind": "behavioral_observation",
        "model": model,
        "endpoint": endpoint,
        "results": results,
        "repeat_consistency": consistency,
        "requested_output_token_cap": budget,
        "hypotheses": [
            {
                "claim": "Internal architecture identifiable from these probes",
                "status": "not_identifiable",
                "confidence": None,
            }
        ],
        "limitations": [
            "Prompt and provider overhead tokens cost extra; this is not a monetary budget.",
            "Latency is end-to-end; it does not identify parameter count, weights or context limit.",
            "JSON validity is measured, not a guarantee of schema compliance.",
            "No behavioral or weight reconstruction is established.",
        ],
    }
