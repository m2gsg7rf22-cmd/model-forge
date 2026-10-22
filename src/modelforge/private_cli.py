"""CLI for private local models (no hosted inference)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import capability_eval as ce
from . import private_runtime as pr


def _model(args: Any) -> str:
    if args.model:
        return str(args.model)
    models = pr.list_private_models()
    if not models:
        raise SystemExit(f"No private models in {pr.private_home() / 'models'}")
    return str(models[0]["name"])


def run_private(args: Any) -> None:
    if args.action == "list":
        print(json.dumps(pr.list_private_models(), indent=2, ensure_ascii=False))
        return
    if args.action == "leaderboard":
        from .improvement import leaderboard

        print(json.dumps(leaderboard(args.split), indent=2))
        return
    if args.action == "experiments":
        from .improvement import home

        ledger = home() / "ledger.jsonl"
        for line in ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []:
            event = json.loads(line)
            print(json.dumps({k: event.get(k) for k in ("kind", "name", "candidate", "mean", "keep", "p_value", "notes") if k in event}))
        return
    if args.action == "serve":
        pr.MANAGER.load(_model(args))
        print(f"Serving private model on http://127.0.0.1:{args.port}/v1/chat/completions", flush=True)
        pr.serve_api(port=args.port)
        return
    status = pr.MANAGER.load(_model(args))
    print(f"[loaded {Path(status['path']).name} in {status['load_seconds']:.1f}s]", file=sys.stderr)
    if args.action == "eval":
        model = pr.MANAGER.current
        assert model is not None
        report = ce.evaluate(
            lambda t: model.chat(t.messages, max_tokens=t.max_tokens, tools=t.tools)["reply"],
            args.split, args.per_category,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"scores": report["scores"], "mean": report["mean"], "meets_target": report["meets_target"]}, indent=1))
        return
    settings = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    history: list[dict[str, Any]] = []
    prompts = [args.prompt] if args.prompt else None
    while True:
        if prompts is not None:
            if not prompts:
                return
            text = prompts.pop()
        else:
            try:
                text = input("you> ").strip()
            except EOFError:
                return
            if text in {"/exit", "/quit"}:
                return
            if text == "/reset":
                history.clear()
                continue
            if not text:
                continue
        history.append({"role": "user", "content": text})
        result = pr.MANAGER.chat(history, **settings)
        history.append({"role": "assistant", "content": result["reply"]})
        print(result["reply"], flush=True)
        print(f"[{result.get('generation_tokens')} tok · {result.get('generation_tps', 0):.1f} tok/s · "
              f"peak {result.get('peak_memory_gb', 0):.2f} GB]", file=sys.stderr)
