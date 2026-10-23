"""Autonomous private-model improvement engine: evaluate, compare, keep or revert.

Candidates are (model bundle, optional adapter, inference strategy). Every run is
appended to a private experiment ledger and summarized in a leaderboard under
MODELFORGE_PRIVATE_HOME. Decisions use paired item-level sign tests on the
validation split against a fixed baseline run; the test split is evaluated only
for a retained candidate and never used for selection.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import eval_suite as es
from .capability_eval import _last_number
from .private_runtime import PrivateChatModel, private_home


def home() -> Path:
    root = private_home() / "improvement"
    (root / "runs").mkdir(parents=True, exist_ok=True)
    return root


def _append(event: dict[str, Any]) -> None:
    with (home() / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**event, "time": time.time()}, ensure_ascii=False) + "\n")


def make_chat(model: PrivateChatModel, strategy: str, system: str | None = None, budget: int = 0) -> es.ChatFn:
    """greedy | sc<k> (self-consistency: k samples, majority vote on the final number)."""

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        max_tokens = max(max_tokens, budget)
        greedy = str(model.chat(msgs, max_tokens=max_tokens, tools=tools)["reply"])
        if strategy == "greedy" or tools or "Answer: <" not in str(messages[-1].get("content", "")):
            return greedy
        k = int(strategy.removeprefix("sc"))
        replies = [greedy] + [str(model.chat(msgs, max_tokens=max_tokens, temperature=0.7, top_p=0.95)["reply"]) for _ in range(k - 1)]
        votes = collections.Counter(v for r in replies if (v := _last_number(r)) is not None)
        if not votes:
            return greedy
        best = votes.most_common(1)[0][0]
        return next(r for r in replies if _last_number(r) == best)

    if strategy != "greedy" and not (strategy.startswith("sc") and strategy[2:].isdigit()):
        raise ValueError("strategy must be greedy or sc<k>")
    return chat


def run(name: str, model_path: str, *, adapter: str | None = None, strategy: str = "greedy", system: str | None = None,
        split: str = "validation", per_category: int = 30, categories: tuple[str, ...] = es.CATEGORIES,
        notes: str = "", budget: int = 0, start_index: int = 0) -> dict[str, Any]:
    out = home() / "runs" / f"{name}--{split}.json"
    if out.exists():
        raise ValueError(f"Run {out.name} exists; experiments are immutable, choose a new name")
    model = PrivateChatModel(model_path, adapter_path=adapter)
    model.load()
    chat = make_chat(model, strategy, system, budget)
    started = time.perf_counter()
    speed = model.chat([{"role": "user", "content": "Count from 1 to 40 separated by spaces."}], max_tokens=128)
    report = es.evaluate(chat, split, per_category, categories, log=lambda s: print(s, flush=True), start_index=start_index)
    report.update({"name": name, "model": model_path, "adapter": adapter, "strategy": strategy, "system": system,
                   "notes": notes, "budget": budget, "start_index": start_index, "generation_tps": speed.get("generation_tps"), "peak_memory_gb": model.status()["peak_memory_bytes"] / 1e9,
                   "wall_seconds": time.perf_counter() - started})
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    _append({"kind": "run", "name": name, "split": split, "mean": report["mean"], "pooled": report["pooled"],
             "scores": report["scores"], "strategy": strategy, "adapter": adapter, "model": model_path, "notes": notes})
    model.unload()
    return report


def load_run(name: str, split: str = "validation") -> dict[str, Any]:
    """Load a run; ``a+b`` merges disjoint runs (e.g. a validation extension) by rows."""
    parts = [json.loads((home() / "runs" / f"{n}--{split}.json").read_text(encoding="utf-8")) for n in name.split("+")]
    rows = [r for part in parts for r in part["rows"]]
    if len({(r["category"], r["id"]) for r in rows}) != len(rows):
        raise ValueError("Merged runs overlap")
    data: dict[str, Any] = {**parts[0], "rows": rows, "name": name}
    return data


def decide(baseline: str, candidate: str, split: str = "validation") -> dict[str, Any]:
    """Keep only a significant paired improvement without category regressions."""
    cmp = es.compare(load_run(baseline, split), load_run(candidate, split))
    overall = cmp["__all__"]
    keep = bool(overall["significant_improvement"] and not overall["regressed_categories"])
    decision = {"kind": "decision", "baseline": baseline, "candidate": candidate, "split": split, "keep": keep,
                "wins": overall["wins"], "losses": overall["losses"], "p_value": overall["p_value"],
                "regressed_categories": overall["regressed_categories"],
                "per_category": {c: v for c, v in cmp.items() if c != "__all__"}}
    _append(decision)
    return decision


def leaderboard(split: str = "validation") -> list[dict[str, Any]]:
    rows = []
    for path in sorted((home() / "runs").glob(f"*--{split}.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"name": r["name"], "mean": round(r["mean"], 4), "pooled_ci95": [round(x, 3) for x in r["pooled"]["ci95"]],
                     "n": r["pooled"]["n"], "strategy": r["strategy"], "adapter": r["adapter"],
                     "tps": round(r.get("generation_tps") or 0, 1), "peak_gb": round(r.get("peak_memory_gb") or 0, 2),
                     "eval_minutes": round(r["seconds"] / 60, 1),
                     "scores": {c: round(v, 3) for c, v in r["scores"].items()}})
    rows.sort(key=lambda r: -r["mean"])
    (home() / f"leaderboard-{split}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def weakest(name: str, split: str = "validation") -> list[tuple[str, float]]:
    """Categories ordered by the upper 95% bound: those certainly below target first."""
    r = load_run(name, split)
    return sorted(((c, v["ci95"][1]) for c, v in r["categories"].items()), key=lambda t: t[1])


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m modelforge.improvement")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("model")
    p.add_argument("--adapter")
    p.add_argument("--strategy", default="greedy")
    p.add_argument("--system")
    p.add_argument("--split", default="validation")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--categories")
    p.add_argument("--notes", default="")
    p.add_argument("--budget", type=int, default=0, help="minimum max_tokens (reasoning models)")
    p.add_argument("--start", type=int, default=0, help="first item index per category (validation extension)")
    p = sub.add_parser("decide")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p = sub.add_parser("leaderboard")
    p.add_argument("--split", default="validation")
    p = sub.add_parser("weakest")
    p.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "run":
        cats = tuple(a.categories.split(",")) if a.categories else es.CATEGORIES
        r = run(a.name, a.model, adapter=a.adapter, strategy=a.strategy, system=a.system, split=a.split,
                per_category=a.n, categories=cats, notes=a.notes, budget=a.budget, start_index=a.start)
        print("RESULT", json.dumps({"mean": round(r["mean"], 4), "pooled": r["pooled"], "scores": {c: round(v, 3) for c, v in r["scores"].items()}}))
    elif a.cmd == "decide":
        print(json.dumps(decide(a.baseline, a.candidate), indent=1))
    elif a.cmd == "leaderboard":
        json.dump(leaderboard(a.split), sys.stdout, indent=1)
    else:
        print(json.dumps(weakest(a.name), indent=1))


if __name__ == "__main__":
    main()
