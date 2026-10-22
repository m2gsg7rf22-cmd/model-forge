"""Build pipeline for private ModelForge chat models (data, LoRA, fuse, card).

Training data comes only from (a) openly licensed records supplied by the user
(e.g. CC0 Wikidata labels) and (b) rejection sampling of the base model's own
answers on the procedural *train* split, kept only when a deterministic checker
verifies them. Evaluation items (validation/test seeds, Hebrew evaluation
capitals) are excluded. No hosted-provider outputs are used.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import capability_eval as ce

EVAL_COUNTRIES = {country for country, _ in ce.HE_CAPITALS}


def wikidata_capital_examples(path: str | Path, seed: int = 0) -> list[dict[str, Any]]:
    """Hebrew capital/continent QA from a Wikidata SPARQL JSON export (CC0)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["results"]["bindings"]
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        country, capital = row["cHe"]["value"], row["capHe"]["value"]
        if country in EVAL_COUNTRIES or not capital or capital == country:
            continue
        entry = pairs.setdefault(country, {"capitals": set(), "continents": set()})
        entry["capitals"].add(capital)
        if "contHe" in row:
            entry["continents"].add(row["contHe"]["value"])
    rng, out = random.Random(seed), []
    for country, info in sorted(pairs.items()):
        if len(info["capitals"]) != 1:
            continue  # ambiguous (multiple capitals) records are skipped
        capital = next(iter(info["capitals"]))
        q = rng.choice([f"מהי עיר הבירה של {country}?", f"מה בירת {country}?", f"איזו עיר היא הבירה של {country}?"])
        out.append(_chat(q, f"עיר הבירה של {country} היא {capital}."))
        out.append(_chat(f"{capital} היא עיר הבירה של איזו מדינה? ענה בשם המדינה בלבד.", country))
        if len(info["continents"]) == 1:
            continent = next(iter(info["continents"]))
            out.append(_chat(f"באיזו יבשת נמצאת {country}?", f"{country} נמצאת ב{continent}."))
    return out


def _chat(user: str, assistant: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def rejection_sample(generate: Callable[[ce.Task], str], per_category: int,
                     categories: tuple[str, ...] = ce.CATEGORIES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the model's own verified answers on the procedural train split."""
    kept: list[dict[str, Any]] = []
    stats: dict[str, list[int]] = {}
    for task in ce.build_tasks("train", per_category, categories):
        if task.category == "hebrew" and task.reference in {a[0] for _, a in ce.HE_CAPITALS}:
            continue  # evaluation capitals never enter training
        reply = generate(task)
        ok = bool(task.check(reply))
        stats.setdefault(task.category, [0, 0])
        stats[task.category][0] += ok
        stats[task.category][1] += 1
        if ok:
            example: dict[str, Any] = {"messages": task.messages + [{"role": "assistant", "content": reply}]}
            if task.tools:
                example["tools"] = task.tools
            kept.append(example)
    return kept, {c: {"kept": k, "sampled": n} for c, (k, n) in stats.items()}


def write_split(examples: list[dict[str, Any]], directory: str | Path, valid_fraction: float = 0.08, seed: int = 0) -> dict[str, int]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    items = list(examples)
    random.Random(seed).shuffle(items)
    n_valid = max(4, int(len(items) * valid_fraction))
    splits = {"valid": items[:n_valid], "train": items[n_valid:]}
    for name, rows in splits.items():
        with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {name: len(rows) for name, rows in splits.items()}


def train_lora(model: str | Path, data: str | Path, adapter: str | Path, *, iters: int = 300, batch_size: int = 4,
               learning_rate: float = 1e-5, num_layers: int = 16, max_seq_length: int = 1024,
               timeout_seconds: int = 3600) -> dict[str, Any]:
    """Run mlx-lm (Q)LoRA training in a child process; returns parsed losses."""
    adapter_path = Path(adapter)
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise ValueError("Adapter destination must be new")
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", str(model), "--train", "--data", str(data),
           "--adapter-path", str(adapter_path), "--iters", str(iters), "--batch-size", str(batch_size),
           "--learning-rate", str(learning_rate), "--num-layers", str(num_layers),
           "--max-seq-length", str(max_seq_length), "--mask-prompt", "--steps-per-report", "10",
           "--steps-per-eval", str(max(50, iters // 4)), "--save-every", str(iters), "--grad-checkpoint"]
    start = time.perf_counter()
    adapter_path.mkdir(parents=True)
    log_path = adapter_path / "train.log"
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=timeout_seconds,
                              env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    log = log_path.read_text(encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"LoRA training failed:\n{log[-3000:]}")
    train = [float(x.split("Train loss")[1].split(",")[0]) for x in log.splitlines() if "Train loss" in x]
    val = [float(x.split("Val loss")[1].split(",")[0]) for x in log.splitlines() if "Val loss" in x]
    peak = [float(x.split("Peak mem")[1].split("GB")[0]) for x in log.splitlines() if "Peak mem" in x]
    return {"seconds": time.perf_counter() - start, "train_loss": train, "val_loss": val,
            "peak_memory_gb": max(peak) if peak else None, "iters": iters, "learning_rate": learning_rate,
            "num_layers": num_layers, "batch_size": batch_size}


def fuse(model: str | Path, adapter: str | Path, destination: str | Path, card: dict[str, Any]) -> Path:
    """Fuse a LoRA adapter into a standalone private bundle with a model card."""
    target = Path(destination)
    if target.exists():
        raise ValueError("Fused model destination must be new")
    cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", str(model), "--adapter-path", str(adapter),
           "--save-path", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    if proc.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"Fuse failed:\n{(proc.stdout + proc.stderr)[-3000:]}")
    from .private_runtime import bundle_digest

    card = {**card, "bundle_sha256": bundle_digest(target), "created_unix": time.time()}
    (target / "modelforge.json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    return target
