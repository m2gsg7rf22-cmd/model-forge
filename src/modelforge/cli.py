"""Command-line entry points for ModelForge."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
from .service import Service


def main() -> None:
    parser = argparse.ArgumentParser(prog="modelforge")
    parser.add_argument("--home", help="Research data directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gui")
    sub.add_parser("hardware")
    sub.add_parser("projects")
    sub.add_parser("tasks")
    sub.add_parser("run-tasks")
    p = sub.add_parser("create")
    p.add_argument("name")
    p = sub.add_parser("inspect")
    p.add_argument("project_id")
    p.add_argument("path")
    p = sub.add_parser("build")
    p.add_argument("project_id")
    p = sub.add_parser("reconstruct")
    p.add_argument("project_id")
    p.add_argument("source_path")
    p = sub.add_parser("quantize")
    p.add_argument("project_id")
    p = sub.add_parser("compare-experiments")
    p.add_argument("project_id")
    p.add_argument("left_id")
    p.add_argument("right_id")
    p = sub.add_parser("benchmark")
    p.add_argument("project_id")
    p.add_argument("finding_id")
    p.add_argument("expected", help="JSON mapping of prompt to exact expected output")
    p = sub.add_parser("infer")
    p.add_argument("project_id")
    p.add_argument("text")
    p = sub.add_parser("train")
    p.add_argument("project_id")
    p.add_argument("text")
    p.add_argument("--steps", type=int, default=100)
    p = sub.add_parser("findings")
    p.add_argument("project_id")
    p = sub.add_parser("export")
    p.add_argument("project_id")
    p.add_argument("destination")
    p = sub.add_parser("enqueue")
    p.add_argument("project_id")
    p.add_argument("kind")
    p.add_argument("payload", help="JSON object")
    p = sub.add_parser("example")
    p.add_argument("--destination", default="example-export.zip")
    p = sub.add_parser("engine", help="Invoke a typed local research service operation")
    p.add_argument(
        "operation",
        choices=[
            "local_runtime_status",
            "install_local_runtime",
            "create_local_fixture",
            "attach_local_model",
            "local_generate",
            "train_local_model",
            "train_mlx_model",
            "reconstruct_local",
            "mlx_benchmark",
            "run_umrre",
            "run_discovery",
            "run_attention_research",
            "run_representation_research",
            "resume_research",
            "benchmark_mlx_cache",
            "trace_local_model",
            "run_experiment",
        ],
    )
    p.add_argument(
        "arguments",
        nargs="?",
        default="{}",
        help="JSON object matching the operation API",
    )
    p = sub.add_parser(
        "local-example", help="Real offline Llama creation, LoRA, merge and export"
    )
    p.add_argument("--destination", default="local-model-example.zip")
    p = sub.add_parser("private", help="Private local model: list, chat, serve, eval")
    p.add_argument("action", choices=["list", "chat", "serve", "eval"])
    p.add_argument("--model", help="Private model name or bundle path")
    p.add_argument("--prompt", help="One-shot prompt; omit for an interactive chat")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--per-category", type=int, default=8)
    p.add_argument("--output", help="Evaluation report path")
    args = parser.parse_args()
    if args.home:
        os.environ["MODELFORGE_HOME"] = str(Path(args.home).resolve())
    if args.command == "gui":
        from .gui import launch

        launch()
        return
    if args.command == "private":
        from .private_cli import run_private

        run_private(args)
        return
    service = Service(args.home)
    result: Any
    try:
        if args.command == "engine":
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Engine arguments must be a JSON object")
            result = getattr(service, args.operation)(**arguments)
        elif args.command == "local-example":
            project = service.create_project("Offline Llama adaptation example")
            pid = project["id"]
            built = service.create_local_fixture(pid, "llama")
            before = service.local_generate(pid, "hello world", max_new_tokens=4)
            trained = service.train_local_model(
                pid,
                "hello world model forge " * 12,
                steps=4,
                authorized=True,
                rights="Project-owned synthetic fixture and user-created example text",
            )
            after = service.local_generate(pid, "hello world", max_new_tokens=4)
            reconstructed = service.reconstruct_local(
                pid,
                ["hello world"],
                redistributable=True,
                rights="Project-owned randomly initialized fixture and adaptation",
            )
            exported = service.export(pid, args.destination)
            result = dict(
                project=project,
                build=built,
                before=before,
                training=trained,
                after=after,
                reconstruction=reconstructed,
                export=exported,
            )
        elif args.command in {"hardware", "projects", "tasks"}:
            result = getattr(service, args.command)()
        elif args.command == "run-tasks":
            result = service.run_pending()
        elif args.command == "create":
            result = service.create_project(args.name)
        elif args.command == "inspect":
            result = service.inspect(args.project_id, args.path)
        elif args.command == "build":
            result = service.build(args.project_id)
        elif args.command == "reconstruct":
            result = service.reconstruct(args.project_id, args.source_path)
        elif args.command == "quantize":
            result = service.quantize(args.project_id)
        elif args.command == "compare-experiments":
            result = service.compare_experiments(
                args.project_id, args.left_id, args.right_id
            )
        elif args.command == "benchmark":
            result = service.benchmark(
                args.project_id, args.finding_id, json.loads(args.expected)
            )
        elif args.command == "infer":
            result = service.infer(args.project_id, args.text)
        elif args.command == "train":
            result = service.train(args.project_id, args.text, args.steps)
        elif args.command == "findings":
            result = service.findings(args.project_id)
        elif args.command == "export":
            result = service.export(args.project_id, args.destination)
        elif args.command == "enqueue":
            result = service.enqueue(
                args.project_id, args.kind, json.loads(args.payload)
            )
        else:
            from .builder import reconstruct_model

            project = service.create_project("ModelForge complete local example")
            pid = project["id"]
            built = service.build(pid)
            inspection = service.inspect(pid, str(service.root / pid / "model"))
            inference = service.infer(pid, "Hello")
            training = service.train(pid, "hello world " * 8, 20)
            model_dir = service.root / pid / "model"
            comparison = reconstruct_model(
                str(model_dir), str(service.root / pid / "reconstructed")
            )
            exported = service.export(pid, args.destination)
            result = {
                "project": project,
                "build": built,
                "inspection": inspection,
                "inference": inference,
                "training": training,
                "reconstructed_fixture_parity": comparison,
                "export": exported,
            }
        print(json.dumps(result, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
