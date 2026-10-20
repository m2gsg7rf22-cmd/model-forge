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
    args = parser.parse_args()
    if args.home:
        os.environ["MODELFORGE_HOME"] = str(Path(args.home).resolve())
    if args.command == "gui":
        from .gui import launch

        launch()
        return
    service = Service(args.home)
    result: Any
    try:
        if args.command in {"hardware", "projects", "tasks"}:
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
