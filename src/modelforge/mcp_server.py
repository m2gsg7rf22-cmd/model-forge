"""Official MCP SDK stdio interface to the real shared service."""

from __future__ import annotations
import os
from typing import Any
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from .service import Service


def create_server(data_dir: str | None = None) -> FastMCP:
    service = Service(data_dir)
    server = FastMCP("ModelForge")

    @server.tool()
    def create_project(name: str) -> dict[str, Any]:
        """Create a persistent model research project."""
        return service.create_project(name)

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, openWorldHint=False
        )
    )
    def list_projects() -> list[dict[str, Any]]:
        """List saved projects."""
        return service.projects()

    @server.tool()
    def inspect_model(project_id: str, path: str) -> dict[str, Any]:
        """Inspect authorized local model metadata without executing model code."""
        return service.inspect(project_id, path)

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, openWorldHint=False
        )
    )
    def read_findings(project_id: str) -> list[dict[str, Any]]:
        """Read evidence, architectural reports and experiment findings."""
        return service.findings(project_id)

    @server.tool()
    def build_model(project_id: str) -> dict[str, Any]:
        """Build an independent tiny causal model owned by this project."""
        return service.build(project_id)

    @server.tool()
    def reconstruct_model(project_id: str, source_path: str) -> dict[str, Any]:
        """Reconstruct supported project fixture in an empty project and compare logits."""
        return service.reconstruct(project_id, source_path)

    @server.tool()
    def quantize_model(project_id: str) -> dict[str, Any]:
        """Export lossy INT8 fixture weights without claiming INT8 compute acceleration."""
        return service.quantize(project_id)

    @server.tool()
    def compare_experiments(
        project_id: str, left_id: str, right_id: str
    ) -> dict[str, Any]:
        """Compare stored behavioral findings by prompt and sample occurrence."""
        return service.compare_experiments(project_id, left_id, right_id)

    @server.tool()
    def benchmark_experiment(
        project_id: str, finding_id: str, expected: dict[str, str]
    ) -> dict[str, Any]:
        """Score a stored behavioral finding against explicit exact-match text labels."""
        return service.benchmark(project_id, finding_id, expected)

    @server.tool()
    def generate(project_id: str, text: str) -> dict[str, Any]:
        """Run local CPU inference using project-generated weights."""
        return service.infer(project_id, text)

    @server.tool()
    def train_model(project_id: str, text: str, steps: int = 100) -> dict[str, Any]:
        """Train the tiny model output head on user-authorized text, not API probe outputs."""
        return service.train(project_id, text, steps)

    @server.tool()
    def probe_model(
        project_id: str,
        endpoint: str,
        model: str,
        prompts: list[str],
        authorized: bool = False,
        budget: int = 1024,
    ) -> dict[str, Any]:
        """Run bounded behavioral probes on an explicitly authorized endpoint."""
        return service.probe(project_id, endpoint, model, prompts, authorized, budget)

    @server.tool()
    def export_project(project_id: str, destination: str) -> dict[str, Any]:
        """Export findings and project-generated model; never copy imported weights."""
        return service.export(project_id, destination)

    @server.tool()
    def enqueue_task(
        project_id: str, kind: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Queue an inspect/build/infer/train or explicitly authorized agent job."""
        return service.enqueue(project_id, kind, payload)

    @server.tool()
    def run_tasks() -> dict[str, Any]:
        """Execute up to two queued jobs with persisted status."""
        return service.run_pending()

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, openWorldHint=False
        )
    )
    def task_status() -> list[dict[str, Any]]:
        """List persistent task statuses and results."""
        return service.tasks()

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, openWorldHint=False
        )
    )
    def hardware_status() -> dict[str, Any]:
        """Report actual hardware and current memory."""
        return service.hardware()

    # Bound signatures remain explicit typed MCP schemas; every tool calls the shared engines.
    for name in (
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
        "trace_local_model",
        "run_experiment",
    ):
        server.tool()(getattr(service, name))

    return server


def main() -> None:
    create_server(os.environ.get("MODELFORGE_HOME")).run(transport="stdio")


if __name__ == "__main__":
    main()
