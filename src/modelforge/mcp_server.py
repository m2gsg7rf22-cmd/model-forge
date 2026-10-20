"""Official MCP SDK stdio interface to the real shared service."""

from __future__ import annotations
import os
from typing import Any
from mcp.server.fastmcp import FastMCP
from .service import Service


def create_server(data_dir: str | None = None) -> FastMCP:
    service = Service(data_dir)
    server = FastMCP("ModelForge")

    @server.tool()
    def create_project(name: str) -> dict[str, Any]:
        """Create a persistent model research project."""
        return service.create_project(name)

    @server.tool()
    def list_projects() -> list[dict[str, Any]]:
        """List saved projects."""
        return service.projects()

    @server.tool()
    def inspect_model(project_id: str, path: str) -> dict[str, Any]:
        """Inspect authorized local model metadata without executing model code."""
        return service.inspect(project_id, path)

    @server.tool()
    def read_findings(project_id: str) -> list[dict[str, Any]]:
        """Read evidence, architectural reports and experiment findings."""
        return service.findings(project_id)

    @server.tool()
    def build_model(project_id: str) -> dict[str, Any]:
        """Build an independent tiny causal model owned by this project."""
        return service.build(project_id)

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

    @server.tool()
    def task_status() -> list[dict[str, Any]]:
        """List persistent task statuses and results."""
        return service.tasks()

    @server.tool()
    def hardware_status() -> dict[str, Any]:
        """Report actual hardware and current memory."""
        return service.hardware()

    return server


def main() -> None:
    create_server(os.environ.get("MODELFORGE_HOME")).run(transport="stdio")


if __name__ == "__main__":
    main()
