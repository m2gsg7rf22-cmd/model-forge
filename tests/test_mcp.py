"""Real SDK client/server subprocess transport, no mocked MCP responses."""

import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_real_mcp_transport(tmp_path):
    async def run():
        env = dict(os.environ, MODELFORGE_HOME=str(tmp_path))
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "modelforge.mcp_server"], env=env
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert len(tools.tools) >= 12
                created = await session.call_tool(
                    "create_project", {"name": "MCP transport"}
                )
                assert not created.isError
                projects = await session.call_tool("list_projects", {})
                assert "MCP transport" in projects.content[0].text
                hardware = await session.call_tool("hardware_status", {})
                assert "memory_available" in hardware.content[0].text
                inventory = await session.call_tool(
                    "video_runtime_inventory", {"model_root": str(tmp_path / "missing")}
                )
                assert not inventory.isError
                assert '"generation_verified":false' in inventory.content[0].text.replace(" ", "")

    asyncio.run(run())
