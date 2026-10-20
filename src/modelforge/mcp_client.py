"""Narrow compatibility guard for MCP v1's Windows stdio shutdown race.

Only closed-stream errors raised after a normally completed client body are
handled. Initialization, tool errors, cancellation, and unrelated failures remain
visible. The official SDK still owns transport and child-process cleanup.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import sys
from typing import Any
import warnings

from anyio import BrokenResourceError, ClosedResourceError
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client as sdk_stdio_client


@asynccontextmanager
async def safe_stdio_client(parameters: StdioServerParameters) -> AsyncIterator[Any]:
    """Use SDK stdio, tolerating only its known Windows post-session close race."""
    completed = False
    try:
        async with sdk_stdio_client(parameters) as streams:
            yield streams
            completed = True
    except BaseExceptionGroup as group:
        if not completed or sys.platform != "win32":
            raise
        closed, remainder = group.split((BrokenResourceError, ClosedResourceError))
        if remainder is not None:
            raise remainder
        if closed is None:
            raise
        warnings.warn(
            "MCP SDK closed-stream race after completed Windows session; all requested operations returned before transport shutdown",
            RuntimeWarning,
            stacklevel=2,
        )
