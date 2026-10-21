"""The Windows shutdown guard must not hide operation or transport failures."""

from contextlib import asynccontextmanager
import asyncio
from typing import Any

from anyio import BrokenResourceError
from mcp import StdioServerParameters
import pytest
import modelforge.mcp_client as client


@pytest.mark.parametrize(
    "platform,body_error,other_error,accepted",
    [
        ("win32", False, False, True),
        ("darwin", False, False, False),
        ("win32", True, False, False),
        ("win32", False, True, False),
    ],
)
def test_close_race_is_only_tolerated_after_successful_windows_body(
    monkeypatch: Any, platform: str, body_error: bool, other_error: bool, accepted: bool
) -> None:
    @asynccontextmanager
    async def transport(_: Any) -> Any:
        try:
            yield ("actual-read", "actual-write")
        finally:
            errors: list[Exception] = [BrokenResourceError()]
            if other_error:
                errors.append(ValueError("real failure"))
            raise ExceptionGroup("SDK teardown", errors)

    monkeypatch.setattr(client, "sdk_stdio_client", transport)
    monkeypatch.setattr(client.sys, "platform", platform)

    async def run() -> None:
        async with client.safe_stdio_client(
            StdioServerParameters(command="unused")
        ) as streams:
            assert streams == ("actual-read", "actual-write")
            if body_error:
                raise RuntimeError("operation did not complete")

    if accepted:
        with pytest.warns(RuntimeWarning, match="after completed"):
            asyncio.run(run())
    else:
        with pytest.raises(BaseExceptionGroup):
            asyncio.run(run())


@pytest.mark.parametrize("body_error", [False, True])
def test_session_teardown_race_after_actual_body(
    monkeypatch: Any, body_error: bool
) -> None:
    @asynccontextmanager
    async def transport(_: Any) -> Any:
        yield ("read", "write")

    @asynccontextmanager
    async def session(*_: Any) -> Any:
        try:
            yield "session"
        finally:
            raise ExceptionGroup("session close", [BrokenResourceError()])

    monkeypatch.setattr(client, "sdk_stdio_client", transport)
    monkeypatch.setattr(client, "ClientSession", session)
    monkeypatch.setattr(client.sys, "platform", "win32")

    async def run() -> None:
        async with client.safe_mcp_session(
            StdioServerParameters(command="unused")
        ) as value:
            assert value == "session"
            if body_error:
                raise RuntimeError("unfinished operation")

    if body_error:
        with pytest.raises(BaseExceptionGroup):
            asyncio.run(run())
    else:
        with pytest.warns(RuntimeWarning, match="after completed"):
            asyncio.run(run())
