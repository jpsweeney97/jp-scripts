from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from typing import Any

import pytest


INIT_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    },
}
SHUTDOWN_REQUEST = {"jsonrpc": "2.0", "id": 2, "method": "shutdown"}
EXIT_NOTIFICATION = {"jsonrpc": "2.0", "method": "exit"}


async def _write_message(proc: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
    message = json.dumps(payload)
    assert proc.stdin is not None
    proc.stdin.write(message.encode("utf-8") + b"\n")
    await proc.stdin.drain()


async def _read_json_message(stream: asyncio.StreamReader, timeout: float = 10.0) -> str:
    content_length: int | None = None
    # Support either Content-Length-delimited or newline-delimited JSON.
    while True:
        line = await asyncio.wait_for(stream.readline(), timeout=timeout)
        if not line:
            return ""
        stripped = line.strip()
        if not stripped:
            if content_length:
                body = await asyncio.wait_for(stream.readexactly(content_length), timeout=timeout)
                return body.decode("utf-8", errors="replace")
            continue
        decoded = line.decode("utf-8", errors="ignore")
        if decoded.lower().startswith("content-length:"):
            try:
                content_length = int(decoded.split(":", 1)[1].strip())
            except ValueError:
                content_length = None
            continue
        if stripped.startswith(b"{"):
            return line.decode("utf-8", errors="replace")
    return ""


@pytest.mark.asyncio
async def test_mcp_server_handshake() -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "jpscripts.mcp.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        await _write_message(proc, INIT_REQUEST)
        assert proc.stdout is not None
        response = await _read_json_message(proc.stdout, timeout=15.0)

        assert response, "No response received from MCP server."
        payload = json.loads(response)
        assert "result" in payload
        server_info = payload.get("result", {}).get("serverInfo", {})
        assert server_info.get("name") == "jpscripts"

        await _write_message(proc, SHUTDOWN_REQUEST)
        await _write_message(proc, EXIT_NOTIFICATION)
        if proc.stdin:
            proc.stdin.close()

        await asyncio.wait_for(proc.wait(), timeout=15.0)
        assert proc.returncode == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
