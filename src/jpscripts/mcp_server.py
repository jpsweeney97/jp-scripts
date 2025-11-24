import asyncio
from mcp.server.fastmcp import FastMCP
from jpscripts.commands import notes, system

mcp = FastMCP("jpscripts")

@mcp.tool()
def append_daily_note(message: str) -> str:
    """Append a log entry to the user's daily note system."""
    # Wrap your existing logic from notes.py here
    return f"Logged to daily note: {message}"

@mcp.tool()
def kill_process_by_port(port: int) -> str:
    """Kill a process on a specific port (God-Mode admin action)."""
    # Wrap your existing logic from system.py here
    return f"Process on port {port} terminated."

if __name__ == "__main__":
    mcp.run()
