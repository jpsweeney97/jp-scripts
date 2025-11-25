import asyncio
from mcp.server.fastmcp import FastMCP
from jpscripts.core.config import load_config
from jpscripts.core.notes_impl import append_to_daily_note
from jpscripts.core.console import get_logger

# Initialize logger for observability
logger = get_logger("mcp")

# Initialize the server
mcp = FastMCP("jpscripts")

# Load config independently of Typer
# We ignore the env_overrides meta return value here as we just need the config object
try:
    config, _ = load_config()
    logger.info(f"MCP Server loaded config from {config.notes_dir}")
except Exception as e:
    logger.error("Failed to load config during MCP startup", exc_info=e)
    # Fallback or exit could happen here, but we'll let the tool fail gracefully below

@mcp.tool()
def append_daily_note(message: str) -> str:
    """
    Append a log entry to the user's daily note system.
    This uses the configured 'notes_dir' from ~/.jpconfig.
    """
    try:
        # Resolve the path from the loaded config
        target_dir = config.notes_dir.expanduser()

        logger.debug(f"Appending note to {target_dir}: {message[:20]}...")

        # Execute the core logic
        path = append_to_daily_note(target_dir, message)

        return f"Successfully logged to daily note: {path}"
    except Exception as e:
        # Capture the full stack trace in the logs, return a clean error to the agent
        logger.error("MCP Tool Execution Failed: append_daily_note", exc_info=e)
        return f"Error appending note: {str(e)}"

if __name__ == "__main__":
    mcp.run()
