import asyncio
from mcp.server.fastmcp import FastMCP
from jpscripts.core.config import load_config
from jpscripts.core.notes_impl import append_to_daily_note

# Initialize the server
mcp = FastMCP("jpscripts")

# Load config independently of Typer
# We ignore the env_overrides meta return value here as we just need the config object
config, _ = load_config()

@mcp.tool()
def append_daily_note(message: str) -> str:
    """
    Append a log entry to the user's daily note system.
    This uses the configured 'notes_dir' from ~/.jpconfig.
    """
    try:
        # Resolve the path from the loaded config
        target_dir = config.notes_dir.expanduser()

        # Execute the core logic
        path = append_to_daily_note(target_dir, message)

        return f"Successfully logged to daily note: {path}"
    except Exception as e:
        return f"Error appending note: {str(e)}"

if __name__ == "__main__":
    mcp.run()
