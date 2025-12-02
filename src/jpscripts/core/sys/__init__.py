"""System utilities package.

Organized submodules:
- execution: Command execution, sandboxes, run_safe_shell
- process: Process management (find, kill)
- audio: Audio device control (macOS)
- network: SSH hosts, temp HTTP server
- package: Homebrew utilities
"""

from jpscripts.core.sys.audio import get_audio_devices, set_audio_device
from jpscripts.core.sys.execution import (
    CommandResult,
    DockerSandbox,
    LocalSandbox,
    SandboxProtocol,
    get_sandbox,
    run_safe_shell,
)
from jpscripts.core.sys.network import get_ssh_hosts, run_temp_server
from jpscripts.core.sys.package import get_brew_info, search_brew
from jpscripts.core.sys.process import (
    ProcessInfo,
    find_processes,
    kill_process,
    kill_process_async,
)

__all__ = [
    # execution
    "CommandResult",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxProtocol",
    "get_sandbox",
    "run_safe_shell",
    # process
    "ProcessInfo",
    "find_processes",
    "kill_process",
    "kill_process_async",
    # audio
    "get_audio_devices",
    "set_audio_device",
    # network
    "get_ssh_hosts",
    "run_temp_server",
    # package
    "get_brew_info",
    "search_brew",
]
