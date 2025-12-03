"""System utilities package.

Organized submodules:
- execution: Command execution, sandboxes, run_safe_shell
- process: Process management (find, kill)
- audio: Audio device control (macOS)
- network: SSH hosts, temp HTTP server
- package: Homebrew utilities
"""

from jpscripts.system.audio import get_audio_devices, set_audio_device
from jpscripts.system.execution import (
    CommandResult,
    DockerSandbox,
    LocalSandbox,
    SandboxProtocol,
    get_sandbox,
    run_cpu_bound,
    run_safe_shell,
)
from jpscripts.system.network import get_ssh_hosts, run_temp_server
from jpscripts.system.package import get_brew_info, search_brew
from jpscripts.system.process import (
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
    # process
    "ProcessInfo",
    "SandboxProtocol",
    "find_processes",
    # audio
    "get_audio_devices",
    # package
    "get_brew_info",
    "get_sandbox",
    # network
    "get_ssh_hosts",
    "kill_process",
    "kill_process_async",
    "run_cpu_bound",
    "run_safe_shell",
    "run_temp_server",
    "search_brew",
    "set_audio_device",
]
