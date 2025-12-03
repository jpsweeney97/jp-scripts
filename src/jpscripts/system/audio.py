"""Audio device management utilities (macOS).

Provides:
- get_audio_devices to list available audio output devices
- set_audio_device to switch the active audio device

Requires SwitchAudioSource to be installed (brew install switchaudio-osx).
"""

from __future__ import annotations

import asyncio
import shutil

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)


async def get_audio_devices() -> Result[list[str], SystemResourceError]:
    """List available audio output devices.

    Returns:
        Result containing list of device names on success
    """
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))

    try:
        proc = await asyncio.create_subprocess_exec(
            switch_cmd,
            "-a",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))
    except Exception as exc:
        return Err(SystemResourceError("Failed to list audio devices", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return Err(
            SystemResourceError(
                "SwitchAudioSource failed",
                context={"stderr": stderr.decode().strip(), "returncode": proc.returncode},
            )
        )

    return Ok([line.strip() for line in stdout.decode().splitlines() if line.strip()])


async def set_audio_device(device_name: str) -> Result[None, SystemResourceError]:
    """Set the active audio output device.

    Args:
        device_name: Name of the device to activate

    Returns:
        Result containing None on success
    """
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))

    try:
        proc = await asyncio.create_subprocess_exec(
            switch_cmd,
            "-s",
            device_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))
    except Exception as exc:
        return Err(
            SystemResourceError("Failed to switch audio device", context={"error": str(exc)})
        )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode().strip() or stdout.decode().strip()
        return Err(
            SystemResourceError(
                "Failed to switch device", context={"stderr": message, "device": device_name}
            )
        )

    return Ok(None)


__all__ = [
    "get_audio_devices",
    "set_audio_device",
]
