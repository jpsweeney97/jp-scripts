from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.context import gather_context, read_file_context
from jpscripts.core.engine import AgentEngine, Message, PreparedPrompt

logger = get_logger(__name__)


class AgentRole(Enum):
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    QA = "qa"

    @property
    def label(self) -> str:
        return {
            AgentRole.ARCHITECT: "Architect",
            AgentRole.ENGINEER: "Engineer",
            AgentRole.QA: "QA",
        }[self]


ROLE_PRIMERS: dict[AgentRole, str] = {
    AgentRole.ARCHITECT: (
        "You design the technical plan and break down the work. Produce concise steps and"
        " call out risks for implementation. Respond using the Swarm State JSON schema"
        " only—no markdown or YAML."
    ),
    AgentRole.ENGINEER: (
        "You execute the plan, propose concrete code changes, and resolve edge cases."
        " Keep responses terse and implementation-focused."
    ),
    AgentRole.QA: (
        "You validate the plan and implementation. Identify missing tests, regressions,"
        " and safety issues. Provide clear acceptance criteria."
    ),
}


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    constraints: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    status: Literal["pending", "in_progress", "done"] = "pending"


class SwarmState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: Objective
    plan_steps: list[PlanStep] = Field(default_factory=list)
    current_phase: Literal["planning", "coding", "verifying"] = "planning"
    artifacts: list[str] = Field(default_factory=list)


class AgentTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    swarm_state: SwarmState
    next_step: Literal["architect", "engineer", "qa", "finish"] = "finish"


@dataclass
class AgentUpdate:
    role: AgentRole
    kind: str  # stdout | stderr | status | exit
    content: str


def _codex_path() -> str:
    path = shutil.which("codex")
    if not path:
        raise RuntimeError("codex binary not found. Install with `brew install codex` or `npm install -g @openai/codex`.")
    return path


def _build_env(config: AppConfig, safe_mode: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["JP_NOTES_DIR"] = str(config.notes_dir.expanduser())
    env["JP_WORKSPACE_ROOT"] = str(config.workspace_root.expanduser())
    env["JP_LOG_LEVEL"] = str(config.log_level)
    if config.worktree_root:
        env["JP_WORKTREE_ROOT"] = str(config.worktree_root.expanduser())
    if safe_mode:
        env["JP_SAFE_MODE"] = "1"
    return env


def _render_config_context(config: AppConfig, safe_mode: bool) -> str:
    lines = [
        f"workspace_root: {config.workspace_root.expanduser()}",
        f"notes_dir: {config.notes_dir.expanduser()}",
        f"log_level: {config.log_level}",
    ]
    if config.worktree_root:
        lines.append(f"worktree_root: {config.worktree_root.expanduser()}")
    if safe_mode:
        lines.append("safe_mode: true (default config due to load error)")
    return "\n".join(lines)


def _format_file_snippets(files: Sequence[Path], max_files: int = 3, max_chars: int = 1200) -> str:
    snippets: list[str] = []
    for path in list(files)[:max_files]:
        snippet = read_file_context(path, max_chars)
        if snippet:
            snippets.append(f"File: {path}\n```\n{snippet}\n```")
    if not snippets:
        return ""
    return "\n\nContext files:\n" + "\n\n".join(snippets)


def _compose_prompt(
    role: AgentRole,
    objective: str,
    swarm_state: SwarmState,
    context_log: str,
    config: AppConfig,
    safe_mode: bool,
    repo_root: Path,
    context_files: Sequence[Path],
    max_file_context_chars: int,
) -> str:
    primer = ROLE_PRIMERS.get(role, "")
    config_summary = _render_config_context(config, safe_mode)
    context_section = f"\n\nRepository context (from git status):\n{context_log.strip()}" if context_log.strip() else ""
    file_section = _format_file_snippets(context_files, max_chars=max_file_context_chars)
    swarm_json = swarm_state.model_dump_json(indent=2)
    handoff_guidance = {
        AgentRole.ARCHITECT: "Set `next_step` to 'engineer' after drafting the plan.",
        AgentRole.ENGINEER: "Set `next_step` to 'qa' after proposing the implementation.",
        AgentRole.QA: "Set `next_step` to 'finish' if tests pass, or 'engineer' if changes are required.",
    }[role]
    schema_instruction = (
        "\n\nRespond ONLY with JSON that matches this schema. Do not add markdown or explanations:\n"
        f"{json.dumps(AgentTurnResponse.model_json_schema(), indent=2)}\n"
        f"Handoff rules: {handoff_guidance}"
    )
    return (
        f"You are the {role.label} in a three-agent swarm (Architect, Engineer, QA).\n"
        f"{primer}\n"
        f"Objective:\n{objective.strip()}\n\n"
        f"Current Swarm State (JSON):\n{swarm_json}\n\n"
        f"Repo root: {repo_root}\n"
        f"Safe Mode Config:\n{config_summary}"
        f"{context_section}"
        f"{file_section}"
        f"{schema_instruction}\n\n"
        "Coordinate asynchronously; keep responses compact so they can be relayed in real time."
    )


def _parse_event_line(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    data = payload.get("data") or {}
    for key in ("delta", "assistant_message", "message"):
        value = data.get(key) or payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    event_type = payload.get("event") or payload.get("type") or ""
    return f"{event_type}: {raw}"


async def _collect_fallback_context(repo_root: Path, timeout: float) -> tuple[str, list[Path]]:
    fallback_timeout = max(timeout / 2, 1.0)
    try:
        log, files = await asyncio.wait_for(gather_context("ls -a", repo_root), timeout=fallback_timeout)
        trimmed_log = log[-2000:] if len(log) > 2000 else log
        return f"Fallback directory listing:\n{trimmed_log}", sorted(files)
    except asyncio.TimeoutError:
        logger.warning("Fallback context collection timed out after %.2f seconds.", fallback_timeout)
    except Exception as exc:
        logger.debug("Fallback context collection failed: %s", exc)
    return "", []


async def _collect_repo_context(repo_root: Path, config: AppConfig) -> tuple[str, list[Path]]:
    """
    Run git status with a configurable timeout to prevent hanging swarm startup.
    """
    timeout = max(config.git_status_timeout, 0.1)
    try:
        log, files = await asyncio.wait_for(gather_context("git status --short", repo_root), timeout=timeout)
        trimmed_log = log[-4000:] if len(log) > 4000 else log
        return trimmed_log, sorted(files)
    except asyncio.TimeoutError:
        logger.warning("Git status timed out after %.2f seconds; attempting fallback.", timeout)
        fallback_log, fallback_files = await _collect_fallback_context(repo_root, timeout)
        if fallback_log:
            return fallback_log, fallback_files
        return "Git status timed out; Architect context will be limited.", []
    except Exception as exc:
        logger.debug("Context collection failed: %s", exc)
        return f"Context error: {exc}", []


def _parse_agent_turn_payload(payload: str, fallback: str = "") -> tuple[AgentTurnResponse | None, str]:
    last_error = ""
    for candidate in (payload, fallback):
        if not candidate:
            continue
        try:
            return AgentTurnResponse.model_validate_json(candidate), ""
        except ValidationError as exc:
            last_error = exc.json()
    return None, last_error or "No output captured from agent."


def _parse_agent_turn(obj: object, stdout: str = "") -> tuple[AgentTurnResponse | None, str]:
    if isinstance(obj, str):
        return _parse_agent_turn_payload(obj, stdout)
    raw = getattr(obj, "captured_raw", "") if obj is not None else ""
    fallback = getattr(obj, "captured_stdout", "") if obj is not None else ""
    return _parse_agent_turn_payload(raw, fallback)


def parse_swarm_response(payload: str) -> AgentTurnResponse:
    return AgentTurnResponse.model_validate_json(payload)


class SwarmController:
    def __init__(
        self,
        objective: str,
        roles: Iterable[AgentRole],
        config: AppConfig | None,
        repo_root: Path | None,
        model: str | None,
        safe_mode: bool,
        *,
        max_turns: int = 15,
    ) -> None:
        self.objective = objective.strip()
        self.roles = list(roles)
        self.config = config or AppConfig()
        self.root = (repo_root or Path.cwd()).expanduser()
        self.model = model or self.config.default_model
        self.safe_mode = safe_mode
        self.max_turns = max_turns
        self.queue: asyncio.Queue[AgentUpdate] = asyncio.Queue()
        self.codex_bin = _codex_path()
        self.env = _build_env(self.config, safe_mode)
        self.context_log: str = ""
        self.context_files: list[Path] = []
        self.swarm_state = SwarmState(
            objective=Objective(summary=self.objective),
            plan_steps=[],
            current_phase="planning",
            artifacts=[],
        )
        self.max_file_context_chars = self.config.max_file_context_chars
        self._next_role: AgentRole | None = None
        self._engines: dict[AgentRole, AgentEngine[AgentResponse]] = {}

    def _starting_role(self) -> AgentRole | None:
        if AgentRole.ARCHITECT in self.roles:
            return AgentRole.ARCHITECT
        return self.roles[0] if self.roles else None

    def _default_next_role(self, current: AgentRole) -> AgentRole | None:
        if current is AgentRole.ARCHITECT and AgentRole.ENGINEER in self.roles:
            return AgentRole.ENGINEER
        if current is AgentRole.ENGINEER and AgentRole.QA in self.roles:
            return AgentRole.QA
        return None

    def _normalize_next_step(self, step: str | None) -> AgentRole | None:
        if not step:
            return None
        lowered = step.lower().strip()
        if lowered == "finish":
            return None
        for role in AgentRole:
            if role.value == lowered:
                return role
        return None

    def _resolve_next_role(self, current: AgentRole, requested_step: str | None) -> AgentRole | None:
        suggested = self._normalize_next_step(requested_step)
        allowed: dict[AgentRole, set[AgentRole | None]] = {
            AgentRole.ARCHITECT: {AgentRole.ENGINEER},
            AgentRole.ENGINEER: {AgentRole.QA},
            AgentRole.QA: {AgentRole.ENGINEER, None},
        }

        if suggested in allowed.get(current, set()) and (suggested is None or suggested in self.roles):
            return suggested

        return self._default_next_role(current)

    async def _initialize(self) -> None:
        context_log, context_files = await _collect_repo_context(self.root, self.config)
        self.context_log = context_log
        self.context_files = context_files
        self.swarm_state = SwarmState(
            objective=Objective(summary=self.objective),
            plan_steps=[],
            current_phase="planning",
            artifacts=[str(path) for path in context_files],
        )

    async def _run_turn(self, role: AgentRole) -> AsyncIterator[AgentUpdate]:
        prompt = _compose_prompt(
            role,
            self.objective,
            self.swarm_state,
            self.context_log,
            self.config,
            self.safe_mode,
            self.root,
            self.context_files,
            self.max_file_context_chars,
        )

        async def _prompt_builder(_history: Sequence[Message]) -> PreparedPrompt:
            return PreparedPrompt(prompt=prompt, attached_files=self.context_files)

        async def _fetch_response(prepared: PreparedPrompt) -> str:
            cmd = [
                self.codex_bin,
                "exec",
                prepared.prompt,
                "--json",
                "--model",
                self.model,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            stdout = stdout_bytes.decode(errors="replace").strip()
            stderr = stderr_bytes.decode(errors="replace").strip()
            if stderr:
                await self.queue.put(AgentUpdate(role, "stderr", stderr))
            for line in stdout.splitlines():
                parsed_line = _parse_event_line(line)
                await self.queue.put(AgentUpdate(role, "stdout", parsed_line))
            return stdout or ""

        engine = AgentEngine[AgentTurnResponse](
            persona=role.label,
            model=self.model,
            prompt_builder=_prompt_builder,
            fetch_response=_fetch_response,
            parser=parse_swarm_response,
            template_root=self.root,
        )
        self._engines[role] = engine

        await self.queue.put(AgentUpdate(role, "status", "starting"))
        response = await engine.step([])
        await self.queue.put(AgentUpdate(role, "exit", "0"))

        self.swarm_state = response.swarm_state
        next_step = response.next_step

        yield AgentUpdate(role, "status", f"completed turn -> next: {next_step or 'finish'}")
        next_role = self._resolve_next_role(role, next_step)
        if next_role is not None:
            yield AgentUpdate(next_role, "status", "queued")
        self._next_role = next_role

    async def run(self) -> AsyncIterator[AgentUpdate]:
        await self._initialize()
        current_role = self._starting_role()
        if current_role is None:
            yield AgentUpdate(AgentRole.ARCHITECT, "stderr", "No roles available for swarm.")
            return

        yield AgentUpdate(current_role, "status", "context loaded")
        turn = 0
        while current_role is not None and turn < self.max_turns:
            turn += 1
            async for update in self._run_turn(current_role):
                yield update
            current_role = getattr(self, "_next_role", None)

        if turn >= self.max_turns:
            yield AgentUpdate(current_role or AgentRole.QA, "stderr", f"Max turns ({self.max_turns}) reached.")


async def swarm_chat(
    objective: str,
    roles: Iterable[AgentRole],
    config: AppConfig | None = None,
    repo_root: Path | None = None,
    model: str | None = None,
    safe_mode: bool = False,
) -> AsyncIterator[AgentUpdate]:
    """
    Launch a swarm of Codex agents and stream their output via a shared queue.
    """
    controller = SwarmController(
        objective=objective,
        roles=roles,
        config=config,
        repo_root=repo_root,
        model=model,
        safe_mode=safe_mode,
    )
    async for update in controller.run():
        yield update
