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


async def _collect_repo_context(repo_root: Path) -> tuple[str, list[Path]]:
    """
    Run git status with a timeout to prevent hanging the swarm startup.
    """
    try:
        # FIX: Added 5-second timeout to prevent deadlock
        log, files = await asyncio.wait_for(gather_context("git status --short", repo_root), timeout=5.0)
        trimmed_log = log[-4000:] if len(log) > 4000 else log
        return trimmed_log, sorted(files)
    except asyncio.TimeoutError:
        logger.warning("Context collection timed out.")
        return "Context collection timed out (git status took too long).", []
    except Exception as exc:
        logger.debug("Context collection failed: %s", exc)
        return f"Context error: {exc}", []


class AgentProcess:
    def __init__(
        self,
        role: AgentRole,
        prompt: str,
        codex_bin: str,
        model: str,
        attached_files: Sequence[Path],
        env: dict[str, str],
        queue: asyncio.Queue[AgentUpdate],
        max_file_context_chars: int,
    ) -> None:
        self.role = role
        self._prompt = prompt
        self._codex_bin = codex_bin
        self._model = model
        self._files = list(attached_files)
        self._env = env
        self._queue = queue
        self._max_file_context_chars = max_file_context_chars
        self._process: asyncio.subprocess.Process | None = None
        self._stream_tasks: list[asyncio.Task] = []
        self._launched = False
        self._captured_output: list[str] = []
        self._captured_raw: list[str] = []

    async def _read_context_files(self) -> str:
        """Read context files asynchronously and truncate to 10KB each."""
        chunks: list[str] = []
        for path in self._files:
            snippet = await asyncio.to_thread(read_file_context, path, self._max_file_context_chars)
            if snippet is None:
                continue
            chunks.append(f"File: {path}\n```\n{snippet}\n```")

        if not chunks:
            return ""
        return "\n\nContext files:\n" + "\n\n".join(chunks)

    async def run(self) -> None:
        context_blob = await self._read_context_files()
        final_prompt = self._prompt
        if context_blob:
            final_prompt += "\n\n" + context_blob

        # codex CLI expects prompt before flags; pattern: [bin, exec, prompt, --json, --model, model]
        cmd: list[str] = [self._codex_bin, "exec", final_prompt, "--json", "--model", self._model]

        await self._queue.put(AgentUpdate(self.role, "status", "starting"))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=self._env,
        )
        self._launched = True
        await self._queue.put(
            AgentUpdate(
                self.role,
                "stdout",
                f"launched {self._codex_bin} (model: {self._model})",
            )
        )

        if self._process.stdout:
            self._stream_tasks.append(asyncio.create_task(self._pump_stream(self._process.stdout, "stdout")))
        if self._process.stderr:
            self._stream_tasks.append(asyncio.create_task(self._pump_stream(self._process.stderr, "stderr")))

        await self._queue.put(AgentUpdate(self.role, "status", "streaming"))
        return_code = await self._process.wait()

        await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        exit_msg = f"exit {return_code}"
        if return_code != 0 and self._launched:
            exit_msg += "; check `codex login` and network access"
        await self._queue.put(AgentUpdate(self.role, "exit", exit_msg))

    async def _pump_stream(self, stream: asyncio.StreamReader, kind: str) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode(errors="replace").rstrip()
            if kind == "stdout":
                self._captured_raw.append(decoded)
            message = _parse_event_line(decoded)
            if kind == "stdout":
                self._captured_output.append(message)
            await self._queue.put(AgentUpdate(self.role, kind, message))

    async def terminate(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        for task in self._stream_tasks:
            if not task.done():
                task.cancel()
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks, return_exceptions=True)

    @property
    def captured_stdout(self) -> str:
        return "\n".join(self._captured_output).strip()

    @property
    def captured_raw(self) -> str:
        return "\n".join(self._captured_raw).strip()


async def _stream_agent_updates(
    agents: Sequence[AgentProcess],
    queue: asyncio.Queue[AgentUpdate],
) -> AsyncIterator[AgentUpdate]:
    tasks = [asyncio.create_task(agent.run()) for agent in agents]
    active = len(tasks)

    try:
        while active:
            update = await queue.get()
            if update.kind == "exit":
                active -= 1
            yield update
    finally:
        for agent in agents:
            await agent.terminate()
        await asyncio.gather(*tasks, return_exceptions=True)


def _parse_agent_turn(agent: AgentProcess) -> tuple[AgentTurnResponse | None, str]:
    last_error: str = ""
    for payload in (agent.captured_raw, agent.captured_stdout):
        if not payload:
            continue
        try:
            return AgentTurnResponse.model_validate_json(payload), ""
        except ValidationError as exc:
            last_error = exc.json()
    return None, last_error or "No output captured from agent."


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
        context_log, context_files = await _collect_repo_context(self.root)
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
        agent = AgentProcess(
            role=role,
            prompt=prompt,
            codex_bin=self.codex_bin,
            model=self.model,
            attached_files=self.context_files,
            env=self.env,
            queue=self.queue,
            max_file_context_chars=self.max_file_context_chars,
        )
        async for update in _stream_agent_updates([agent], self.queue):
            yield update

        parsed, error_text = _parse_agent_turn(agent)
        if parsed:
            self.swarm_state = parsed.swarm_state
            next_step = parsed.next_step
        else:
            next_step = None
            yield AgentUpdate(role, "stderr", f"Invalid response; falling back to defaults: {error_text}")

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
