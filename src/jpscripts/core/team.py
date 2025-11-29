from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Iterable, Literal, Sequence, cast

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML, YAMLError

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.context import gather_context, read_file_context
from jpscripts.core.engine import AgentEngine, Message, PreparedPrompt
from jpscripts.core import security
from jpscripts.providers import (
    CompletionOptions,
    LLMProvider,
    Message as ProviderMessage,
)
from jpscripts.providers.factory import get_provider

logger = get_logger(__name__)


@dataclass
class Persona:
    name: str
    style: str
    color: str = "cyan"

    @property
    def label(self) -> str:
        return self.name


class _PersonaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    style: str
    color: str | None = None


class _SwarmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    personas: list[_PersonaConfig]


def _load_swarm_config(config_path: Path) -> dict[str, object] | None:
    yaml_loader = YAML(typ="safe")

    def _read() -> dict[str, object] | None:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml_loader.load(handle)
        if loaded is None:
            return {}
        if isinstance(loaded, dict):
            return cast(dict[str, object], loaded)
        raise ValueError("Swarm config must be a mapping.")

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.to_thread(_read))

    logger.debug("Skipping swarm config load while event loop is running; using defaults.")
    return None


def _load_configured_swarm() -> list[Persona] | None:
    config_path = Path.home() / ".jpscripts" / "swarms.yaml"
    if not config_path.exists():
        logger.debug("Swarm config not found at %s; using defaults.", config_path)
        return None

    if not config_path.is_file():
        logger.debug("Swarm config path is not a file: %s; using defaults.", config_path)
        return None

    try:
        raw_data = _load_swarm_config(config_path)
    except (OSError, YAMLError, ValueError) as exc:
        logger.debug("Failed to load swarm config at %s: %s; using defaults.", config_path, exc)
        return None

    if raw_data is None:
        logger.debug("Swarm config at %s returned no data; using defaults.", config_path)
        return None

    try:
        config = _SwarmConfig.model_validate(raw_data)
    except ValidationError as exc:
        logger.debug("Invalid swarm config at %s: %s; using defaults.", config_path, exc)
        return None

    if not config.personas:
        logger.debug("Swarm config at %s contains no personas; using defaults.", config_path)
        return None

    personas: list[Persona] = []
    for persona_config in config.personas:
        persona_kwargs: dict[str, str] = {
            "name": persona_config.name,
            "style": persona_config.style,
        }
        if persona_config.color:
            persona_kwargs["color"] = persona_config.color
        personas.append(Persona(**persona_kwargs))

    return personas


def get_default_swarm() -> list[Persona]:
    """Return the default Architect/Engineer/QA personas."""
    configured = _load_configured_swarm()
    if configured:
        return configured

    return [
        Persona(
            name="Architect",
            style=(
                "You design the technical plan and break down the work. Produce concise steps and "
                "call out risks for implementation. Respond using the Swarm State JSON schema "
                "only—no markdown or YAML."
            ),
            color="cyan",
        ),
        Persona(
            name="Engineer",
            style=(
                "You execute the plan, propose concrete code changes, and resolve edge cases. "
                "Keep responses terse and implementation-focused."
            ),
            color="green",
        ),
        Persona(
            name="QA",
            style=(
                "You validate the plan and implementation. Identify missing tests, regressions, "
                "and safety issues. Provide clear acceptance criteria."
            ),
            color="yellow",
        ),
    ]


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


class SpawnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: str
    objective: str
    context_files: list[str] = Field(default_factory=list)


class AgentTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    swarm_state: SwarmState
    spawn_tasks: list[SpawnRequest] = Field(default_factory=list)
    next_step: Literal["architect", "engineer", "qa", "finish"] | None = Field(
        default=None, description="DEPRECATED: single-step handoff; use spawn_tasks."
    )


@dataclass
class AgentUpdate:
    role: Persona
    kind: str  # stdout | stderr | status | exit
    content: str


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


def _resolve_template_root() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    return security.validate_path(package_root / "templates", package_root)


@lru_cache(maxsize=1)
def _get_template_environment(template_root: Path) -> Environment:
    return Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)


def _render_swarm_prompt(
    persona: Persona,
    objective: str,
    swarm_state: SwarmState,
    context_log: str,
    config: AppConfig,
    safe_mode: bool,
    repo_root: Path,
    context_files: Sequence[Path],
    max_file_context_chars: int,
) -> str:
    template_root = _resolve_template_root()
    env = _get_template_environment(template_root)
    template_name = f"swarm_{persona.name.lower()}.j2"
    try:
        template = env.get_template(template_name)
    except TemplateNotFound as exc:
        raise FileNotFoundError(f"Template {template_name} not found in {template_root}") from exc

    context_section = f"\n\nRepository context (from git status):\n{context_log.strip()}" if context_log.strip() else ""
    file_section = _format_file_snippets(context_files, max_chars=max_file_context_chars)
    handoff_guidance = (
        "Use `spawn_tasks` to launch parallel work (e.g., multiple Engineers on distinct files). "
        "Use `next_step` only if a single sequential handoff is needed (deprecated)."
    )

    render_context = {
        "persona_label": persona.label,
        "persona_style": persona.style,
        "objective": objective.strip(),
        "swarm_json": swarm_state.model_dump_json(indent=2),
        "repo_root": repo_root,
        "config_summary": _render_config_context(config, safe_mode),
        "context_log": context_section,
        "file_section": file_section,
        "schema_json": json.dumps(AgentTurnResponse.model_json_schema(), indent=2),
        "handoff_guidance": handoff_guidance,
    }
    return template.render(**render_context)


async def _collect_fallback_context(repo_root: Path, timeout: float) -> tuple[str, list[Path]]:
    fallback_timeout = max(timeout / 2, 1.0)
    try:
        context_result = await asyncio.wait_for(gather_context("ls -a", repo_root), timeout=fallback_timeout)
        log = context_result.output
        files = context_result.files
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
        context_result = await asyncio.wait_for(gather_context("git status --short", repo_root), timeout=timeout)
        log = context_result.output
        files = context_result.files
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
        roles: Iterable[Persona],
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
        self.provider: LLMProvider = get_provider(self.config, model_id=self.model)
        self.context_log: str = ""
        self.context_files: list[Path] = []
        self.swarm_state = SwarmState(
            objective=Objective(summary=self.objective),
            plan_steps=[],
            current_phase="planning",
            artifacts=[],
        )
        self.max_file_context_chars = self.config.max_file_context_chars
        self._next_role: Persona | None = None
        self._engines: dict[str, AgentEngine[AgentTurnResponse]] = {}
        self.pending_tasks: list[SpawnRequest] = []
        self.extra_context_files: list[Path] = []

    def _starting_role(self) -> Persona | None:
        if not self.roles:
            return None
        return self.roles[0]

    def _default_next_role(self, current: Persona) -> Persona | None:
        try:
            current_index = self.roles.index(current)
        except ValueError:
            return None
        if current_index + 1 < len(self.roles):
            return self.roles[current_index + 1]
        return None

    def _normalize_next_step(self, step: str | None) -> Persona | None:
        if not step:
            return None
        lowered = step.lower().strip()
        if lowered == "finish":
            return None
        for persona in self.roles:
            if persona.name.lower() == lowered:
                return persona
        return None

    def _resolve_next_role(self, current: Persona, requested_step: str | None) -> Persona | None:
        suggested = self._normalize_next_step(requested_step)
        if suggested in self.roles:
            return suggested
        return self._default_next_role(current)

    def _select_roles_for_spawn(self, persona_name: str | None) -> list[Persona]:
        if persona_name:
            matched = [role for role in self.roles if role.name.lower() == persona_name.lower()]
            if matched:
                # Include QA if available to verify spawned work
                qa_roles = [role for role in self.roles if role.name.lower() == "qa"]
                return matched + qa_roles
        # Fallback to all roles minus architect to avoid recursive planning
        return [role for role in self.roles if role.name.lower() != "architect"]

    def _create_subcontroller(self, request: SpawnRequest) -> "SwarmController":
        sub_roles = self._select_roles_for_spawn(request.persona)
        controller = SwarmController(
            objective=request.objective,
            roles=sub_roles,
            config=self.config,
            repo_root=self.root,
            model=self.model,
            safe_mode=self.safe_mode,
            max_turns=self.max_turns,
        )
        context_paths: list[Path] = []
        for raw in request.context_files:
            try:
                resolved = security.validate_path(self.root / raw, self.root)
                context_paths.append(resolved)
            except Exception:
                continue
        controller.extra_context_files = context_paths
        return controller

    async def _initialize(self) -> None:
        context_log, context_files = await _collect_repo_context(self.root, self.config)
        self.context_log = context_log
        self.context_files = context_files + self.extra_context_files
        self.swarm_state = SwarmState(
            objective=Objective(summary=self.objective),
            plan_steps=[],
            current_phase="planning",
            artifacts=[str(path) for path in context_files],
        )

    async def _run_turn(self, role: Persona) -> AsyncIterator[AgentUpdate]:
        prompt = _render_swarm_prompt(
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
            messages = [ProviderMessage(role="user", content=prepared.prompt)]
            options = CompletionOptions(
                temperature=prepared.temperature,
                reasoning_effort=prepared.reasoning_effort,
                max_tokens=8192,
            )
            chunks: list[str] = []
            async for chunk in self.provider.stream(messages, model=self.model, options=options):
                if chunk.content:
                    chunks.append(chunk.content)
                    await self.queue.put(AgentUpdate(role, "stdout", chunk.content))
            return "".join(chunks)

        engine = AgentEngine[AgentTurnResponse](
            persona=role.label,
            model=self.model,
            prompt_builder=_prompt_builder,
            fetch_response=_fetch_response,
            parser=parse_swarm_response,
            template_root=self.root,
        )
        self._engines[role.name] = engine

        await self.queue.put(AgentUpdate(role, "status", "starting"))
        response = await engine.step([])
        await self.queue.put(AgentUpdate(role, "exit", "0"))

        self.swarm_state = response.swarm_state
        if response.spawn_tasks:
            self.pending_tasks.extend(response.spawn_tasks)

        next_step = response.next_step
        yield AgentUpdate(role, "status", f"completed turn -> next: {next_step or 'auto'}")
        next_role = self._resolve_next_role(role, next_step)
        if next_role is not None:
            yield AgentUpdate(next_role, "status", "queued")
        self._next_role = next_role

    async def run(self) -> AsyncIterator[AgentUpdate]:
        await self._initialize()
        current_role = self._starting_role()
        if current_role is None:
            yield AgentUpdate(Persona(name="Unknown", style="", color="red"), "stderr", "No roles available for swarm.")
            return

        yield AgentUpdate(current_role, "status", "context loaded")
        turn = 0
        while current_role is not None and turn < self.max_turns:
            turn += 1
            async for update in self._run_turn(current_role):
                yield update
            if self.pending_tasks:
                controllers = [self._create_subcontroller(task) for task in self.pending_tasks]
                self.pending_tasks = []

                async def _collect_updates(controller: "SwarmController") -> list[AgentUpdate]:
                    collected: list[AgentUpdate] = []
                    async for upd in controller.run():
                        collected.append(upd)
                    return collected

                results = await asyncio.gather(*[_collect_updates(ctrl) for ctrl in controllers], return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):  # pragma: no cover - defensive
                        yield AgentUpdate(Persona(name="Architect", style="", color="red"), "stderr", f"Sub-swarm error: {result}")
                        continue
                    updates = cast(list[AgentUpdate], result)
                    for upd in updates:
                        yield upd
            current_role = getattr(self, "_next_role", None)

        if turn >= self.max_turns:
            yield AgentUpdate(
                current_role or Persona(name="QA", style="", color="yellow"),
                "stderr",
                f"Max turns ({self.max_turns}) reached.",
            )


async def swarm_chat(
    objective: str,
    roles: Iterable[Persona],
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
