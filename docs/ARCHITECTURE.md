# jpscripts Architecture

This document describes the high-level architecture of jpscripts, including module interactions and data flow.

---

## Module Interaction Diagram

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        main[main.py]
        commands[commands/]
    end

    subgraph Core["Core Layer"]
        config[config.py]
        console[console.py]
        runtime[runtime.py]
        result[result.py]
        mcp_registry[mcp_registry.py]

        subgraph Agent["Agent Subsystem"]
            agent_engine[agent/engine.py]
            agent_exec[agent/execution.py]
            agent_prompt[agent/prompting.py]
            agent_strategies[agent/strategies.py]
        end

        subgraph Memory["Memory Subsystem"]
            memory_store[memory/store/]
            memory_embed[memory/embedding.py]
            memory_retrieve[memory/retrieval.py]
            memory_patterns[memory/patterns.py]
        end

        subgraph CoreSecurity["Security"]
            security[core/security/]
            command_val[core/security/command.py]
            safety[core/security/safety.py]
            rate_limit[core/security/rate_limit.py]
        end

        subgraph Evolution["Evolution Subsystem"]
            evolution_orch[core/evolution/orchestrator.py]
            evolution_pr[core/evolution/pr.py]
        end

        subgraph SystemOps["System Operations"]
            sys_exec[core/sys/execution.py]
            sys_process[core/sys/process.py]
        end
    end

    subgraph Analysis["Analysis Subsystem"]
        analysis_complexity[analysis/complexity.py]
        analysis_skeleton[analysis/skeleton.py]
    end

    subgraph Features["Features"]
        feat_nav[features/navigation/]
        feat_notes[features/notes/]
        feat_team[features/team/]
    end

    subgraph Git["Git Layer"]
        git_client[git/client.py]
        git_ops[git/worktree.py]
    end

    subgraph MCP["MCP Layer"]
        mcp_server[mcp/server.py]
        mcp_tools[mcp/tools/]
    end

    subgraph Providers["Provider Layer"]
        anthropic[providers/anthropic.py]
        openai[providers/openai.py]
        factory[providers/factory.py]
    end

    subgraph Net["Network Layer"]
        net_web[net/web.py]
        net_search[net/search.py]
    end

    %% CLI Layer connections
    main --> commands
    commands --> config
    commands --> console
    commands --> Agent
    commands --> git_client

    %% Agent to Providers
    agent_engine --> factory
    factory --> anthropic
    factory --> openai

    %% Security connections
    agent_exec --> security
    agent_exec --> command_val

    %% Memory connections
    Agent --> Memory
    memory_store --> memory_embed
    memory_retrieve --> memory_store

    %% MCP connections
    mcp_server --> mcp_registry
    mcp_registry --> mcp_tools
    mcp_tools --> security
    mcp_tools --> git_client

    %% Evolution connections
    evolution_orch --> Agent
    evolution_pr --> git_client

    %% Analysis connections
    commands --> Analysis

    style CLI fill:#e1f5fe
    style Core fill:#fff3e0
    style Git fill:#e8f5e9
    style MCP fill:#f3e5f5
    style Providers fill:#fce4ec
    style Analysis fill:#e0f2f1
    style Features fill:#fce4ec
    style Net fill:#e8eaf6
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| **main.py** | CLI entry point, dynamic command discovery |
| **commands/** | CLI command implementations (nav, agent, evolve, etc.) |
| **agent/** | Agent engine, middleware, parsing, execution loop |
| **memory/** | Persistent memory storage, embeddings, retrieval |
| **governance/** | AST checker, secret scanner, constitutional compliance |
| **core/** | Shared infrastructure (config, security, runtime, errors) |
| **core/security/** | Path validation, command validation, rate limiting, safety |
| **core/evolution/** | Code evolution orchestrator, PR generation |
| **core/sys/** | System operations (process, package, network, audio) |
| **git/** | Git operations via subprocess |
| **mcp/** | Model Context Protocol server and tools |
| **providers/** | LLM provider implementations |
| **swarm/** | Parallel execution with git worktree isolation |
| **ai/** | Token utilities, AI-specific helpers |
| **analysis/** | Static code analysis, complexity metrics, AST extraction |
| **capabilities/** | Capability registry for feature flags and tool availability |
| **features/** | Domain-specific features (navigation, notes, team) |
| **net/** | Network utilities (web scraping, search integration) |
| **structures/** | Data structures (DAG, etc.) |
| **ui/** | UI layer (agent UI components) |
| **templates/** | Template management and rendering |

---

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI (main.py)
    participant Agent as Agent Executor
    participant Engine as Agent Engine
    participant Provider as LLM Provider
    participant Tools as Tool Executor
    participant Memory as Memory Store

    User->>CLI: jp agent run "fix bug"
    CLI->>Agent: execute(task)

    %% Context gathering
    Agent->>Memory: query_memory(task)
    Memory-->>Agent: relevant memories

    Agent->>Agent: build_context()
    Agent->>Engine: run_agent_loop()

    loop Until task complete or max iterations
        Engine->>Provider: complete(messages)
        Provider-->>Engine: response

        Engine->>Engine: parse_response()
        Engine->>Engine: enforce_governance()

        alt Has tool calls
            Engine->>Tools: execute_tool(name, args)
            Tools->>Tools: validate_command()
            Tools-->>Engine: tool_result
            Engine->>Memory: save_memory(observation)
        end

        alt Task complete
            Engine-->>Agent: final_response
        else Continue
            Engine->>Engine: append_to_messages()
        end
    end

    Agent->>Memory: save_memory(outcome)
    Agent-->>CLI: result
    CLI-->>User: display_output()
```

---

## Request Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        A[User Input]
    end

    subgraph Processing
        B[Command Parser]
        C[Context Builder]
        D[Token Budget Manager]
        E[Agent Engine]
    end

    subgraph LLM
        F[Provider Factory]
        G[Anthropic/OpenAI]
    end

    subgraph Safety
        H[Governance Checker]
        I[Command Validator]
        J[Path Validator]
    end

    subgraph Output
        K[Response Parser]
        L[Tool Executor]
        M[Memory Store]
        N[User Output]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> K
    K --> H
    H --> I
    I --> J
    J --> L
    L --> M
    M --> E
    K --> N

    style Input fill:#e3f2fd
    style Processing fill:#fff8e1
    style LLM fill:#f3e5f5
    style Safety fill:#ffebee
    style Output fill:#e8f5e9
```

---

## Component Responsibilities

### CLI Layer
- **main.py**: Entry point, dynamic command discovery, help generation
- **commands/**: Individual CLI commands using Typer

### Agent Subsystem (`agent/`)
- **engine.py**: AgentEngine class, main orchestration
- **execution.py**: Main agent loop, repair strategies
- **middleware.py**: Middleware pipeline (tracing, governance, circuit breaker)
- **parsing.py**: LLM response parsing, JSON extraction
- **circuit.py**: Circuit breaker, cost velocity tracking
- **governance.py**: Governance enforcement for agent responses
- **prompting.py**: Prompt construction, context assembly
- **archive.py**: Agent execution archive/history
- **context.py**: Execution context management
- **factory.py**: Agent factory, provider selection
- **models.py**: Type definitions for agent responses
- **ops.py**: Core operations (step, observe, act)
- **patching.py**: Code patching utilities
- **single_shot.py**: Single-shot agent execution
- **strategies.py**: Repair and recovery strategies
- **tools.py**: Agent tool definitions
- **tracing.py**: Request tracing and observability

### Memory Subsystem (`memory/`)
- **api.py**: Memory API interface
- **models.py**: Data models (MemoryEntry, SearchResult)
- **embedding.py**: Sentence transformer embeddings
- **retrieval.py**: Semantic search, clustering
- **patterns.py**: Pattern extraction and synthesis
- **store/**: Storage backend package with pluggable implementations
  - **jsonl.py**: JSONL file-based storage with keyword search
  - **lance.py**: LanceDB vector store with semantic search
  - **hybrid.py**: Combined storage with RRF fusion

### Governance Subsystem (`governance/`)
- **ast_checker.py**: Constitutional AI checks, AST analysis
- **secret_scanner.py**: API key detection in code
- **compliance.py**: High-level compliance scanning
- **config.py**: Safety rules configuration loader
- **diff_parser.py**: Diff analysis for pre-commit checks
- **types.py**: Violation type definitions

### Core (`core/`)
- **config.py**: Application configuration
- **console.py**: Logging and output utilities
- **runtime.py**: Context variables, circuit breaker state
- **result.py**: Result[T, E] type for error handling
- **errors.py**: Exception types (ConfigurationError, NetworkError)
- **context.py**: Execution context management
- **context_gatherer.py**: Gathers context for agent execution
- **cost_tracker.py**: Token and cost tracking
- **mcp_registry.py**: MCP server registry
- **error_middleware.py**: Error handling middleware
- **merge_resolver.py**: Git merge conflict resolution
- **command_validation.py**: CLI command validation
- **decorators.py**: Common decorators
- **diagnostics.py**: Diagnostic utilities
- **registry.py**: Tool and command registry
- **replay.py**: Execution replay
- **serializer.py**: Object serialization
- **templates.py**: Template rendering
- **rate_limit.py**: Token bucket rate limiter (legacy location)
- **safety.py**: Safe shell execution policies (legacy location)
- **security/**: Security package (consolidated)
  - **path.py**: Path traversal prevention, atomic operations
  - **command.py**: Shell command allowlist/blocklist
  - **safety.py**: Safe shell execution policies
  - **rate_limit.py**: Token bucket rate limiter
- **sys/**: System operations package
  - **execution.py**: Process execution
  - **package.py**: Package management
  - **network.py**: Network diagnostics
  - **process.py**: Process management
  - **audio.py**: Audio operations
- **evolution/**: Code evolution package
  - **orchestrator.py**: Evolution orchestration
  - **pr.py**: Pull request generation
  - **prompting.py**: Evolution prompting
  - **types.py**: Type definitions
  - **verification.py**: Solution verification

### Analysis Subsystem (`analysis/`)
- **complexity.py**: Code complexity metrics
- **skeleton.py**: AST skeleton extraction
- **dependency_walker.py**: Dependency graph traversal
- **structure.py**: Code structure analysis
- **cache.py**: Analysis result caching

### Features (`features/`)
- **navigation/**: Navigation feature service
- **notes/**: Notes feature service
- **team/**: Team feature models

### Network (`net/`)
- **web.py**: Web scraping and HTTP utilities
- **search.py**: Search engine integration

### Capabilities (`capabilities/`)
- **registry.py**: Capability registry implementation

### Structures (`structures/`)
- **dag.py**: Directed acyclic graph implementation

### UI (`ui/`)
- **agent_ui.py**: Agent UI components

### Provider Layer
- **factory.py**: Provider selection based on model, frozen ProviderConfig
- **anthropic.py**: Claude API integration
- **openai.py**: OpenAI/Azure integration

---

## Key Design Patterns

### Result[T, E] Pattern
Error handling uses explicit Result types for recoverable errors:
```python
def load_config() -> Result[AppConfig, ConfigError]:
    ...
```

### Protocol-Based Abstractions
Core interfaces use Protocol types for flexibility:
```python
class MemoryStore(Protocol):
    def search(self, query: str, limit: int) -> list[MemoryEntry]: ...
```

### Lazy Loading
Heavy dependencies are loaded only when needed:
```python
def get_embedding_client():
    from sentence_transformers import SentenceTransformer  # Lazy
    ...
```

### Context Variables
Runtime state uses context variables for thread-safety:
```python
workspace_var: ContextVar[Path] = ContextVar("workspace")
```

---

## Security Architecture

```mermaid
flowchart TB
    subgraph Input["User Input"]
        A[CLI Command]
        B[Agent Task]
        C[MCP Tool Call]
    end

    subgraph Validation["Validation Layer"]
        D[Command Validator]
        E[Path Validator]
        F[Governance Checker]
    end

    subgraph Execution["Safe Execution"]
        G[Shell Executor]
        H[File Operations]
        I[Git Operations]
    end

    A --> D
    B --> F
    C --> E

    D -->|Allowed| G
    D -->|Blocked| X1[Reject]

    E -->|In Workspace| H
    E -->|Traversal| X2[Reject]

    F -->|Compliant| I
    F -->|Violation| X3[Reject]

    G --> |No shell=True| Safe1[Safe]
    H --> |Atomic + O_NOFOLLOW| Safe2[Safe]
    I --> |Hooks Disabled| Safe3[Safe]

    style Validation fill:#ffebee
    style Execution fill:#e8f5e9
```

### Security Controls

1. **Command Validation**: Allowlist/blocklist for shell commands
2. **Path Validation**: All paths checked against workspace root
3. **Governance Checking**: AST analysis for code safety
4. **Atomic Operations**: TOCTOU-safe file operations with O_NOFOLLOW
5. **Hook Disabling**: Git hooks disabled during automated operations
6. **Rate Limiting**: MCP tools have per-minute limits
7. **Secret Detection**: Pattern matching for API keys in code

---

## See Also

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
- [HANDBOOK.md](../HANDBOOK.md) - Agent protocol reference
- [AGENTS.md](../AGENTS.md) - Agent personas
