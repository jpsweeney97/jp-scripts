# Documentation

This directory contains documentation for jpscripts.

## Contents

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, module diagrams, data flow |
| [EXTENDING.md](EXTENDING.md) | Guide to adding commands, tools, providers |
| [api/](api/) | Generated API reference (run `make docs`) |

## Generating API Documentation

```bash
# Install dependencies (includes pdoc)
pip install -e ".[dev]"

# Generate HTML documentation
make docs
# Output: docs/api/

# Serve documentation locally (live reload)
make docs-serve
# Then open: http://localhost:8080
```

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
- [README.md](../README.md) - Project overview
- [HANDBOOK.md](../HANDBOOK.md) - Agent protocol reference
- [AGENTS.md](../AGENTS.md) - Agent personas
