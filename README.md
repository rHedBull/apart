# apart

Multi-agent orchestration framework for running configurable simulation scenarios.

## Quick Start

Run a simulation:
```bash
uv run src/main.py
```

## Documentation

- **[Architecture](docs/architecture.md)** - System design and component overview
- **[Scenario Creation Guide](docs/scenario-creation.md)** - How to create custom scenarios

## Configuration

Edit `scenarios/config.yaml` to customize:
- Number of simulation steps
- Orchestrator messages
- Agent behaviors and responses
- Global and per-agent variables with type validation
