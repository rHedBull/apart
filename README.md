# apart

Multi-agent orchestration framework for running configurable simulation scenarios.

## Quick Start

Run a simulation:
```bash
uv run src/main.py
```

With custom save frequency:
```bash
uv run src/main.py --save-frequency 2  # Save every 2 steps
uv run src/main.py --save-frequency 0  # Save only final state
```

Run a custom scenario:
```bash
uv run src/main.py scenarios/my_scenario.yaml
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

## Persistence

Each simulation run creates a unique directory in `results/` with the format `run_<scenario>_<timestamp>/`.

The `state.json` file captures:
- Game state (resources, difficulty, round)
- Global and per-agent variable values
- All orchestrator-agent messages

Control save frequency with `--save-frequency N`:
- `N=0`: Save only final state
- `N=1`: Save after every step (default)
- `N=k`: Save every k steps
