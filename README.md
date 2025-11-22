# apart

Multi-agent orchestration framework for running configurable simulation scenarios.

**Note:** As of v2.0, all scenarios require an LLM-powered simulation engine. See [Design Documentation](docs/plans/2025-11-22-llm-simulation-engine-design.md) for details.

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

### Output Files

**`state.json`** - Simulation snapshots:
- Game state (resources, difficulty, round)
- Global and per-agent variable values
- All orchestrator-agent messages

**`simulation.jsonl`** - Structured logs:
- Human-readable console output with colors
- Machine-readable JSONL for analysis
- Predefined message codes (SIM001, AGT002, etc.)
- Performance metrics and timing data

Control save frequency with `--save-frequency N`:
- `N=0`: Save only final state
- `N=1`: Save after every step (default)
- `N=k`: Save every k steps

### Analyzing Logs

Filter by message code:
```bash
# All agent responses
grep "AGT003" results/run_*/simulation.jsonl | jq .

# Persistence operations
grep "PER00" results/run_*/simulation.jsonl | jq .

# Performance metrics
grep "PRF001" results/run_*/simulation.jsonl | jq .
```
