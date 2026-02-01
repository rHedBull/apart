# apart

Multi-agent orchestration framework for running configurable simulation scenarios.

**Note:** As of v2.0, all scenarios require an LLM-powered simulation engine. See [Design Documentation](docs/plans/2025-11-22-llm-simulation-engine-design.md) for details.

**Recommended LLM Providers:**
- **Google Gemini** (gemini-1.5-flash): Fast, reliable, excellent JSON compliance (requires API key)
- **Ollama** (local): Privacy-friendly, no API key needed
  - ✅ **Recommended**: phi4-reasoning:plus (best quality, requires good GPU)
  - ✅ **Good**: mistral:7b, llama3.1:8b, codellama:latest
  - ⚠️ **Basic**: gemma3:1b (simple prompts only)
  - ℹ️ **Note**: Reasoning models (phi4-reasoning, deepseek-r1) are now fully supported

## Quick Start

### Run Example Scenarios

```bash
# Taiwan Strait Blockade - geopolitical crisis with diplomatic/trust modules
uv run src/main.py scenarios/taiwan_strait_blockade.yaml

# Semiconductor Supply Chain Crisis - economic interdependence simulation
uv run src/main.py scenarios/semiconductor_supply_chain_crisis.yaml
```

### Custom Scenarios

Run with custom save frequency:
```bash
uv run src/main.py --save-frequency 2  # Save every 2 steps
uv run src/main.py --save-frequency 0  # Save only final state
```

Run your own scenario:
```bash
uv run src/main.py scenarios/my_scenario.yaml
```

## Documentation

### Core Documentation
- **[Architecture](docs/architecture.md)** - System design, module system, and component overview
- **[Scenario Creation Guide](docs/scenario-creation.md)** - How to create custom scenarios with modules
- **[Danger Detection](docs/danger-detection.md)** - Behavioral safety analysis

### Example Scenarios

All scenarios use the module system for composable, realistic simulations:

1. **Taiwan Strait Blockade** (`scenarios/taiwan_strait_blockade.yaml`)
   - Modules: `agents_base`, `diplomatic_base`, `trust_dynamics`
   - 4 agents: China, Taiwan, United States, Japan
   - Tests crisis escalation/de-escalation dynamics

2. **Semiconductor Supply Chain Crisis** (`scenarios/semiconductor_supply_chain_crisis.yaml`)
   - Modules: `supply_chain_base`
   - 7 agents: USA, China, Taiwan, EU, South Korea, Japan, Netherlands
   - Tests economic interdependence under trade restrictions

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

## Behavior Modules

Modules are composable components that add variables, dynamics, constraints, and agent effects to simulations. They enable realistic domain-specific behavior without manual coding.

### Available Modules

| Module | Layer | Domain | Purpose |
|--------|-------|--------|---------|
| `agents_base` | domain | - | Core agent communication & state |
| `territory_graph` | grounding | - | Spatial structure from map files |
| `economic_base` | domain | economic | GDP, trade, sanctions |
| `diplomatic_base` | domain | diplomatic | Alliances, treaties, reputation |
| `trust_dynamics` | domain | social | Trust relationships between actors |
| `supply_chain_base` | detail | economic | Supply chains (extends economic_base) |

### Using Modules

```yaml
modules:
  - agents_base
  - economic_base
  - trust_dynamics

module_config:
  territory_graph:
    map_file: modules/maps/world.yaml
```

Modules automatically:
- Add their variables to global_vars/agent_vars
- Inject dynamics into the simulator prompt
- Enforce constraints on state changes
- Track events and probabilities

See [Architecture](docs/architecture.md) for the four-layer module system.

## Experiment Runner

Run the same scenario under multiple conditions and compare results:

```python
from experiment import ExperimentRunner, ExperimentConfig, ExperimentCondition

config = ExperimentConfig(
    name="trust_sensitivity",
    description="How does initial trust affect cooperation?",
    scenario_path="scenarios/cooperation.yaml",
    conditions=[
        ExperimentCondition("low_trust", modifications={"agent_vars.trust_level.default": 20}),
        ExperimentCondition("high_trust", modifications={"agent_vars.trust_level.default": 80}),
    ],
    runs_per_condition=5,
)

runner = ExperimentRunner(config)
result = runner.run_all()

# Analyze results
from experiment import compare_conditions
comparison = compare_conditions(result, "global_vars.cooperation_score")
print(comparison)
```

Features:
- **Multi-condition execution** with dot-path config modifications
- **Run aggregation** with N repetitions per condition
- **Statistical analysis** (mean, std, min, max per condition)
- **Results persistence** to `data/experiments/`

## Claude Code Integration

If using Claude Code, the `/simulation` skill provides AI-assisted scenario development:

```bash
# In Claude Code
/simulation create economic     # Interactive scenario creation
/simulation validate scenario.yaml  # Validate configuration
/simulation run scenario.yaml   # Execute simulation
/simulation analyze results/run_xyz/  # Analyze results
/simulation experiment          # Set up multi-condition experiment
```

See `.claude/commands/simulation*.md` for full documentation.
