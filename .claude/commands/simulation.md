---
name: simulation
description: Create, validate, run, and analyze multi-agent simulations
arguments:
  - name: subcommand
    description: "Action: create, validate, run, analyze, experiment"
    required: true
  - name: target
    description: "Scenario file path or experiment config"
    required: false
---

# Simulation Skill

You help users work with the APART multi-agent simulation framework.

## Subcommands

### `/simulation create [domain]`
Interactive scenario creation. Guide the user through:

1. **Domain selection** - What type of scenario? (geopolitical, economic, social, custom)
2. **Agents** - Who are the actors? What are their objectives and constraints?
3. **Variables** - What state needs tracking? (global and per-agent)
4. **Modules** - Which behavior modules to enable?
5. **Timeline** - How many steps? What's the time scale?
6. **Events** - Any scripted events to inject?

Always validate the scenario before saving.

### `/simulation validate <path>`
Validate a scenario configuration:

1. Load the YAML file
2. Check required sections (engine, agents)
3. Validate variable definitions
4. Check module compatibility (granularity, conflicts)
5. Validate agent prompts have OBJECTIVES section
6. Report issues with suggestions

Run: `uv run python -c "from utils.scenario_validator import ScenarioValidator; ScenarioValidator().validate_file('<path>')"`

### `/simulation run <path>`
Execute a simulation:

1. Validate the scenario first
2. Run: `uv run python -m main --config <path> --scenario <name>`
3. Monitor output for errors
4. Report run directory location when complete

### `/simulation analyze <run_dir>`
Analyze simulation results:

1. Read final state from `<run_dir>/final_state.json`
2. Read step snapshots from `<run_dir>/snapshots/`
3. Summarize:
   - Variable trajectories over time
   - Key turning points
   - Agent behavior patterns
4. Generate insights about the simulation outcome

### `/simulation experiment <config>`
Run multi-condition experiments:

```python
from experiment import ExperimentRunner, ExperimentConfig, ExperimentCondition

config = ExperimentConfig(
    name="experiment_name",
    description="What we're testing",
    scenario_path="scenarios/base.yaml",
    conditions=[
        ExperimentCondition("baseline", modifications={}),
        ExperimentCondition("variant", modifications={"agent_vars.x.default": 10}),
    ],
    runs_per_condition=3,
)
runner = ExperimentRunner(config)
result = runner.run_all()
```

---

## Scenario Structure

```yaml
# Required
max_steps: 10
engine:
  provider: gemini|openai|anthropic|ollama
  model: model-name
  system_prompt: "Engine instructions..."
  simulation_plan: "What happens in this simulation..."

# Required - at least one agent
agents:
  - name: "Agent Name"
    system_prompt: |
      OBJECTIVES: What this agent wants
      CONSTRAINTS: What limits the agent
      INFORMATION ACCESS: What the agent knows
    llm:
      provider: gemini
      model: gemini-1.5-flash
    variables:
      var_name: value  # Override defaults

# Optional
global_vars:
  var_name:
    type: int|float|bool
    default: value
    min: value
    max: value

agent_vars:
  var_name:
    type: int|float|bool
    default: value

modules:
  - agents_base
  - economic_base
  - diplomatic_base

module_config:
  territory_graph:
    map_file: path/to/map.yaml
```

## Available Modules

| Module | Layer | Domain | Purpose |
|--------|-------|--------|---------|
| `agents_base` | domain | - | Core agent communication |
| `territory_graph` | grounding | - | Spatial structure |
| `economic_base` | domain | economic | Economic variables |
| `diplomatic_base` | domain | diplomatic | Alliances, treaties |
| `trust_dynamics` | domain | social | Trust relationships |
| `supply_chain_base` | detail | economic | Supply chains (extends economic_base) |

See `/simulation-schema` for full module documentation.

## Agent Prompt Best Practices

Always include these sections in agent system prompts:

```
OBJECTIVES:
- Primary goal
- Secondary goals

CONSTRAINTS:
- What the agent cannot do
- Resource limitations

INFORMATION ACCESS:
- What the agent knows
- What they don't know
```

## Quick Start Examples

**Minimal scenario:**
```yaml
max_steps: 5
engine:
  provider: gemini
  model: gemini-1.5-flash
  system_prompt: "You simulate a negotiation."
  simulation_plan: "Two parties negotiate a deal."

agents:
  - name: "Buyer"
    system_prompt: "You want to buy at the lowest price."
  - name: "Seller"
    system_prompt: "You want to sell at the highest price."
```

**With modules:**
```yaml
max_steps: 10
modules:
  - agents_base
  - economic_base
  - trust_dynamics

engine:
  provider: openai
  model: gpt-4o
  system_prompt: "Simulate economic cooperation."
  simulation_plan: "Agents trade resources over 10 rounds."

agents:
  - name: "Country A"
    system_prompt: |
      OBJECTIVES: Maximize trade surplus
      CONSTRAINTS: Limited resources
      INFORMATION ACCESS: Know your own resources, estimate others
    variables:
      trust_level: 60
      resource_stockpile: 100
```

See `/simulation-templates` for more complete examples.
