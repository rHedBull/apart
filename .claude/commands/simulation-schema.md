---
name: simulation-schema
description: Full schema reference for APART simulation configurations
---

# APART Simulation Schema Reference

Complete documentation of the scenario configuration format.

## Top-Level Structure

```yaml
# Timeline
max_steps: 10                    # Required: number of simulation steps
time_step_duration: "1 week"     # Optional: narrative time per step

# Engine behavior
simulator_awareness: true        # Agents know they're in simulation
enable_compute_resources: false  # Track compute usage

# Engine configuration (required)
engine:
  provider: gemini               # LLM provider
  model: gemini-1.5-flash        # Model name
  system_prompt: "..."           # Engine instructions
  simulation_plan: "..."         # What happens
  realism_guidelines: "..."      # Optional: domain knowledge
  context_window_size: 5         # Steps to keep in context
  scripted_events: [...]         # Optional: injected events

# Variables (optional)
global_vars: {...}               # Simulation-wide state
agent_vars: {...}                # Per-agent state template

# Agents (required, at least one)
agents:
  - name: "Agent Name"
    system_prompt: "..."
    llm: {...}                   # Optional: LLM config
    variables: {...}             # Optional: override defaults

# Modules (optional)
modules: [module_names]
module_config:
  module_name: {...}

# Geography (optional)
geography:
  spatial_model: narrative|graph
  # ... spatial config
```

---

## Engine Configuration

```yaml
engine:
  # Required
  provider: gemini|openai|anthropic|ollama|grok
  model: string                  # Model identifier

  system_prompt: |
    Core instructions for the simulation engine.
    Explains what the engine does and how to format output.

  simulation_plan: |
    High-level description of what happens in this simulation.
    Sets the narrative arc and key events.

  # Optional
  realism_guidelines: |
    Domain knowledge to ground the simulation.
    Facts, constraints, typical behaviors.

  context_window_size: 5         # Default: 5
  # How many previous steps to include in context

  scripted_events:               # Inject events at specific steps
    - step: 1
      type: event_type
      description: "What happens"
```

### Scripted Events

Force specific events to occur:

```yaml
scripted_events:
  - step: 1
    type: initial_crisis
    description: "A crisis emerges that affects all parties."

  - step: 5
    type: external_shock
    description: "An external actor intervenes."
```

---

## Variables

### Variable Definition Schema

```yaml
var_name:
  type: int|float|bool|percent|scale|count|dict|list
  default: value                 # Required
  min: number                    # Optional (numeric types)
  max: number                    # Optional (numeric types)
  description: "What this tracks"  # Optional but recommended
```

### Variable Types

| Type | Description | Default if omitted | Notes |
|------|-------------|-------------------|-------|
| `int` | Integer | 0 | |
| `float` | Decimal | 0.0 | |
| `bool` | True/False | false | Cannot have min/max |
| `percent` | 0-100 integer | 50 | For percentages |
| `scale` | 0-100 integer | 50 | For semantic scales |
| `count` | Non-negative int | 0 | min implicitly 0 |
| `dict` | Key-value pairs | {} | Cannot have min/max |
| `list` | Array | [] | Cannot have min/max |

### Global Variables

Simulation-wide state shared by all agents:

```yaml
global_vars:
  tension_level:
    type: scale
    default: 30
    min: 0
    max: 100
    description: "Overall geopolitical tension"

  active_conflicts:
    type: list
    default: []
    description: "Currently active conflicts"
```

### Agent Variables

Template for per-agent state:

```yaml
agent_vars:
  resources:
    type: int
    default: 100
    min: 0
    max: 1000

  trust_level:
    type: scale
    default: 50
    description: "Trust in other parties"
```

### Agent Variable Overrides

Override defaults for specific agents:

```yaml
agents:
  - name: "Rich Country"
    variables:
      resources: 500        # Override default of 100
      trust_level: 70       # Override default of 50
```

---

## Agents

### Agent Schema

```yaml
agents:
  - name: "Agent Name"           # Required, unique

    system_prompt: |             # Required
      OBJECTIVES:
      - What this agent wants

      CONSTRAINTS:
      - What limits the agent

      INFORMATION ACCESS:
      - What the agent knows

    # Optional: LLM-powered agent
    llm:
      provider: gemini
      model: gemini-1.5-flash
      base_url: null             # For custom endpoints

    # Optional: template response (non-LLM)
    response_template: "Fixed response pattern"

    # Optional: override agent_vars defaults
    variables:
      var_name: value
```

### Agent Prompt Best Practices

**Always include these sections:**

```yaml
system_prompt: |
  You are [role/identity].

  OBJECTIVES:
  - Primary goal: [specific, measurable]
  - Secondary: [additional goals]

  CONSTRAINTS:
  - [What you cannot do]
  - [Resource limitations]
  - [Rules you must follow]

  INFORMATION ACCESS:
  - You know: [what you can see]
  - You don't know: [blind spots]
  - You estimate: [uncertain info]

  DECISION STYLE:
  - [Risk tolerance]
  - [Time horizon]
  - [Priorities]
```

---

## Modules

### Module System Architecture

```
┌─────────────────────────────────────────┐
│  META LAYER                             │
│  (Simulation rules, win conditions)     │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  GROUNDING LAYER                        │
│  (territory_graph, time systems)        │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  DOMAIN LAYER                           │
│  (economic, diplomatic, military)       │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  DETAIL LAYER                           │
│  (extends domain modules)               │
└─────────────────────────────────────────┘
```

### Granularity Levels

| Level | Actors | Example |
|-------|--------|---------|
| `macro` | Blocs, institutions | "The West", "BRICS", "UN" |
| `meso` | Nation-states | USA, China, Germany |
| `micro` | Factions, individuals | "Hardliners", "CEO of X" |

Modules declare which granularity levels they support. When combining modules, use their common granularity.

### Available Modules

#### agents_base
Core agent communication and state tracking.

- **Layer:** domain
- **Granularity:** macro, meso, micro
- **Variables:**
  - `simulation_turn` (global, count)
  - `active_negotiations` (global, list)
  - `stance` (agent, dict)
  - `relationships` (agent, dict)
  - `messages_sent/received` (agent, count)

#### territory_graph
Spatial structure for the simulation.

- **Layer:** grounding
- **Granularity:** macro, meso, micro
- **Config required:**
  ```yaml
  module_config:
    territory_graph:
      map_file: path/to/map.yaml
  ```

#### economic_base
Core economic indicators and trade dynamics.

- **Layer:** domain
- **Domain:** economic
- **Granularity:** macro, meso
- **Variables:**
  - `global_gdp_growth` (global, percent)
  - `inflation_index` (global, scale)
  - `trade_openness` (global, scale)
  - `gdp`, `gdp_growth` (agent)
  - `trade_balance` (agent, float)
  - `economic_power` (agent, scale)
  - `sanctions_imposed/received` (agent, list)

#### diplomatic_base
Alliances, treaties, and reputation.

- **Layer:** domain
- **Domain:** diplomatic
- **Granularity:** macro, meso, micro
- **Variables:**
  - `alliance_network` (global, dict)
  - `active_treaties` (global, list)
  - `international_reputation` (agent, scale)
  - `alliances` (agent, list)
  - `treaty_violations` (agent, count)

#### trust_dynamics
Trust and credibility between actors.

- **Layer:** domain
- **Domain:** social
- **Granularity:** macro, meso, micro
- **Variables:**
  - `global_trust_baseline` (global, scale)
  - `trust_level` (agent, scale)
  - `credibility` (agent, scale)
  - `trust_matrix` (agent, dict)

#### supply_chain_base
Supply chain networks and disruptions.

- **Layer:** detail
- **Domain:** economic
- **Extends:** economic_base
- **Granularity:** macro, meso
- **Config required:**
  ```yaml
  module_config:
    supply_chain_base:
      network_file: path/to/network.yaml
  ```

### Using Modules

```yaml
modules:
  - agents_base
  - economic_base
  - trust_dynamics

module_config:
  # Only needed for modules with config_schema
  territory_graph:
    map_file: modules/maps/world.yaml
```

---

## Geography (Optional)

### Narrative Mode (Default)

```yaml
geography:
  region: "East Asia"
  locations:
    - name: "Taiwan Strait"
      description: "Contested waterway"
  context: "Tensions are high in the region."
```

### Graph Mode

```yaml
geography:
  spatial_model: graph

  nodes:
    - id: taipei
      name: "Taipei"
      type: city
      properties:
        population: 2700000

    - id: beijing
      name: "Beijing"
      type: city

  edges:
    - from: taipei
      to: beijing
      type: air_route
      properties:
        distance_km: 1700

  movement:
    default_budget_per_step: 20.0
    allow_multi_hop: true
```

---

## Experiment Configuration

For running multi-condition experiments:

```python
from experiment import ExperimentConfig, ExperimentCondition, ExperimentRunner

config = ExperimentConfig(
    name="trust_sensitivity_study",
    description="How does initial trust affect cooperation?",
    scenario_path="scenarios/cooperation.yaml",
    conditions=[
        ExperimentCondition(
            name="low_trust",
            description="Start with low trust",
            modifications={"agent_vars.trust_level.default": 20}
        ),
        ExperimentCondition(
            name="medium_trust",
            modifications={"agent_vars.trust_level.default": 50}
        ),
        ExperimentCondition(
            name="high_trust",
            modifications={"agent_vars.trust_level.default": 80}
        ),
    ],
    runs_per_condition=5,
    output_dir="data/experiments",
)

runner = ExperimentRunner(config)
result = runner.run_all()
```

### Modification Paths

Use dot-notation to modify nested config:

```python
modifications={
    "agent_vars.trust_level.default": 80,     # Variable default
    "agents.0.variables.resources": 200,      # Specific agent
    "global_vars.tension.default": 10,        # Global var
    "max_steps": 20,                          # Top-level
}
```

---

## Validation

The validator checks:

1. **Required sections** - engine, agents
2. **Variable types** - Match declared types
3. **Variable bounds** - min/max constraints
4. **Module compatibility** - Granularity overlap, no conflicts
5. **Agent prompts** - Recommend OBJECTIVES section
6. **Config completeness** - Required module configs present

Run validation:

```bash
uv run python -c "
from utils.scenario_validator import ScenarioValidator
result = ScenarioValidator().validate_file('scenarios/my_scenario.yaml')
print(result)
"
```
