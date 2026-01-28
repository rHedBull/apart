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

**IMPORTANT:** Before writing ANY code, invoke the `superpowers:brainstorming` skill to explore the user's intent.

#### Phase 1: Brainstorming (REQUIRED)

Invoke the brainstorming skill first:
```
Use Skill tool: superpowers:brainstorming
```

During brainstorming, explore:
- What question or hypothesis does this simulation answer?
- What makes this simulation interesting or useful?
- What would success look like?

#### Phase 2: Iterative Interview

Use `AskUserQuestion` to gather details in stages. Do NOT proceed until each stage is complete.

**Stage 1: Domain & Purpose**
```
AskUserQuestion:
  question: "What domain is this simulation exploring?"
  options:
    - label: "Geopolitical"
      description: "International relations, conflicts, diplomacy"
    - label: "Economic"
      description: "Markets, trade, resource allocation"
    - label: "Social/Organizational"
      description: "Group dynamics, trust, cooperation"
    - label: "AI Safety"
      description: "AI governance, alignment, risks"
```

Then ask: "In 1-2 sentences, what scenario do you want to simulate?"

**Stage 2: Actors**
```
AskUserQuestion:
  question: "How many agents should participate?"
  options:
    - label: "2-3 agents"
      description: "Focused interaction, faster runs"
    - label: "4-6 agents"
      description: "Moderate complexity"
    - label: "7+ agents"
      description: "Complex multi-party dynamics"
```

For each agent, ask:
- "What is this agent's name and role?"
- "What is their primary objective?"
- "What constraints or limitations do they have?"
- "What information do they have access to?"

**Stage 3: Dynamics**
```
AskUserQuestion:
  question: "What variables should the simulation track?"
  multiSelect: true
  options:
    - label: "Trust/relationships"
      description: "How agents view each other"
    - label: "Resources/economy"
      description: "Wealth, assets, trade"
    - label: "Power/influence"
      description: "Relative strength, leverage"
    - label: "Custom variables"
      description: "Define your own metrics"
```

If custom: "What custom variables do you need? (name, type, range)"

**Stage 4: Timeline & Events**
```
AskUserQuestion:
  question: "How long should the simulation run?"
  options:
    - label: "Short (3-5 steps)"
      description: "Quick test, focused scenario"
    - label: "Medium (6-12 steps)"
      description: "Allows dynamics to develop"
    - label: "Long (13-30 steps)"
      description: "Full arc, complex evolution"
```

Ask: "What time period does each step represent? (hour/day/week/month/year)"

Ask: "Should any events be scripted to occur at specific steps? (optional)"

**Stage 5: Technical Config**
```
AskUserQuestion:
  question: "Which LLM provider should run this simulation?"
  options:
    - label: "Ollama (local)"
      description: "Free, private, requires local setup. Use deepseek-coder-v2:latest"
    - label: "Gemini"
      description: "Fast, good JSON. Requires GEMINI_API_KEY"
    - label: "OpenAI"
      description: "GPT-4o. Requires OPENAI_API_KEY"
    - label: "Anthropic"
      description: "Claude. Requires ANTHROPIC_API_KEY"
```

#### Phase 3: Generate & Validate

Only after ALL stages complete:

1. Generate the scenario YAML based on gathered information
2. Write to `scenarios/<name>.yaml`
3. Run validation:
   ```python
   from utils.scenario_validator import ScenarioValidator
   result = ScenarioValidator().validate(config)
   ```
4. Show validation results
5. Ask user to confirm or iterate

#### Checklist Before Generating

- [ ] Domain and purpose clearly defined
- [ ] All agents have: name, role, objectives, constraints, information access
- [ ] Variables defined with types and ranges
- [ ] Modules selected based on domain
- [ ] Timeline and time scale set
- [ ] LLM provider chosen
- [ ] Any scripted events defined

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
