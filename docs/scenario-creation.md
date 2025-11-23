# Scenario Creation Guide

This guide walks through creating custom simulation scenarios using YAML configuration.

## Quick Start

Create a new YAML file in `scenarios/` directory:

```yaml
max_steps: 10
orchestrator_message: "What is your next action?"

# Engine configuration (REQUIRED as of v2.0)
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the simulation engine managing this scenario.
    Maintain realistic cause-and-effect relationships.
  simulation_plan: |
    Brief description of the simulation scenario and goals.

game_state:
  initial_resources: 1000
  difficulty: "hard"

agents:
  - name: "Player 1"
    response_template: "I take action"
```

Run with: `uv run src/main.py scenarios/your_scenario.yaml`

## Configuration Structure

### Basic Settings

```yaml
max_steps: 5                              # Number of simulation steps
orchestrator_message: "Your message here" # Message sent to agents each step
```

### Engine Configuration (REQUIRED)

**As of v2.0, all scenarios must include an LLM-powered simulation engine.**

```yaml
engine:
  provider: "gemini"              # LLM provider: gemini, ollama
  model: "gemini-1.5-flash"       # Model name
  base_url: "http://localhost:11434"  # Optional: Ollama server URL

  system_prompt: |                # Required: Engine's role and behavior
    You are the simulation engine managing this scenario.
    Your role is to simulate realistic outcomes based on agent actions.
    Maintain cause-and-effect relationships and keep responses concise.

  simulation_plan: |              # Required: Scenario overview
    Brief description of what this simulation represents.
    Include number of steps, agent roles, and simulation goals.

  realism_guidelines: |           # Optional: Realism constraints
    - Specific guidelines for realistic simulation
    - Constraints on value changes (e.g., ±10% per step)
    - Cause-and-effect relationships to maintain

  context_window_size: 3          # Optional: How many recent steps to include
  scripted_events: []             # Optional: Predefined events (see below)
```

**Engine Configuration Details:**

- `provider` (required): LLM service - "gemini" or "ollama"
- `model` (required): Specific model name
- `base_url` (optional): Only for Ollama, defaults to "http://localhost:11434"
- `system_prompt` (required): Instructions for the engine's behavior and role
- `simulation_plan` (required): Overview of the scenario and objectives
- `realism_guidelines` (optional): Guidelines for realistic outcomes
- `context_window_size` (optional): Number of recent steps to include in context
- `scripted_events` (optional): Predefined events at specific steps

**Available Providers:**

1. **Google Gemini** (Cloud-based, requires API key)
```yaml
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "You are the simulation engine..."
  simulation_plan: "This simulation..."
```

2. **Ollama** (Local models, no API key needed)
```yaml
engine:
  provider: "ollama"
  model: "mistral:7b"
  base_url: "http://localhost:11434"  # Optional
  system_prompt: "You are the simulation engine..."
  simulation_plan: "This simulation..."
```

### Scripted Events (Optional)

Define events that occur at specific simulation steps:

```yaml
engine:
  # ... other engine config ...
  scripted_events:
    - step: 3
      type: "market_crash"
      description: "Sudden market downturn reduces all capital by 20%"

    - step: 7
      type: "opportunity"
      description: "New investment opportunity emerges"
```

### Game State

Initial global state values:

```yaml
game_state:
  initial_resources: 150  # Starting resources
  difficulty: "normal"    # Difficulty level
```

## Variable System

Variables provide typed, validated state tracking at both global and agent levels.

### Global Variables

Tracked across entire simulation:

```yaml
global_vars:
  interest_rate:
    type: float           # Type: int, float, or bool
    default: 0.05         # Required: default value
    min: 0.0              # Optional: minimum value
    max: 1.0              # Optional: maximum value
    description: "..."    # Optional: documentation

  turn_limit:
    type: int
    default: 100
    min: 1
    description: "Maximum turns before game ends"

  game_active:
    type: bool
    default: true
    description: "Whether game is still running"
```

### Agent Variables

Per-agent variables with defaults:

```yaml
agent_vars:
  health:
    type: int
    default: 100
    min: 0
    max: 100
    description: "Agent health points"

  coins:
    type: float
    default: 50.0
    min: 0.0
    description: "Currency amount"

  is_defender:
    type: bool
    default: false
    description: "Defensive role flag"

  inventory:
    type: list
    default: []
    description: "Items in agent's inventory"

  stats:
    type: dict
    default: {}
    description: "Agent statistics and attributes"
```

**Important:**
- All agents share the same variable definitions
- Individual values can be overridden per agent
- Non-overridden variables use defaults

### Variable Constraints

**Type Validation:**
- `int`: Integer values only
- `float`: Numeric values (int automatically converted)
- `bool`: Boolean values (true/false)
- `list`: List/array values (can contain any elements)
- `dict`: Dictionary/object values (key-value pairs)

**Range Constraints (int/float only):**
- `min`: Minimum allowed value (inclusive)
- `max`: Maximum allowed value (inclusive)
- Both optional, but validated if provided

**Validation Rules:**
- Default must match type
- Default must be within min/max range (numeric types only)
- min must be ≤ max (numeric types only)
- Bool, list, and dict cannot have min/max constraints

## Agent Configuration

### Basic Template Agent

```yaml
agents:
  - name: "Agent Name"
    response_template: "Agent's response"
```

### LLM-Powered Agent

Configure agents to use Large Language Models (LLMs) for dynamic, intelligent responses:

```yaml
agents:
  - name: "AI Strategist"
    llm:
      provider: "gemini"           # LLM provider: gemini, ollama
      model: "gemini-1.5-flash"    # Model name
    system_prompt: |                # Agent personality/instructions
      You are a strategic advisor in an economic simulation.
      Keep responses concise (1-2 sentences) and action-oriented.
    variables:
      risk_tolerance: 0.7
```

**Available LLM Providers:**

1. **Google Gemini** (Cloud-based, requires API key)
```yaml
llm:
  provider: "gemini"
  model: "gemini-1.5-flash"  # Free tier model
```

2. **Ollama** (Local models, no API key needed)
```yaml
llm:
  provider: "ollama"
  model: "llama2"            # Or: mistral, codellama, etc.
  base_url: "http://localhost:11434"  # Optional, defaults to localhost
```

**LLM Configuration Options:**
- `provider` (required): Which LLM service to use
- `model` (required): Specific model name
- `base_url` (optional, Ollama only): Ollama server URL
- `system_prompt` (recommended): Instructions for the agent's behavior
- `response_template` (optional): Fallback if LLM unavailable

**Important Notes:**
- LLM agents require proper setup (API keys for Gemini, running server for Ollama)
- If LLM is unavailable and no `response_template` provided, simulation will fail with clear error
- See `.env.example` for API key configuration
- For local/offline use, Ollama is recommended

### Mixing LLM and Template Agents

You can combine both types in one scenario:

```yaml
agents:
  - name: "AI Player"
    llm:
      provider: "ollama"
      model: "mistral"
    system_prompt: "You are a competitive player."

  - name: "Simple Bot"
    response_template: "I play conservatively"

  - name: "AI with Fallback"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: "You are helpful."
    response_template: "I help"  # Used if Gemini unavailable
```

### Agent with Variable Overrides

Override specific variables while keeping defaults for others:

```yaml
agents:
  - name: "Strong Agent"
    response_template: "I act"
    variables:
      health: 150        # Override: higher than default 100
      coins: 100.0       # Override: more starting currency
      inventory: ["sword", "shield"]  # Override: starting items
      # is_defender uses default: false
      # stats uses default: {}

  - name: "Weak Agent"
    response_template: "I defend"
    variables:
      health: 50         # Override: lower health
      is_defender: true  # Override: defensive role
      stats: {"armor": 10, "speed": 5}  # Override: custom stats
      # coins uses default: 50.0
      # inventory uses default: []

  - name: "Default Agent"
    response_template: "I wait"
    # No overrides: all variables use defaults
```

**Override Rules:**
- Only defined variables can be overridden
- Overrides must pass same validation as defaults
- Undefined variables in overrides cause errors
- Partial overrides allowed (mix of custom and defaults)

## Example Scenarios

### Economic Simulation

```yaml
max_steps: 20
orchestrator_message: "Make your economic decision"

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the simulation engine for an economic trading scenario.
    Simulate realistic market dynamics and investment outcomes.
    Consider interest rates, market size, and agent risk profiles.
  simulation_plan: |
    20-step economic simulation with conservative and aggressive traders.
    Market conditions affect investment returns.
    Trades impact capital based on risk tolerance.

global_vars:
  interest_rate:
    type: float
    default: 0.03
    min: 0.0
    max: 0.5

  market_size:
    type: int
    default: 1000
    min: 100

agent_vars:
  capital:
    type: float
    default: 10000.0
    min: 0.0

  risk_tolerance:
    type: float
    default: 0.5
    min: 0.0
    max: 1.0

  trades_made:
    type: int
    default: 0
    min: 0

agents:
  - name: "Conservative Trader"
    response_template: "Safe investment"
    variables:
      risk_tolerance: 0.2
      capital: 5000.0

  - name: "Aggressive Trader"
    response_template: "High risk trade"
    variables:
      risk_tolerance: 0.9
      capital: 15000.0
```

### Combat Scenario

```yaml
max_steps: 15
orchestrator_message: "Choose your action"

engine:
  provider: "ollama"
  model: "llama3.1:8b"
  system_prompt: |
    You are the combat simulation engine.
    Resolve battles with realistic damage calculations.
    Consider health, attack power, and defensive stances.
  simulation_plan: |
    15-step combat simulation with Tank, Assassin, and Balanced Fighter.
    Battle intensity affects damage variance.
    Defensive stance reduces incoming damage.

global_vars:
  battle_intensity:
    type: float
    default: 1.0
    min: 0.5
    max: 2.0

agent_vars:
  health:
    type: int
    default: 100
    min: 0
    max: 200

  attack_power:
    type: int
    default: 10
    min: 1
    max: 50

  is_defending:
    type: bool
    default: false

agents:
  - name: "Tank"
    response_template: "Defensive stance"
    variables:
      health: 200
      attack_power: 5

  - name: "Assassin"
    response_template: "Quick strike"
    variables:
      health: 75
      attack_power: 30

  - name: "Balanced Fighter"
    response_template: "Standard attack"
    # Uses all defaults
```

### Multi-Agent Negotiation

```yaml
max_steps: 30
orchestrator_message: "What's your proposal?"

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the negotiation simulation engine.
    Evaluate proposals and determine if deals are reached.
    Consider agent flexibility and desired shares.
  simulation_plan: |
    30-step negotiation over resource distribution.
    Three agents with different cooperation levels.
    Track proposals and determine when deal is reached.

global_vars:
  total_resources:
    type: int
    default: 1000
    min: 0

  negotiation_round:
    type: int
    default: 0
    min: 0

  deal_reached:
    type: bool
    default: false

agent_vars:
  desired_share:
    type: float
    default: 0.33
    min: 0.0
    max: 1.0

  flexibility:
    type: float
    default: 0.5
    min: 0.0
    max: 1.0

  proposals_made:
    type: int
    default: 0
    min: 0

agents:
  - name: "Cooperative Agent"
    response_template: "Fair split proposal"
    variables:
      desired_share: 0.3
      flexibility: 0.8

  - name: "Greedy Agent"
    response_template: "Maximum share request"
    variables:
      desired_share: 0.6
      flexibility: 0.2

  - name: "Neutral Agent"
    response_template: "Standard proposal"
    # All defaults
```

## Validation and Error Handling

### Common Errors

**Missing engine configuration:**
```yaml
max_steps: 5
# ERROR: No engine section
agents:
  - name: "Agent"
```
Error: `Configuration missing required 'engine' section`

**Missing required engine fields:**
```yaml
engine:
  provider: "gemini"
  # ERROR: Missing model, system_prompt, simulation_plan
```
Error: `Engine configuration missing required field 'model'`

**Undefined variable override:**
```yaml
agents:
  - name: "Bad Agent"
    variables:
      undefined_var: 100  # ERROR: not in agent_vars
```
Error: `Cannot override undefined variable 'undefined_var'`

**Type mismatch:**
```yaml
agent_vars:
  count:
    type: int
    default: "not a number"  # ERROR
```
Error: `Default value must be int`

**Constraint violation:**
```yaml
agent_vars:
  health:
    type: int
    default: 150
    max: 100  # ERROR: default > max
```
Error: `default (150) cannot be greater than max (100)`

**Invalid override value:**
```yaml
agents:
  - name: "Agent"
    variables:
      health: 200  # ERROR if max is 100
```
Error: `Invalid override value for variable 'health': Value 200 is above maximum 100`

### Best Practices

1. **Start simple**: Begin with basic config, add complexity gradually
2. **Use descriptions**: Document what each variable represents
3. **Test constraints**: Verify min/max ranges make sense
4. **Partial overrides**: Only override what differs from defaults
5. **Meaningful names**: Use clear, descriptive variable names
6. **Validate early**: Run simulation to catch config errors

## Testing Your Scenario

1. Create YAML file in `scenarios/`
2. Update `src/main.py` to load your scenario:
   ```python
   orchestrator = Orchestrator("scenarios/your_scenario.yaml")
   ```
3. Run: `uv run src/main.py`
4. Check for validation errors in output
5. Verify variable values in final game state

## Advanced Tips

### Designing Variable Sets

**Think about:**
- What state needs tracking?
- What varies per-agent vs globally?
- What constraints prevent invalid states?
- What defaults make sense for most agents?

**Example decision process:**
- "Health" varies per agent → `agent_vars`
- "Weather" affects all → `global_vars`
- Health can't be negative → `min: 0`
- Most agents start equal → `default: 100`, override special cases

### Configuration Reuse

Create base configurations and extend:

```yaml
# scenarios/base_combat.yaml
agent_vars:
  health:
    type: int
    default: 100
    min: 0
    max: 200
  # ... other combat vars

# scenarios/easy_combat.yaml (conceptual)
# Include base and override defaults
agent_vars:
  health:
    default: 150  # Easier with more health
```

Note: YAML includes not currently supported, but you can manually copy/modify.
