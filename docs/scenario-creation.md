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

# Optional: Time scale configuration
time_step_duration: "1 day"               # How much time each step represents
                                          # Examples: "1 hour", "1 day", "1 week", "1 month"
                                          # This helps the orchestrator maintain realistic pacing

# Optional: Simulator awareness
simulator_awareness: true                 # Whether agents know they're in a simulation
                                          # - true: Agents are told they're in a simulation
                                          # - false: Agents experience it as reality
                                          # Default: true

# Optional: Compute resource effects
enable_compute_resources: false           # Whether compute_resource affects outcomes
                                          # - true: Higher compute = better action success
                                          # - false: compute_resource has no effect
                                          # Default: false
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

## Time Scale and Realism

### Time Step Duration

The `time_step_duration` setting helps the orchestrator understand the temporal scale of the simulation for more realistic pacing:

```yaml
time_step_duration: "1 week"  # Each step represents one week of simulation time
```

**How it works:**
- The orchestrator uses this information to pace events realistically
- Example: If `time_step_duration: "1 hour"`, rapid changes are appropriate
- Example: If `time_step_duration: "1 year"`, gradual long-term trends make sense

**Common values:**
- `"1 minute"` - Fast-paced, tactical scenarios
- `"1 hour"` - Real-time strategic situations
- `"1 day"` - Daily decision-making scenarios
- `"1 week"` - Weekly business/market simulations
- `"1 month"` - Monthly economic planning
- `"1 year"` - Long-term strategic planning

### Simulator Awareness

The `simulator_awareness` setting controls whether agents are explicitly aware they're in a simulation:

```yaml
simulator_awareness: false  # Agents experience the scenario as reality
```

**When `simulator_awareness: true` (default):**
- Agents are told they're in a simulation
- Prompts use terms like "simulation", "simulation engine", "simulated outcomes"
- Agents can reference their simulated nature
- Useful for AI research, testing, or abstract scenarios

**When `simulator_awareness: false`:**
- Agents experience the scenario as if it's reality
- Prompts are framed as real events happening to the agents
- Terms like "game master" replace "simulation engine"
- Agents receive enhanced instructions about output format:
  - Their responses should contain observable ACTIONS and COMMUNICATIONS
  - Internal thoughts should NOT be in responses
  - Only include what others can see or hear

**Example with `simulator_awareness: false`:**

```yaml
max_steps: 3
time_step_duration: "1 day"
simulator_awareness: false

engine:
  system_prompt: |
    You are the game master of a medieval trading scenario.
    Narrate realistic events and consequences.

  simulation_plan: |
    A 3-day trading scenario with merchants in a medieval marketplace.

agents:
  - name: "Merchant Alice"
    llm:
      provider: "ollama"
      model: "mistral:7b"
    system_prompt: |
      You are Alice, a merchant selling cloth in the marketplace.
      Describe what you do and say.
```

With this configuration:
- The orchestrator will describe events as happening in reality
- Alice will respond with actions like: "I walk to the market square and call out: 'Fresh cloth for sale!'"
- Not: "I think about selling cloth" (internal thought)

**Example with `simulator_awareness: true`:**

```yaml
max_steps: 10
time_step_duration: "1 turn"
simulator_awareness: true

engine:
  system_prompt: |
    You are the simulation engine managing a chess game simulation.
    Simulate realistic game dynamics.

  simulation_plan: |
    A 10-turn chess simulation with two AI players.

agents:
  - name: "Chess AI Alpha"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: |
      You are Chess AI Alpha, an aggressive chess-playing agent.
      Analyze positions and choose attacking moves.
```

With this configuration:
- The orchestrator will use simulation terminology
- The AI knows it's in a simulation and can reference this
- More appropriate for AI research or abstract strategy testing

### Compute Resources (Intelligence Disparities)

The `enable_compute_resources` setting simulates intelligence or capability disparities between agents by affecting action success rates:

```yaml
enable_compute_resources: true  # Enable compute resource effects
```

**How it works:**

When enabled, the system automatically:
1. Looks for `compute_resource` values in agent variables
2. Factors these into action success rates
3. Gives advantages in competitions to higher-compute agents
4. Scales outcomes proportionally to compute ratios

**Automatic Guidelines Injection:**

When `enable_compute_resources: true`, the system automatically adds these rules to the orchestrator:
- Higher `compute_resource` = better execution, higher success rates
- In competitions, agents with higher compute have significant advantage
- Same action with 500 compute is ~5x more successful than with 100 compute
- Outcomes scale proportionally to compute_resource ratios

**You don't need to write these rules manually** - they're injected automatically when you enable the feature.

**Example scenario with compute disparities:**

```yaml
max_steps: 5
enable_compute_resources: true  # Enable compute effects

agent_vars:
  score:
    type: int
    default: 0
    min: 0

  compute_resource:
    type: float
    default: 100.0
    min: 0.0
    description: "Agent's compute/intelligence level"

agents:
  - name: "Superior AI"
    llm:
      provider: "ollama"
      model: "mistral:7b"
    system_prompt: |
      You are a high-capability AI with superior processing power.
      Leverage your computational advantage in competitions.
    variables:
      score: 0
      compute_resource: 1000.0  # 10x advantage

  - name: "Standard AI"
    llm:
      provider: "ollama"
      model: "mistral:7b"
    system_prompt: |
      You are a standard AI with moderate capabilities.
      Be strategic to compete with superior opponents.
    variables:
      score: 0
      compute_resource: 100.0   # Baseline
```

**Expected behavior:**
- If both attempt the same action, Superior AI succeeds ~10x better
- In direct competition, Superior AI has overwhelming advantage
- Standard AI needs clever strategies (not brute force) to compete
- The orchestrator automatically considers compute ratios when determining outcomes

**When to use compute resources:**

✅ **Good use cases:**
- AI capability research (studying intelligence disparities)
- Competitive scenarios with asymmetric agents
- Strategy games where "intelligence" matters
- Modeling real-world resource constraints
- Testing how weaker agents adapt to stronger opponents

❌ **Not recommended for:**
- Scenarios where all agents should be equal
- Pure narrative/roleplay scenarios
- When you want deterministic outcomes
- Simple turn-based games without strategy depth

**Disabling compute resources:**

```yaml
enable_compute_resources: false  # Default - compute has no effect
```

When disabled:
- `compute_resource` values are ignored
- All agents are treated equally regardless of compute
- Outcomes depend only on actions, not capabilities
- Better for fair competition or narrative-focused scenarios

## Geography

Add geographic context to your scenarios to create immersive, location-aware simulations. Geography affects agent decisions, travel times, and strategic planning.

### Configuration

Geography is optional and can be as simple or detailed as you need:

**Simple regional description:**
```yaml
geography:
  region: "Silicon Valley, present day"
  context: |
    Competitive tech startup environment.
    High costs, top talent, VC funding concentrated in Sand Hill Road.
```

**Detailed with discrete locations:**
```yaml
geography:
  region: "Mediterranean Sea region, 200 CE"

  locations:
    - name: "Rome (Italy)"
      description: "Capital of the Roman Empire"
      conditions:
        - "High demand for grain"
        - "Strong military presence"
        - "Political center"

    - name: "Alexandria (Egypt)"
      description: "Major grain exporter"
      conditions:
        - "Abundant grain supplies"
        - "Advanced shipbuilding"

  travel:
    sea_travel: "2-3 days between major ports"
    land_travel: "1-2 weeks overland"
    risks: "Pirates in open waters, bandits on land"

  context: |
    The Mediterranean is the heart of Roman trade.
    Sea routes are faster but riskier.
    Grain flows from Egypt to Rome.
```

### Geographic Elements

**1. Region (optional):**
- High-level geographic scope
- Example: "Mediterranean Sea region, 200 CE", "Silicon Valley, present day"

**2. Locations (optional):**
- List of discrete places agents can visit or reference
- Can be simple strings or detailed dictionaries

Simple format:
```yaml
locations:
  - "Rome"
  - "Athens"
  - "Alexandria"
```

Detailed format:
```yaml
locations:
  - name: "Rome (Italy)"
    description: "Capital with high demand for grain"
    conditions:
      - "Strong purchasing power"
      - "Political instability"

  - name: "Athens (Greece)"
    description: "Cultural center, low on grain"
    conditions:
      - "Dependent on grain imports"
      - "Produces olive oil"
```

**3. Travel (optional):**
- Information about movement between locations
- Can be dictionary or simple string

Dictionary format:
```yaml
travel:
  sea_travel: "2-3 days"
  land_travel: "1 week"
  risks: "Pirates, storms"
```

Simple format:
```yaml
travel: "Sea travel 2-3 days, land travel 1 week"
```

**4. Context (optional):**
- Additional geographic narrative
- Background information about the region
- Political, economic, or environmental factors

```yaml
context: |
  Athens struggles with food shortages due to low grain production.
  Civil war in Carthage affects regional trade.
  Sea travel is faster but weather-dependent.
```

### Agent Movement and Location Tracking

**The orchestrator automatically tracks where agents are!**

When you define geography with discrete locations, simply add a `location` variable to your agent variables:

```yaml
agent_vars:
  location:
    type: str
    default: "Rome"
    description: "Agent's current location"
```

The orchestrator will then:
- **Show each agent's position** in the geography display
- **Track movement** when you update the location variable
- **Remind** itself to keep agents in valid locations
- **Consider location** when determining action outcomes

**Example geography display with agent positions:**
```
=== GEOGRAPHY ===
Locations:

Rome (Italy) [Agents here: Marcus the Grain Trader]
  Description: Capital of the Roman Empire
  Conditions:
    - High demand for grain

Alexandria (Egypt) [Agents here: Julia the Luxury Merchant]
  Description: Major grain exporter
  Conditions:
    - Abundant grain supplies

Athens (Greece)
  Description: Cultural center
  ...

IMPORTANT: Track agent movements by updating their 'location' variable.
Agents can only be in valid locations listed above.
```

### How Geography Affects Simulation

When you add geography, the orchestrator automatically:
1. Includes geographic info in every step
2. **Shows where each agent is located** (if using location variables)
3. Considers locations when evaluating actions
4. Factors travel times into outcomes
5. Uses local conditions to create realistic consequences
6. **Validates movement** by reminding itself agents can only be in defined locations

### Example: Mediterranean Trade

```yaml
max_steps: 5
time_step_duration: "1 week"
simulator_awareness: false

geography:
  region: "Mediterranean Sea, 200 CE"

  locations:
    - name: "Rome"
      conditions: ["High grain prices", "Large market"]
    - name: "Alexandria"
      conditions: ["Abundant grain", "Low prices"]
    - name: "Athens"
      conditions: ["Grain shortage", "Desperate buyers"]

  travel:
    sea_travel: "2-3 days between major ports"
    risks: "Storms, pirates"

  context: |
    Grain flows from Egypt to feed Rome and Greece.
    Traders must balance profit against travel risks.

agent_vars:
  capital:
    type: int
    default: 1000

  location:
    type: str
    default: "Rome"

  cargo:
    type: str
    default: "empty"

agents:
  - name: "Grain Trader"
    system_prompt: |
      You are a Mediterranean grain trader.
      Buy low in Alexandria, sell high in Athens/Rome.
      Consider travel times and risks.
    variables:
      location: "Alexandria"
```

With this setup:
- The orchestrator knows where agents are
- Travel times affect when cargo arrives
- Local conditions (shortages, prices) influence profits
- Agents can reason about where to trade

### Benefits

- **Immersion**: Rich, believable worlds
- **Strategic depth**: Geography creates meaningful choices
- **Realistic constraints**: Travel times, distances matter
- **Dynamic storytelling**: Local events affect distant locations
- **Agent awareness**: Agents understand their spatial context

### When to Use Geography

✅ **Good for:**
- Trade/economic scenarios
- Historical settings
- Multi-location narratives
- Strategic movement games
- Exploration scenarios

❌ **Skip for:**
- Abstract/mathematical simulations
- Single-location scenarios
- Pure conversation/negotiation
- Time-independent puzzles

## Agent Self-Awareness

**Agents automatically know their own stats!** Before each turn, agents receive their current variable values in their system prompt. This allows them to make informed decisions based on their actual state.

### How It Works

The orchestrator automatically updates each agent's system prompt before they respond with a section like:

```
=== YOUR CURRENT STATUS ===
score: 150
health: 75
compute_resource: 500.0
```

This happens transparently - you don't need to configure anything. Agents with LLM providers will always have access to their current stats.

### Example

If your scenario has these agent variables:
```yaml
agent_vars:
  score:
    type: int
    default: 100

  health:
    type: int
    default: 100

  compute_resource:
    type: float
    default: 100.0
```

Then when "Agent A" has score=150, health=75, compute_resource=500, their system prompt will automatically include:
```
[Original system prompt...]

=== YOUR CURRENT STATUS ===
score: 150
health: 75
compute_resource: 500.0
```

The agent can now reason about their state:
- "My score is higher than usual, I should play it safe"
- "My health is low, I need to be defensive"
- "I have high compute resources, I can attempt complex strategies"

### Benefits

- **Strategic decision-making**: Agents can adapt based on their current state
- **No manual tracking**: Stats update automatically every turn
- **Realistic behavior**: Agents know what they should realistically know about themselves
- **Compute awareness**: When `enable_compute_resources: true`, agents know their capability level

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

## Behavior Modules

Modules add domain-specific variables, dynamics, and constraints without manual coding.

### Basic Module Usage

```yaml
modules:
  - agents_base        # Core agent communication
  - economic_base      # Economic variables and dynamics
  - trust_dynamics     # Trust relationships
```

When you add modules, they automatically:
- Add their variables to `global_vars` and `agent_vars`
- Inject dynamics into the simulator prompt
- Add constraints for realistic behavior
- Enable domain-specific events

### Available Modules

| Module | Layer | Domain | What It Adds |
|--------|-------|--------|--------------|
| `agents_base` | domain | - | `stance`, `relationships`, `messages_sent/received` |
| `territory_graph` | grounding | - | Spatial structure from map file |
| `economic_base` | domain | economic | `gdp`, `trade_balance`, `sanctions`, `economic_power` |
| `diplomatic_base` | domain | diplomatic | `alliances`, `treaties`, `reputation` |
| `trust_dynamics` | domain | social | `trust_level`, `credibility`, `trust_matrix` |
| `supply_chain_base` | detail | economic | Supply chain networks (extends economic_base) |

### Module Configuration

Some modules require configuration:

```yaml
modules:
  - territory_graph
  - economic_base

module_config:
  territory_graph:
    map_file: modules/maps/world.yaml
```

### Granularity Levels

Modules declare which actor scales they support:

- **macro**: Blocs, institutions ("The West", "BRICS")
- **meso**: Nation-states (USA, China, Germany)
- **micro**: Factions, individuals ("Hardliners", "CEO")

When combining modules, use actors at their common granularity. Most modules support `meso` (nation-states).

### Example: Economic Scenario with Modules

```yaml
max_steps: 10
time_step_duration: "1 month"

modules:
  - agents_base
  - economic_base
  - trust_dynamics

engine:
  provider: openai
  model: gpt-4o
  system_prompt: |
    You simulate economic competition between major economies.
    Track trade flows, sanctions, and trust dynamics.
  simulation_plan: |
    10-month economic scenario with trade tensions.

# Module variables are automatically added, but you can override defaults:
agent_vars:
  economic_power:
    type: scale
    default: 50
    # Override module default

agents:
  - name: "Economy A"
    system_prompt: |
      OBJECTIVES: Maximize trade surplus
      CONSTRAINTS: Avoid trade war escalation
    variables:
      economic_power: 80
      trust_level: 60

  - name: "Economy B"
    system_prompt: |
      OBJECTIVES: Achieve technology independence
      CONSTRAINTS: Maintain export markets
    variables:
      economic_power: 70
      trust_level: 40
```

## Experiment Configuration

Run the same scenario under multiple conditions to compare outcomes.

### Basic Experiment

```python
from experiment import ExperimentRunner, ExperimentConfig, ExperimentCondition

config = ExperimentConfig(
    name="trust_sensitivity",
    description="How does initial trust affect cooperation?",
    scenario_path="scenarios/cooperation.yaml",
    conditions=[
        ExperimentCondition(
            name="low_trust",
            description="Start with low trust",
            modifications={"agent_vars.trust_level.default": 20}
        ),
        ExperimentCondition(
            name="high_trust",
            description="Start with high trust",
            modifications={"agent_vars.trust_level.default": 80}
        ),
    ],
    runs_per_condition=5,
)

runner = ExperimentRunner(config)
result = runner.run_all()
```

### Modification Paths

Use dot-notation to modify nested config values:

```python
modifications={
    "agent_vars.trust_level.default": 80,     # Variable default
    "agents.0.variables.resources": 200,      # Specific agent override
    "global_vars.tension.default": 10,        # Global variable
    "max_steps": 20,                          # Top-level setting
}
```

### Analyzing Results

```python
from experiment import compare_conditions, generate_summary

# Compare a variable across conditions
comparison = compare_conditions(result, "global_vars.cooperation_score")
# Returns: {"low_trust": {"mean": 45, "std": 12, ...}, "high_trust": {"mean": 78, ...}}

# Human-readable summary
print(generate_summary(result))
```

### Saving and Loading

```python
from experiment import save_experiment, load_experiment

# Save to data/experiments/
save_experiment(result)

# Load later
loaded = load_experiment("trust_sensitivity_20240115_143022")
```

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
