# Architecture

This document describes the architecture of the apart multi-agent simulation framework.

## Overview

Apart is a modular multi-agent simulation framework that separates concerns between coordination (Orchestrator), game logic (GameEngine), agent behavior (Agent), and state management (GameState/AgentState).

## Core Components

### SimulatorAgent (NEW in v2.0)

**Responsibility:** LLM-powered simulation orchestration

- Interprets agent responses into realistic state changes
- Generates personalized agent messages
- Creates emergent and scripted events
- Enforces constraints through clamping with feedback
- Maintains sliding history window for context

**Key Methods:**
- `initialize_simulation(agent_names)`: Generate initial agent messages
- `process_step(step_number, agent_responses)`: Process step and return next messages

### 1. Orchestrator (`src/orchestrator.py`)

**Responsibility:** Coordination and flow control

- Loads configuration from YAML
- Initializes agents and game engine
- Manages the main simulation loop
- Coordinates message passing between agents and game engine

**Key Methods:**
- `run()`: Main simulation loop

### 2. GameEngine (`src/game_engine.py`)

**Responsibility:** Game logic and state management

- Validates configuration on startup
- Manages global game state
- Generates context-aware messages for agents
- Processes agent responses
- Handles game progression (rounds, events)

**Key Methods:**
- `get_message_for_agent(agent_name)`: Generate message based on game state
- `process_agent_response(agent_name, response)`: Update state based on agent action
- `advance_round()`: Progress to next round

### 3. Agent (`src/agent.py`)

**Responsibility:** Agent behavior

Currently simple response generation. Designed to be extended for:
- LLM integration
- Decision-making logic
- Action selection

**Key Methods:**
- `respond(message)`: Generate response to orchestrator message

### 4. State Models (`src/state.py`)

**Responsibility:** State representation and validation

#### GameState
- Global simulation state
- Agent collection
- Event history
- Global variables (typed with constraints)

#### AgentState
- Per-agent state
- Response history
- Agent-specific variables (typed with constraints)

Both use Pydantic for validation and type safety.

### 5. Variable System (`src/variables.py`)

**Responsibility:** Typed state variables with validation

#### VariableDefinition
- Type specification (int, float, bool)
- Constraints (min/max for numeric types)
- Default values

#### VariableSet
- Collection of variables with definitions
- Runtime validation on get/set operations
- Ensures type safety and constraint enforcement

### 6. Configuration Parser (`src/config_parser.py`)

**Responsibility:** YAML parsing and validation

- Parses variable definitions from YAML
- Validates agent and global variable configurations
- Supports per-agent variable overrides
- Comprehensive error checking with clear messages

## Data Flow

```
YAML Config
    ↓
Orchestrator.load_config()
    ↓
GameEngine.__init__()
    ├→ Validates config
    ├→ Initializes global variables
    └→ Creates GameState
    ↓
Orchestrator.run() [main loop]
    ↓
GameEngine.get_message_for_agent()
    ├→ Updates events
    └→ Returns context-aware message
    ↓
Agent.respond()
    └→ Generates response
    ↓
GameEngine.process_agent_response()
    ├→ Creates AgentState (if first interaction)
    │   └→ Initializes agent variables (with overrides)
    └→ Updates agent state
    ↓
GameEngine.advance_round()
    └→ Increments round counter
```

## State Management

### Global State
- Managed by `GameState` instance in `GameEngine`
- Includes: rounds, events, agents, global variables
- Persists throughout simulation

### Agent State
- Each agent has `AgentState` instance
- Created on first interaction with agent
- Includes: responses, status, agent-specific variables
- Stored in `GameState.agents` dictionary

### Variable State
- Both global and agent-level variables
- Defined in YAML with types and constraints
- Validated on initialization and updates
- Type-safe operations via Pydantic models

## Configuration System

### Structure
```yaml
# Simulation parameters
max_steps: 5
orchestrator_message: "..."

# Initial game state
game_state:
  initial_resources: 150
  difficulty: "normal"

# Global variables (simulation-wide)
global_vars:
  var_name:
    type: float
    default: 0.05
    min: 0.0
    max: 1.0

# Agent variables (per-agent defaults)
agent_vars:
  var_name:
    type: int
    default: 100
    min: 0

# Agent definitions
agents:
  - name: "Agent Alpha"
    response_template: "..."
    variables:  # Optional overrides
      var_name: 150
```

### Validation
- Happens at `GameEngine` initialization
- Checks variable definitions (type, default, constraints)
- Validates agent variable overrides
- Fails fast with descriptive errors

## Testing Strategy

- **Unit tests**: Individual components (state, variables, config parser)
- **Integration tests**: GameEngine with full config
- **Validation tests**: Config parsing and error handling
- All tests use pytest framework
- CI/CD via GitHub Actions

## Module System

The module system provides composable behavior components that add domain-specific variables, dynamics, constraints, and agent effects.

### Four-Layer Architecture

```
┌─────────────────────────────────────────┐
│  META LAYER                             │
│  (Simulation-wide rules, win conditions)│
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  GROUNDING LAYER                        │
│  (territory_graph, time systems)        │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  DOMAIN LAYER                           │
│  (economic_base, diplomatic_base, etc.) │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  DETAIL LAYER                           │
│  (supply_chain_base extends economic)   │
└─────────────────────────────────────────┘
```

### Granularity Levels

Modules declare which actor granularities they support:

| Level | Actors | Example |
|-------|--------|---------|
| `macro` | Blocs, institutions | "The West", "BRICS", "UN" |
| `meso` | Nation-states | USA, China, Germany |
| `micro` | Factions, individuals | "Hardliners", "CEO of X" |

When combining modules, the system finds their common granularity.

### Key Components

**ModuleLoader** (`src/modules/loader.py`)
- Loads module definitions from YAML
- Validates dependencies and conflicts
- Finds common granularity across modules

**ModuleComposer** (`src/modules/composer.py`)
- Merges multiple modules into ComposedModules
- Combines variables, dynamics, constraints
- Loads spatial graphs and networks from config

**BehaviorModule** (`src/modules/models.py`)
- Data model for module definitions
- Contains: variables, dynamics, constraints, agent_effects
- Taxonomy: layer, domain, granularity_support, extends

### Module Configuration

```yaml
modules:
  - agents_base
  - economic_base

module_config:
  territory_graph:
    map_file: modules/maps/world.yaml
```

### Available Modules

| Module | Layer | Domain | Granularity |
|--------|-------|--------|-------------|
| `agents_base` | domain | - | macro/meso/micro |
| `territory_graph` | grounding | - | macro/meso/micro |
| `economic_base` | domain | economic | macro/meso |
| `diplomatic_base` | domain | diplomatic | macro/meso/micro |
| `trust_dynamics` | domain | social | macro/meso/micro |
| `supply_chain_base` | detail | economic | macro/meso |

## Experiment Runner

The experiment runner enables multi-condition simulations for comparative analysis.

### Components

**ExperimentConfig** (`src/experiment/models.py`)
- Defines experiment: name, scenario, conditions, runs_per_condition

**ExperimentCondition** (`src/experiment/models.py`)
- Named condition with dot-path modifications to base config

**ExperimentRunner** (`src/experiment/runner.py`)
- Executes all conditions and repetitions
- Uses Orchestrator internally for each run
- Aggregates results by condition

**Results** (`src/experiment/results.py`)
- Save/load experiment results
- Statistical comparison (mean, std, min, max)
- Human-readable summaries

### Data Flow

```
ExperimentConfig
    ↓
ExperimentRunner.run_all()
    ├→ For each condition:
    │   ├→ apply_modifications(base_config, condition.modifications)
    │   └→ For each repetition:
    │       ├→ Create Orchestrator with modified config
    │       ├→ orchestrator.run()
    │       └→ Collect RunResult
    └→ Return ExperimentResult
```

### Usage

```python
from experiment import ExperimentRunner, ExperimentConfig, ExperimentCondition

config = ExperimentConfig(
    name="trust_study",
    scenario_path="scenarios/cooperation.yaml",
    conditions=[
        ExperimentCondition("low", modifications={"agent_vars.trust.default": 20}),
        ExperimentCondition("high", modifications={"agent_vars.trust.default": 80}),
    ],
    runs_per_condition=3,
)

result = ExperimentRunner(config).run_all()
```

## Scenario Validation

The validator (`src/utils/scenario_validator.py`) checks:

1. Required sections (engine, agents)
2. Variable type matching
3. Module compatibility (granularity, conflicts)
4. Agent prompt structure (OBJECTIVES section)
5. Config completeness for modules

## Design Principles

1. **Separation of Concerns**: Each component has single responsibility
2. **Type Safety**: Pydantic models for validation
3. **Configuration-Driven**: Behavior defined in YAML, not code
4. **Fail Fast**: Validation at startup prevents runtime errors
5. **Extensibility**: Clear extension points for custom logic
6. **Composability**: Modules combine without conflicts
7. **Reproducibility**: Experiments capture all configuration
