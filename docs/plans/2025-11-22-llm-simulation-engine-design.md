# LLM-Powered Simulation Engine Design

**Date:** 2025-11-22
**Status:** Design Complete - Ready for Implementation

## Overview

Add an LLM-powered simulation engine (SimulatorAgent) that acts as an intelligent game master, managing world state evolution, interpreting agent actions, and generating dynamic narratives.

## Motivation

Current system has agents responding to static orchestrator messages with limited world dynamics. This design introduces an LLM-powered engine that:

- Maintains simulation realism through intelligent state management
- Interprets agent responses and applies realistic consequences
- Generates personalized narratives for each agent
- Creates emergent events based on agent actions
- Supports scripted events with flexible execution
- Prevents agents from modifying their own stats (engine is authoritative)

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│  (Coordinates simulation flow)                          │
└────┬──────────────────────┬──────────────────────┬─────┘
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ GameEngine  │◄───│ SimulatorAgent   │    │   Agents    │
│             │    │                  │    │ (Players)   │
│ State Store │    │ LLM-Powered      │    │             │
│ Variables   │    │ Game Master      │    │ Respond to  │
│ Validation  │    │ - Interprets     │    │ Messages    │
│             │    │ - Updates State  │    │             │
│             │    │ - Generates Msgs │    │             │
└─────────────┘    └──────────────────┘    └─────────────┘
```

### Responsibilities

**SimulatorAgent (NEW)**
- LLM-powered intelligent orchestrator
- Processes agent responses from previous step
- Updates world state realistically
- Generates personalized agent messages
- Manages events and scripted triggers
- Enforces constraints through clamping

**GameEngine (SIMPLIFIED)**
- Pure state management (no game logic)
- Variable storage and validation
- State queries (get/set operations)
- Constraint definitions

**EngineValidator (NEW)**
- Validates LLM responses (structure, types, references)
- Enforces min/max constraints
- Retry logic with error feedback
- Fail-fast on critical errors

**Agents (UNCHANGED)**
- Respond to messages (LLM or template-based)
- Cannot modify their own stats
- Express intent through natural language

## Execution Flow

### Step 0: Initialization

```
SimulatorAgent.initialize_simulation(agent_names)
  → Returns: {"Agent A": "Initial message...", "Agent B": "Initial message..."}
```

### Step N (N ≥ 1)

```
1. Agents respond to messages from step N-1
   Agent A: "I invest 200k in startups"
   Agent B: "I build military defenses"

2. SimulatorAgent.process_step(step_number, agent_responses)

   a) Build context:
      - Simulation setup (system prompt, plan, guidelines)
      - Upcoming scripted events
      - Current complete state (all variables)
      - Last N steps (only changed variables + narratives)
      - Agent responses from step N-1

   b) Call LLM (single call per step):
      Prompt → LLM → JSON response

   c) Validate response:
      - JSON structure
      - References (agents, variables exist)
      - Types (int/float/bool/list/dict)
      - Constraints (min/max)
      → If invalid: retry with error feedback (max 2-3 attempts)
      → If max retries: FAIL SIMULATION

   d) Apply constraint clamping:
      - If value < min: clamp to min, log hit
      - If value > max: clamp to max, log hit
      - Feed constraint hits to next step's context

   e) Update GameEngine state:
      - Apply global variable updates
      - Apply agent variable updates
      - Store events

   f) Record step in history:
      - Only changed variables (delta reporting)
      - Events generated
      - Agent responses
      - Reasoning
      - Constraint hits

   g) Return personalized agent messages

3. Repeat at step N+1
```

## Data Flow

### SimulatorAgent Context (LLM Prompt)

```
=== SIMULATION SETUP ===
{system_prompt}

Simulation Plan:
{simulation_plan}

Realism Guidelines:
{realism_guidelines}

=== UPCOMING SCRIPTED EVENTS ===
Step 20: major_war - A great war must begin. Decide how it manifests.
Step 35: natural_disaster - Major earthquake strikes.

=== CURRENT STATE (Step N) ===
Global Variables:
  geopolitical_tension: 0.65
  market_volatility: 0.42

Agent Variables:
  Agent A:
    economic_strength: 1450.0
    military_power: 85
  Agent B:
    economic_strength: 1100.0
    military_power: 60

=== RECENT HISTORY (Last N steps) ===

Step N-2:
  Changes:
    Agent A: military_power +10
    Global: geopolitical_tension +0.15
  Event: Border skirmish erupted
  Agent Responses:
    Agent A: "I mobilize troops to defend our interests"
    Agent B: "I condemn this aggression and call for sanctions"
  Reasoning: Agent A's expansion triggered regional alarm

Step N-1:
  Changes:
    Agent A: economic_strength -200
    Agent B: public_support +0.10
  Event: International sanctions imposed on Agent A
  Agent Responses:
    Agent A: "I invest 300k in domestic production to counter sanctions"
    Agent B: "I strengthen alliances with neighboring states"
  Reasoning: Sanctions hurt aggressor, defender gains diplomatic support
  Constraint Hit: Agent A economic_strength attempted -250, clamped to -200

=== YOUR TASK ===
Process agent responses from Step N-1.
Update world state realistically for Step N.
Generate personalized messages for each agent.

Return JSON format:
{
  "state_updates": {
    "global_vars": {"var_name": new_value, ...},
    "agent_vars": {
      "Agent A": {"var_name": new_value, ...},
      "Agent B": {...}
    }
  },
  "events": [
    {
      "type": "event_type",
      "description": "What happened",
      "affects": ["Agent A", "Agent B"],
      "duration": 3
    }
  ],
  "agent_messages": {
    "Agent A": "Your personalized narrative with context and optional suggestions",
    "Agent B": "Different narrative based on their situation"
  },
  "reasoning": "Why you made these updates and generated these events"
}

IMPORTANT: Only include variables that CHANGED. Omit unchanged variables.
```

### SimulatorAgent Response Example

```json
{
  "state_updates": {
    "global_vars": {
      "geopolitical_tension": 0.80
    },
    "agent_vars": {
      "Agent A": {
        "economic_strength": 1250.0,
        "industrial_capacity": 450
      },
      "Agent B": {
        "economic_strength": 1150.0,
        "public_support": 0.65
      }
    }
  },
  "events": [
    {
      "type": "economic_sanctions",
      "description": "International community imposes severe economic sanctions on Agent A",
      "affects": ["Agent A", "Agent B"],
      "duration": 5
    }
  ],
  "agent_messages": {
    "Agent A": "Your 300k investment in domestic production partially offsets sanctions (-200 economy) but increases industrial capacity (+50). International isolation grows. Options: 1) De-escalate to lift sanctions 2) Double down on self-reliance 3) Seek new allies",
    "Agent B": "Your alliance-building pays off (+50 economy, +0.05 support). You're seen as the defender. The international community backs you with sanctions against Agent A. How do you leverage this position?"
  },
  "reasoning": "Agent A's investment in domestic production shows economic resilience, reducing the sanction impact from -250 to -200 and boosting industrial capacity. Agent B's diplomatic efforts gain international support and economic benefits. Geopolitical tension rises as the conflict escalates."
}
```

## Configuration

### Scenario YAML Format

```yaml
max_steps: 50

engine:                          # REQUIRED
  provider: "gemini"             # LLM provider: gemini, ollama
  model: "gemini-1.5-flash"      # Model name

  system_prompt: |               # Required: Engine's role
    You are the game master for a geopolitical simulation.
    Maintain realistic cause-and-effect relationships.
    Gradual changes are more realistic than sudden shifts.

  simulation_plan: |             # Required: High-level narrative intent
    Simulate emerging geopolitical tensions around steps 10-15.
    Economic volatility throughout the simulation.
    Respond realistically to agent actions and decisions.

  realism_guidelines: |          # Optional: Specific constraints
    - Economic strength changes gradually (±50-200 per step typically)
    - Major events require buildup, not instant occurrence
    - Agent actions have logical consequences
    - Bankruptcy (economic_strength = 0) is possible but severe

  scripted_events:               # Optional: Guaranteed events
    - step: 20
      type: "major_war"
      description: "A great war must begin. Decide how it manifests based on prior tensions."

    - step: 35
      type: "natural_disaster"
      description: "Major earthquake strikes. Affects all agents differently."

  context_window_size: 5         # Optional: History steps (default: 5)

global_vars:
  geopolitical_tension:
    type: float
    default: 0.3
    min: 0.0
    max: 1.0

  market_volatility:
    type: float
    default: 0.2
    min: 0.0
    max: 1.0

agent_vars:
  economic_strength:
    type: float
    default: 1000.0
    min: 0.0

  military_power:
    type: int
    default: 50
    min: 0
    max: 100

  public_support:
    type: float
    default: 0.5
    min: 0.0
    max: 1.0

agents:
  - name: "Agent A"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: "You are an ambitious leader seeking regional dominance."
    variables:
      economic_strength: 1500.0
      military_power: 70

  - name: "Agent B"
    llm:
      provider: "ollama"
      model: "mistral"
    system_prompt: "You are a defensive leader focused on stability."
```

## Validation & Error Handling

### Validation Pipeline

```python
1. JSON Structure Validation
   ✓ Valid JSON format
   ✓ Required keys: state_updates, events, agent_messages, reasoning
   ✓ Correct nested structure

2. Reference Validation
   ✓ Agent names exist in scenario
   ✓ Variable names exist in definitions

3. Type Validation
   ✓ Variable values match types (int/float/bool/list/dict)
   ✓ Event structure correct

4. Constraint Validation & Clamping
   ✓ Check min/max constraints
   ✓ If violated: CLAMP to boundary
   ✓ Log constraint hit for next step

5. Apply Updates
   ✓ Update GameEngine state
   ✓ Store events and messages
```

### Retry Logic

```python
max_attempts = 3
attempt = 1

while attempt <= max_attempts:
    response = llm.generate(context)
    result = validate(response)

    if result.success:
        apply_with_clamping(response)
        break
    else:
        if attempt == max_attempts:
            raise SimulationError(
                f"Engine failed after {max_attempts} attempts.\n"
                f"Error: {result.error}"
            )

        context = add_error_feedback(context, result.error)
        attempt += 1
```

### Constraint Feedback

When constraints are hit, include in next step's context:

```
Step N-1:
  Constraint Hit: Agent A economic_strength attempted -50, clamped to 0
    → Agent A is now BANKRUPT. Cannot go lower.
  Constraint Hit: Agent B military_power attempted 120, clamped to 100
    → Agent B hit maximum military capacity.
```

Engine incorporates this into narratives:
```
"Your economy has COMPLETELY COLLAPSED (0). You are bankrupt..."
"Your military is at MAXIMUM CAPACITY (100). No more expansion possible..."
```

## Implementation

### New Files

```
src/core/simulator_agent.py    # LLM-powered simulation orchestrator
src/core/engine_validator.py   # Validation and constraint enforcement
tools/migrate_scenario.py      # Migration helper for old scenarios
```

### Modified Files

```
src/core/orchestrator.py       # Integrate SimulatorAgent
src/core/game_engine.py        # Simplify to state management only
src/core/state.py              # Add constraint utilities
src/utils/config_parser.py     # Parse engine configuration
src/utils/logging_config.py    # Add ENG001-ENG099 codes
```

### Class Structure

#### SimulatorAgent

```python
class SimulatorAgent:
    """LLM-powered simulation orchestrator."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        game_engine: GameEngine,
        system_prompt: str,
        simulation_plan: str,
        realism_guidelines: str,
        scripted_events: List[ScriptedEvent],
        context_window_size: int = 5
    ):
        self.llm_provider = llm_provider
        self.game_engine = game_engine
        self.system_prompt = system_prompt
        self.simulation_plan = simulation_plan
        self.realism_guidelines = realism_guidelines
        self.scripted_events = scripted_events
        self.context_window_size = context_window_size

        self.step_history: List[StepRecord] = []
        self.constraint_feedback: List[ConstraintHit] = []

    def initialize_simulation(self, agent_names: List[str]) -> Dict[str, str]:
        """Generate initial agent messages for Step 0."""

    def process_step(
        self,
        step_number: int,
        agent_responses: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Process agent responses, update world state, generate new messages.
        Returns: agent_messages for next step
        """
```

#### EngineValidator

```python
class EngineValidator:
    """Validates SimulatorAgent LLM output."""

    @staticmethod
    def validate_structure(response: str) -> ValidationResult:
        """Validate JSON structure and required keys."""

    @staticmethod
    def validate_references(
        output: dict,
        agents: List[str],
        var_defs: dict
    ) -> ValidationResult:
        """Validate agent/variable names exist."""

    @staticmethod
    def validate_types(
        output: dict,
        global_var_defs: dict,
        agent_var_defs: dict
    ) -> ValidationResult:
        """Validate variable values match types."""

    @staticmethod
    def apply_constraints(
        state_updates: dict,
        global_var_defs: dict,
        agent_var_defs: dict
    ) -> Tuple[dict, List[ConstraintHit]]:
        """Apply min/max constraints, return clamped values and hits."""
```

#### GameEngine (Simplified)

```python
class GameEngine:
    """Pure state management - no game logic."""

    def __init__(self, config: dict):
        self.game_state = GameState(...)
        self.global_var_defs = config['global_vars']
        self.agent_var_defs = config['agent_vars']

    def get_global_var(self, var_name: str):
        """Get global variable value."""

    def set_global_var(self, var_name: str, value):
        """Set global variable with type validation."""

    def get_agent_var(self, agent_name: str, var_name: str):
        """Get agent variable value."""

    def set_agent_var(self, agent_name: str, var_name: str, value):
        """Set agent variable with type validation."""

    def get_current_state(self) -> GameState:
        """Get full current state."""

    def apply_state_updates(self, updates: dict) -> List[ConstraintHit]:
        """Apply updates from SimulatorAgent, return constraint violations."""
```

### Data Models

```python
@dataclass
class ScriptedEvent:
    step: int
    type: str
    description: str

@dataclass
class EngineOutput:
    state_updates: Dict  # {global_vars: {...}, agent_vars: {agent_name: {...}}}
    events: List[Event]
    agent_messages: Dict[str, str]
    reasoning: str

@dataclass
class ConstraintHit:
    agent_name: Optional[str]  # None for global vars
    var_name: str
    attempted_value: Union[int, float]
    clamped_value: Union[int, float]
    constraint_type: str  # "min" or "max"

@dataclass
class StepRecord:
    """Record of a single step for context window."""
    step_number: int
    changes: Dict  # Only changed variables
    events: List[Event]
    agent_responses: Dict[str, str]
    reasoning: str
    constraint_hits: List[ConstraintHit]
```

## Logging & Observability

### Logging Codes

```python
ENG001 = "engine_initialization"
ENG002 = "engine_step_start"
ENG003 = "engine_llm_call"
ENG004 = "engine_llm_response"
ENG005 = "engine_validation_success"
ENG006 = "engine_validation_failed"
ENG007 = "engine_retry"
ENG008 = "engine_fatal_error"
ENG009 = "engine_constraint_hit"
ENG010 = "engine_state_update"
ENG011 = "engine_event_generated"
ENG012 = "engine_scripted_event"
```

### Output Formats

**JSONL (simulation.jsonl):**
```jsonl
{"timestamp": "2025-11-22T10:15:30", "code": "ENG001", "step": 0, "message": "SimulatorAgent initialized", "provider": "gemini", "model": "gemini-1.5-flash"}
{"timestamp": "2025-11-22T10:15:33", "code": "ENG010", "step": 1, "message": "State updates applied", "changes": {"global_vars": 1, "agent_vars": 2}}
{"timestamp": "2025-11-22T10:15:33", "code": "SIM_REASONING", "step": 1, "reasoning": "Agents took constructive actions, applied growth"}
{"timestamp": "2025-11-22T10:15:35", "code": "ENG009", "step": 5, "message": "Constraint hit", "agent": "Agent B", "var": "economic_strength", "attempted": -50, "clamped": 0}
```

**Console (human-readable):**
```
[Step 1] 🎮 SIMULATOR REASONING:
         Agents took constructive actions, applied growth

[Step 1] 📊 STATE UPDATES:
         Agent A: economic_strength 1000 → 1100 (+100)
         Agent B: military_power 60 → 65 (+5)

[Step 1] ⚡ EVENT: Economic growth (affects: Agent A)

[Step 5] ⚠️  CONSTRAINT HIT:
         Agent B: economic_strength attempted -50, clamped to 0
         → Agent B is now BANKRUPT
```

**What Gets Logged:**
- SimulatorAgent reasoning (from LLM output) - reused, not generated
- State changes (delta format: old → new)
- Events generated
- Constraint hits
- Agent messages (already logged)

## Testing Strategy

### Unit Tests

```python
tests/unit/test_simulator_agent.py
  - Context building (format, sections)
  - History windowing (last N steps)
  - Constraint feedback formatting
  - Scripted event inclusion
  - Mock LLM responses

tests/unit/test_engine_validator.py
  - JSON structure validation
  - Reference validation
  - Type validation
  - Constraint clamping
  - Error messages

tests/unit/test_game_engine.py (modified)
  - State get/set operations
  - Variable type validation
  - State queries
```

### Integration Tests

```python
tests/integration/test_simulator_integration.py
  - Full SimulatorAgent + GameEngine workflow
  - Multi-step simulations with mock LLM
  - Constraint feedback loop
  - Scripted event triggering
  - Error handling and retries

tests/integration/test_full_simulation.py
  - Complete simulation with mock SimulatorAgent
  - Orchestrator coordination
  - Persistence of engine decisions
  - Logging output (JSONL + console)

tests/integration/test_engine_failure_modes.py
  - Invalid JSON retry logic
  - Constraint violations
  - Max retries exceeded
  - Missing required fields
  - Invalid references
```

### Mock Provider

```python
class MockSimulatorLLM(LLMProvider):
    """Mock LLM for deterministic testing."""

    def __init__(self, responses: List[dict]):
        self.responses = responses
        self.call_count = 0

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.dumps(response)
```

### Test Scenarios

```
scenarios/test_engine_basic.yaml           # Simple 5-step test
scenarios/test_engine_constraints.yaml     # Constraint hitting
scenarios/test_engine_scripted_events.yaml # Scripted event triggering
scenarios/test_engine_long_simulation.yaml # 50+ steps (context window)
```

## Migration Guide

### Breaking Changes

This is a **breaking change** - all existing scenarios must be updated.

**Old format (removed):**
```yaml
max_steps: 10
orchestrator_message: "What do you do?"  # ← REMOVED

game_state:
  initial_resources: 1000
```

**New format (required):**
```yaml
max_steps: 10

engine:                              # ← REQUIRED
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the game master.
  simulation_plan: |
    Manage a simple 10-step simulation.

global_vars:
  resources:
    type: int
    default: 1000
    min: 0
```

### Migration Steps

1. **Remove `orchestrator_message`** - SimulatorAgent generates messages

2. **Add `engine` block:**
   ```yaml
   engine:
     provider: "gemini"
     model: "gemini-1.5-flash"
     system_prompt: |
       You are the game master for this simulation.
       Maintain realistic cause-and-effect.
     simulation_plan: |
       [Describe what should happen]
   ```

3. **Convert `game_state` to `global_vars`:**
   ```yaml
   # Old
   game_state:
     initial_resources: 1000

   # New
   global_vars:
     resources:
       type: int
       default: 1000
       min: 0
   ```

4. **Define `agent_vars`:**
   ```yaml
   agent_vars:
     health:
       type: int
       default: 100
       min: 0
       max: 100
   ```

### Migration Tool

```bash
python tools/migrate_scenario.py scenarios/old_scenario.yaml
# Prompts for engine configuration
# Converts game_state to global_vars
# Validates output
# Writes scenarios/old_scenario_migrated.yaml
```

## Key Design Decisions

1. **Engine is required** - Every scenario must configure SimulatorAgent
   - Rationale: Engine is core to new simulation model

2. **Single LLM call per step** - All outputs in one response
   - Rationale: Efficient context use, coherent updates

3. **Delta reporting** - Only changed variables in history
   - Rationale: Conserve context window space

4. **Constraint clamping + feedback** - Clamp now, inform engine later
   - Rationale: Keeps simulation running, engine learns boundaries

5. **Scripted events visible ahead** - Engine sees upcoming events
   - Rationale: Enables narrative buildup and foreshadowing

6. **Fail-fast on errors** - 2-3 retries, then stop
   - Rationale: Engine is critical infrastructure

7. **GameEngine simplified** - State management only
   - Rationale: Clear separation (state vs logic)

8. **SimulatorAgent naming** - Mirrors player Agent pattern
   - Rationale: Intuitive - both are agents (game master vs players)

## Success Criteria

- ✅ SimulatorAgent generates realistic state updates
- ✅ Agent responses correctly interpreted (both specific and vague)
- ✅ Constraints enforced with feedback loop
- ✅ Scripted events trigger with creative execution
- ✅ Personalized agent messages provide context
- ✅ Context window management efficient (delta reporting)
- ✅ Error handling robust (retry + fail-fast)
- ✅ Logging provides full observability
- ✅ All tests pass (unit + integration)
- ✅ Migration tool works for existing scenarios

## Future Enhancements

- Multi-step event chains (event A triggers event B)
- Agent-to-agent interactions (messages between agents)
- Dynamic context window sizing (grow/shrink based on complexity)
- SimulatorAgent self-reflection (analyze own decisions)
- Alternative LLM providers (Claude, GPT-4, local models)
- Event probability system (engine decides when to trigger)
- State snapshots (rollback to previous states)

## References

- Existing architecture: `docs/architecture.md`
- LLM integration: `docs/LLM_INTEGRATION.md`
- Scenario creation: `docs/scenario-creation.md`
