# Agent Context Builder Design

## Problem

Agent prompts are too thin. The engine currently generates messages like:
> "Our government is mobilizing defenses and seeking international support amid the blockade."

This lacks:
- Current state variable values
- What other agents did
- Specific decision points
- Output format guidance

The engine LLM is overburdened - it must generate state updates, events, reasoning, AND rich contextual messages for each agent.

## Solution

Separate concerns:
1. **Engine LLM** focuses on complex state judgment (what changes, why)
2. **ContextBuilder (code)** assembles rich agent prompts deterministically
3. **Modules** can optionally contribute context sections and/or deterministic state updates

## Architecture

### New Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Simulation Step                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Deterministic Module Updates (code)                         │
│     └─> Modules with compute_state_updates() run first          │
│                                                                 │
│  2. Engine LLM Call                                             │
│     └─> Returns: state_updates, events, reasoning               │
│     └─> NO agent_messages                                       │
│                                                                 │
│  3. Context Builder (code)                                      │
│     └─> Builds rich prompt per agent from:                      │
│         - Current state                                         │
│         - Other agent responses                                 │
│         - Events                                                │
│         - Module-contributed sections                           │
│                                                                 │
│  4. Agent LLM Calls                                             │
│     └─> Each agent receives rich contextual prompt              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Module Code Structure

Modules can optionally have Python code files alongside YAML:

```
modules/definitions/
  agents_base.yaml           # data only
  diplomatic_base.yaml       # data only
  trust_dynamics.yaml        # data
  trust_dynamics.py          # optional code
```

**Module Python file interface:**

```python
# trust_dynamics.py
from typing import Any, Dict, Optional

def build_agent_context(
    agent_name: str,
    agent_state: Dict[str, Any],
    global_state: Dict[str, Any],
) -> Optional[str]:
    """Return context section for this agent, or None to skip."""
    trust = agent_state.get("trust_level", 50)
    if trust < 30:
        return f"WARNING: Trust critically low ({trust}/100). Others view you with suspicion."
    elif trust > 70:
        return f"ADVANTAGE: High trust ({trust}/100). Others are receptive to your proposals."
    return None

def compute_state_updates(
    agent_name: str,
    agent_state: Dict[str, Any],
    global_state: Dict[str, Any],
    step_number: int,
) -> Dict[str, Any]:
    """Return deterministic state changes, or empty dict."""
    updates = {}
    if agent_state.get("had_positive_interaction") is False:
        updates["trust_level"] = max(0, agent_state.get("trust_level", 50) - 1)
    return updates
```

## Components

### AgentContextBuilder Class

**Location:** `src/core/context_builder.py`

```python
class AgentContextBuilder:
    """Builds contextual prompts for agents from simulation state."""

    def __init__(
        self,
        time_step_duration: str,
        simulator_awareness: bool,
        composed_modules: Optional[ComposedModules] = None
    ):
        self.time_step_duration = time_step_duration
        self.simulator_awareness = simulator_awareness
        self.composed_modules = composed_modules

    def build_prompt(
        self,
        agent_name: str,
        step_number: int,
        agent_state: Dict[str, Any],
        global_state: Dict[str, Any],
        other_responses: Dict[str, str],
        events: List[Dict],
    ) -> str:
        sections = []

        # 1. Core sections (always present)
        sections.append(self._build_situation(step_number, global_state, events))
        sections.append(self._build_your_state(agent_name, agent_state))
        sections.append(self._build_other_actions(agent_name, other_responses))

        # 2. Module-contributed sections (optional)
        if self.composed_modules:
            for module in self.composed_modules.modules:
                if module.code and hasattr(module.code, 'build_agent_context'):
                    section = module.code.build_agent_context(
                        agent_name, agent_state, global_state
                    )
                    if section:
                        sections.append(section)

        # 3. Decision framing and output format (always present)
        sections.append(self._build_decision_frame())
        sections.append(self._build_output_format())

        return "\n\n".join(s for s in sections if s)
```

### Engine Output Schema (Simplified)

**Before:**
```json
{
  "state_updates": {...},
  "events": [...],
  "agent_messages": {"Taiwan": "...", "China": "..."},
  "reasoning": "..."
}
```

**After:**
```json
{
  "state_updates": {
    "global_vars": {"crisis_level": 65},
    "agent_vars": {"Taiwan": {"military_readiness": 70}}
  },
  "events": [
    {"type": "naval_incident", "description": "Chinese vessel intercepted cargo ship"}
  ],
  "reasoning": "Crisis escalating due to..."
}
```

### Updated Simulation Loop

```python
def process_step(self, step_number: int, agent_responses: Dict[str, str]):
    # 1. Apply deterministic module updates
    apply_deterministic_updates(self.composed_modules, self.game_engine, step_number)

    # 2. Call engine for complex state changes
    engine_output = self._call_engine(step_number, agent_responses)
    self.game_engine.apply_state_updates(engine_output.state_updates)

    # 3. Build context for each agent
    current_state = self.game_engine.get_current_state()
    agent_prompts = {}
    for agent_name in agent_responses.keys():
        agent_prompts[agent_name] = self.context_builder.build_prompt(
            agent_name=agent_name,
            step_number=step_number,
            agent_state=current_state["agent_vars"][agent_name],
            global_state=current_state["global_vars"],
            other_responses=agent_responses,
            events=engine_output.events,
        )

    return agent_prompts
```

## Example Built Prompt

**What Taiwan's agent receives:**

```
=== SITUATION (Day 3 of crisis) ===
Time: Step 3 (each step = 3 days)
Crisis level: 65/100
Blockade effectiveness: 55%
Diplomatic channels: Open

Recent events:
- Chinese vessel intercepted cargo ship bound for Kaohsiung
- US announced carrier group repositioning

=== YOUR CURRENT STATE ===
Military readiness: 70/100
Domestic pressure: 65/100
Escalation readiness: 40/100
Trust level: 68/100

=== WHAT OTHERS DID (Step 2) ===
China: "We are enforcing legitimate security measures. Foreign interference
will be met with resolve. Naval patrols expanded to cover southern approaches."

United States: "We urge restraint from all parties. The USS Reagan carrier
group is conducting routine repositioning. We remain committed to regional stability."

Japan: "We have activated maritime observation protocols and are coordinating
closely with US allies. Calling for diplomatic resolution."

=== TRUST DYNAMICS ===
Your trust is stable (68/100). Recent US support has maintained confidence
in alliance commitments.

=== YOUR DECISION ===
Based on this situation, what actions do you take?

Consider:
- Military posture adjustments
- Diplomatic communications (to whom? saying what?)
- Public statements
- Requests to allies

=== RESPONSE FORMAT ===
Structure your response as:
1. ASSESSMENT: Your read of the current situation (2-3 sentences)
2. ACTIONS: Specific decisions this turn
3. COMMUNICATIONS: Any messages to other parties
```

**Compare to old prompt:**
> "Our government is mobilizing defenses and seeking international support amid the blockade."

## Files to Change

### New Files

| File | Purpose |
|------|---------|
| `src/core/context_builder.py` | AgentContextBuilder class |

### Modified Files

| File | Change |
|------|--------|
| `src/core/simulator_agent.py` | Remove agent_messages from output, integrate ContextBuilder |
| `src/modules/loader.py` | Load optional .py files alongside .yaml |
| `src/modules/models.py` | Add `code` attribute to BehaviorModule |
| `src/core/engine_models.py` | Remove agent_messages from EngineOutput |

### New Module Code Files (Examples)

| File | Purpose |
|------|---------|
| `src/modules/definitions/trust_dynamics.py` | Trust decay logic, trust-based context |
| `src/modules/definitions/diplomatic_base.py` | Reputation effects, alliance context |

## Benefits

1. **Engine LLM focuses** on complex state judgment only
2. **Agent prompts are rich**, contextual, deterministic
3. **Modules can contribute** context and/or deterministic updates
4. **Easier to debug** - inspect built prompts directly
5. **Testable** - context builder is pure functions
6. **Reduced token usage** - engine prompt simpler without agent_messages formatting

## Migration Path

1. Implement ContextBuilder with core sections
2. Update SimulatorAgent to use ContextBuilder
3. Remove agent_messages from engine output schema
4. Add module code loading to ModuleLoader
5. Create example module code files (trust_dynamics.py)
6. Update tests
