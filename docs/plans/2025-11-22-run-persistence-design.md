# Run Persistence Design

## Overview

Implement persistence for simulation runs, where each execution of `uv run src/main.py` creates a unique run directory with a single JSON file tracking state snapshots and agent messages throughout the simulation.

## Directory Structure

Each run creates a unique directory under `results/` at project root:

```
results/
  run_config_2025-11-22_14-30-52/
    state.json
  run_config_2025-11-22_15-45-10/
    state.json
```

**Run ID Format**: `run_<scenario_name>_<YYYY-MM-DD_HH-MM-SS>`
- Scenario name extracted from config file path (e.g., `scenarios/config.yaml` → `config`)
- Timestamp captured at simulation start
- Scenario name sanitized (replace invalid path characters with underscores, max 50 chars)

## Save File Structure

Single `state.json` file per run, updated throughout execution:

```json
{
  "run_id": "run_config_2025-11-22_14-30-52",
  "scenario": "config",
  "started_at": "2025-11-22T14:30:52",
  "snapshots": [
    {
      "step": 0,
      "game_state": {
        "resources": 150,
        "difficulty": "normal",
        "round": 0
      },
      "global_vars": {
        "interest_rate": 0.05,
        "market_volatility": 0.1,
        "simulation_active": true
      },
      "agent_vars": {
        "Agent Alpha": {
          "economic_strength": 2500.0,
          "risk_tolerance": 0.8,
          "action_count": 0
        },
        "Agent Beta": {
          "economic_strength": 1000.0,
          "risk_tolerance": 0.5,
          "action_count": 0
        }
      },
      "messages": [
        {
          "from": "Orchestrator",
          "to": "Agent Alpha",
          "content": "Please proceed with the next action"
        },
        {
          "from": "Agent Alpha",
          "to": "Orchestrator",
          "content": "Acknowledged, processing task"
        },
        {
          "from": "Orchestrator",
          "to": "Agent Beta",
          "content": "Please proceed with the next action"
        },
        {
          "from": "Agent Beta",
          "to": "Orchestrator",
          "content": "Ready for next instruction"
        }
      ]
    },
    {
      "step": 5,
      "game_state": {...},
      "global_vars": {...},
      "agent_vars": {...},
      "messages": [...]
    }
  ]
}
```

## Save Frequency

**Command-line argument**: `--save-frequency N` (or `-sf N`)

**Behavior**:
- `N = 0`: No intermediate saves, only final state
- `N = 1`: Save after every step + final
- `N = k`: Save every k steps + final
- **Default**: `1` (save final state only when not specified)

**Save Timing**:
- Intermediate saves occur at end of each step (after all agents complete their actions)
- Final save always happens at simulation completion
- Each save appends a new snapshot to the `state.json` file

**File Update Process**:
1. Read existing `state.json` (if exists)
2. Append new snapshot to `snapshots` array
3. Write entire structure back to file atomically

## Implementation Components

### New Module: `src/persistence.py`

```python
class RunPersistence:
    """Manages persistence of simulation run data."""

    def __init__(self, scenario_name: str, save_frequency: int):
        """
        Initialize persistence layer.
        - Creates results/ directory if needed
        - Creates unique run directory
        - Initializes state.json with metadata
        """

    def save_snapshot(
        self,
        step: int,
        game_state: dict,
        global_vars: dict,
        agent_vars: dict,
        messages: list[dict]
    ):
        """Append snapshot to state.json."""

    def should_save(self, step: int) -> bool:
        """Determine if current step should be saved."""

    def save_final(
        self,
        step: int,
        game_state: dict,
        global_vars: dict,
        agent_vars: dict,
        messages: list[dict]
    ):
        """Always save final state regardless of frequency."""
```

### Modified Files

**`src/main.py`**:
- Add `argparse` for `--save-frequency` argument
- Extract scenario name from config path
- Pass save_frequency to Orchestrator

**`src/orchestrator.py`**:
- Initialize `RunPersistence` instance
- Collect messages in a buffer during each step
- Call `persistence.should_save()` and `persistence.save_snapshot()` after each step
- Call `persistence.save_final()` at completion

**`src/game_engine.py`**:
- Add method to serialize state into JSON-compatible dict:
  - `get_state_snapshot() -> dict` returning game_state, global_vars, agent_vars

## Error Handling

**Directory Creation**:
- Create `results/` if it doesn't exist (use `exist_ok=True`)
- If run directory already exists, append `_1`, `_2`, etc. to make unique

**File I/O**:
- Atomic writes using temp file + rename to prevent corruption
- Graceful handling if `state.json` read fails (treat as new file)
- Log warnings on save failure, but continue simulation

**Scenario Name Extraction**:
- Extract from config path: `scenarios/my_scenario.yaml` → `my_scenario`
- Sanitize: replace `/`, `\`, spaces, special characters with `_`
- Truncate to 50 characters maximum

**Edge Cases**:
- `save_frequency > max_steps`: Only final save occurs
- `save_frequency = 0`: Only final save occurs
- Empty messages list: Save snapshot with empty messages array
- Simulation crash: Partial `state.json` remains valid with last saved snapshots

## Message Collection

Messages are collected during orchestrator's step execution:

```python
messages = []

for agent in self.agents:
    orchestrator_msg = self.game_engine.get_message_for_agent(agent.name)
    messages.append({
        "from": "Orchestrator",
        "to": agent.name,
        "content": orchestrator_msg
    })

    agent_response = agent.respond(orchestrator_msg)
    messages.append({
        "from": agent.name,
        "to": "Orchestrator",
        "content": agent_response
    })

    self.game_engine.process_agent_response(agent.name, agent_response)
```

Then pass `messages` list to persistence layer when saving.
