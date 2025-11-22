# Migration Guide: v1.0 to v2.0 (Engine-Powered Simulation)

## Breaking Changes

Version 2.0 introduces the **SimulatorAgent** - an LLM-powered game master that replaces the old orchestrator message system. All scenarios must be updated.

## What Changed

### Removed
- `orchestrator_message` field (replaced by engine-generated messages)
- Static message generation in GameEngine

### Required
- `engine` configuration block (LLM provider for simulation)
- All scenarios must have engine configuration

## Migration Steps

### Option 1: Use Migration Tool

```bash
python tools/migrate_scenario.py scenarios/old_scenario.yaml
```

Follow prompts to configure engine.

### Option 2: Manual Migration

1. **Remove `orchestrator_message`**
2. **Add `engine` block:**

```yaml
engine:
  provider: "gemini"  # or "ollama"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the game master.
    Maintain realistic cause-and-effect.
  simulation_plan: |
    Describe what should happen in your simulation.
  context_window_size: 5  # Optional, defaults to 5
```

3. **Convert `game_state` to `global_vars`** (if present)
4. **Test the scenario**

## Example

### Before (v1.0)
```yaml
max_steps: 10
orchestrator_message: "What do you do?"

game_state:
  initial_resources: 1000

agents:
  - name: "Agent A"
    response_template: "I act"
```

### After (v2.0)
```yaml
max_steps: 10

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the game master.
  simulation_plan: |
    Manage a 10-step simulation.

global_vars:
  resources:
    type: int
    default: 1000
    min: 0

agents:
  - name: "Agent A"
    response_template: "I act"
```

## New Features Available

- **Personalized Messages:** Each agent gets custom narrative
- **Emergent Events:** Engine creates events based on agent actions
- **Scripted Events:** Guarantee key moments while allowing flexible execution
- **Stat Management:** Agents can't modify their own stats - engine is authoritative
- **Constraint Enforcement:** Min/max limits enforced with feedback

## Getting Help

- Design doc: `docs/plans/2025-11-22-llm-simulation-engine-design.md`
- Test scenario: `scenarios/engine_test.yaml`
- Issues: File bug reports with scenario that fails
