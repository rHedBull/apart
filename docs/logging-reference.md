# Logging Reference

## Overview

The simulation uses structured logging with predefined message codes for easy filtering and analysis. Each log entry includes:
- Timestamp (ISO 8601 format)
- Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Message code (e.g., SIM001, AGT002)
- Human-readable message
- Contextual data (step, agent_name, etc.)
- Optional performance metrics (duration_ms)

## Message Codes

### Simulation Lifecycle (SIM)

| Code | Level | Description | Context Fields |
|------|-------|-------------|----------------|
| SIM001 | INFO | Simulation started | num_agents, max_steps |
| SIM002 | INFO | Simulation completed | total_steps |
| SIM003 | INFO | Step started | step, max_steps |
| SIM004 | INFO | Step completed | step |
| SIM005 | DEBUG | Round advanced | round |

### Agent Operations (AGT)

| Code | Level | Description | Context Fields |
|------|-------|-------------|----------------|
| AGT001 | INFO | Agent initialized | agent_name |
| AGT002 | INFO | Agent message sent | agent_name, step, content |
| AGT003 | INFO | Agent response received | agent_name, step, response |
| AGT004 | DEBUG | Agent state updated | agent_name, step |
| AGT005 | ERROR | Agent error | agent_name, error |

### Persistence Operations (PER)

| Code | Level | Description | Context Fields |
|------|-------|-------------|----------------|
| PER001 | INFO | Run directory created | run_id, run_dir |
| PER002 | INFO | Snapshot saved | step |
| PER003 | INFO | Final state saved | step |
| PER004 | ERROR | Persistence error | error |

### Game Engine (GME)

| Code | Level | Description | Context Fields |
|------|-------|-------------|----------------|
| GME001 | INFO | Game state initialized | - |
| GME002 | DEBUG | Game state updated | - |
| GME003 | DEBUG | Variable changed | var_name, old_value, new_value |
| GME004 | ERROR | Game engine error | error |

### Configuration (CFG)

| Code | Level | Description | Context Fields |
|------|-------|-------------|----------------|
| CFG001 | INFO | Configuration loaded | config_path |
| CFG002 | ERROR | Configuration validation error | error |

### Performance (PRF)

| Code | Level | Description | Context Fields |
|------|-------|-------------|----------------|
| PRF001 | DEBUG | Operation timing | duration_ms, operation details |

## Output Formats

### Console Output

Human-readable format with ANSI colors:
```
16:05:39.753 [INFO] [AGT001] Agent initialized [agent_name=Agent Alpha]
16:05:39.753 [INFO] [AGT002] Message sent to agent [agent_name=Agent Alpha, step=1]
```

Color codes:
- DEBUG: Cyan
- INFO: Green
- WARNING: Yellow
- ERROR: Red
- CRITICAL: Magenta

### JSONL Output

Machine-readable JSON lines in `simulation.jsonl`:
```json
{
  "timestamp": "2025-11-22T16:05:39.753533",
  "level": "INFO",
  "code": "AGT001",
  "message": "Agent initialized",
  "context": {
    "agent_name": "Agent Alpha"
  },
  "duration_ms": null
}
```

## Analysis Examples

### Filter by Code Pattern

```bash
# All agent operations
grep "AGT" results/run_*/simulation.jsonl | jq .

# All errors
grep "ERROR" results/run_*/simulation.jsonl | jq .

# Specific agent's responses
grep "AGT003" results/run_*/simulation.jsonl | jq 'select(.context.agent_name == "Agent Alpha")'
```

### Extract Performance Metrics

```bash
# Average step duration
grep "PRF001" results/run_*/simulation.jsonl | jq '.duration_ms' | awk '{sum+=$1; count++} END {print sum/count}'

# Find slowest operations
grep "PRF001" results/run_*/simulation.jsonl | jq -c '{op: .message, duration: .duration_ms, step: .context.step}' | sort -t: -k3 -n
```

### Load into Pandas

```python
import pandas as pd
import json

# Load JSONL into DataFrame
logs = []
with open('results/run_config_2025-11-22_16-05-39/simulation.jsonl') as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Expand context into columns
df = pd.concat([df.drop('context', axis=1), df['context'].apply(pd.Series)], axis=1)

# Filter agent responses
agent_responses = df[df['code'] == 'AGT003']

# Count messages by agent
message_counts = df[df['code'] == 'AGT002'].groupby('agent_name').size()
```

## Log Levels

- **DEBUG**: Detailed diagnostic information (agent state updates, round advances)
- **INFO**: General informational messages (agent init, messages sent/received)
- **WARNING**: Warning messages (not currently used)
- **ERROR**: Error messages (persistence failures, agent errors)
- **CRITICAL**: Critical errors (not currently used)

Default minimum level: INFO (set via `min_log_level` parameter)
