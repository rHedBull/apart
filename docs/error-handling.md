# Error Handling

## Overview

The simulation implements comprehensive error handling with graceful degradation for non-critical failures. Errors are logged with structured message codes and contextual information.

## Error Categories

### Fatal Errors (Exit Immediately)

These errors cause immediate termination with appropriate exit codes:

1. **Missing Configuration File**
   - Exit code: 1
   - Example: `Error: Scenario file not found: scenarios/nonexistent.yaml`

2. **Invalid YAML Syntax**
   - Exit code: 1
   - Logged as: CFG002
   - Example: `Configuration Error: Invalid YAML in configuration file`

3. **Configuration Validation Errors**
   - Exit code: 1
   - Example: Invalid variable constraints, missing required fields

4. **Keyboard Interrupt (Ctrl+C)**
   - Exit code: 130
   - Logged as: SIM002 (WARNING level)
   - Gracefully closes logger before exit

5. **Unhandled Exceptions**
   - Exit code: 1
   - Full traceback printed to stderr
   - Logged as: SIM002 (CRITICAL level)

### Non-Fatal Errors (Continue Simulation)

These errors are logged but allow simulation to continue:

1. **Agent Message Generation Failure**
   - Message code: AGT005 (ERROR)
   - Behavior: Skip agent for current step, continue with next agent
   - Logged context: agent_name, step, error

2. **Agent Response Failure**
   - Message code: AGT005 (ERROR)
   - Behavior: Log error response, continue with next agent
   - Error message added to conversation history

3. **Agent State Update Failure**
   - Message code: GME004 (ERROR)
   - Behavior: Skip state update, continue with next agent
   - Logged context: agent_name, step, error

4. **Snapshot Save Failure**
   - Message code: PER004 (ERROR)
   - Behavior: Log warning, continue simulation
   - Note: Final state save failure is logged as CRITICAL but doesn't prevent summary

5. **Round Advance Failure**
   - Message code: GME004 (ERROR)
   - Behavior: Log warning, continue to next step
   - Logged context: step, error

## Error Logging

All errors are logged to both console and `simulation.jsonl`:

### Console Output
```
ERROR: Agent failed to respond: division by zero
WARNING: Failed to save snapshot: Permission denied
```

### JSONL Output
```json
{
  "timestamp": "2025-11-22T16:14:12.728",
  "level": "ERROR",
  "code": "AGT005",
  "message": "Agent failed to respond: division by zero",
  "context": {
    "agent_name": "Agent Alpha",
    "step": 3,
    "error": "division by zero"
  }
}
```

## Error Recovery

### Graceful Degradation

The simulation attempts to continue despite errors:

1. **Per-Agent Isolation**: Errors in one agent don't affect others
2. **Step Continuation**: Failed operations don't terminate the step
3. **Partial State Saves**: Successful snapshots are saved even if later ones fail
4. **Summary Display**: Simulation summary shown even if final save fails

### Logger Cleanup

The `finally` block ensures logger is always closed, even on:
- Unhandled exceptions
- Keyboard interrupts
- Fatal configuration errors (after logger initialization)

## Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | Fatal error (config, validation, unhandled exception) |
| 130 | Keyboard interrupt (SIGINT) |

## Finding Errors in Logs

### Search for All Errors
```bash
grep -E "ERROR|WARNING|CRITICAL" results/run_*/simulation.jsonl
```

### Count Errors by Type
```bash
cat results/run_*/simulation.jsonl | jq -r 'select(.level == "ERROR") | .code' | sort | uniq -c
```

### Extract Agent Errors
```bash
grep "AGT005" results/run_*/simulation.jsonl | jq -c '{timestamp, agent: .context.agent_name, error: .context.error}'
```

### Find Failed Steps
```bash
cat results/run_*/simulation.jsonl | jq 'select(.level == "ERROR") | {step: .context.step, code, error: .context.error}'
```

## Best Practices

### For Users

1. **Check Exit Codes**: Non-zero exit indicates a problem
2. **Monitor Console**: Errors printed to stderr in real-time
3. **Review Logs**: Check `simulation.jsonl` for complete error history
4. **Validate Config**: Use small test runs to catch config errors early

### For Developers

1. **Log Context**: Always include relevant context (agent_name, step, etc.)
2. **Use Appropriate Codes**: Choose correct message code for error type
3. **Fail Fast for Fatal**: Don't continue if state is unrecoverable
4. **Graceful for Non-Fatal**: Allow simulation to continue when possible
5. **Clean Up Resources**: Use try/finally for resource cleanup

## Error Handling Flow

```
main.py
  ├─ Validates scenario file exists
  ├─ try:
  │   ├─ Initialize Orchestrator
  │   │   ├─ Load config (may raise FileNotFoundError, ValueError)
  │   │   ├─ Initialize persistence (creates logger)
  │   │   ├─ Initialize agents
  │   │   └─ Initialize game engine (may raise ValueError)
  │   │
  │   └─ Run simulation
  │       ├─ try:
  │       │   ├─ For each step:
  │       │   │   ├─ For each agent:
  │       │   │   │   ├─ try: Generate message
  │       │   │   │   ├─ try: Get response
  │       │   │   │   └─ try: Update state
  │       │   │   ├─ try: Save snapshot (if needed)
  │       │   │   └─ try: Advance round
  │       │   │
  │       │   └─ try: Save final state
  │       │
  │       ├─ except KeyboardInterrupt: Log and re-raise
  │       ├─ except Exception: Log and re-raise
  │       └─ finally: Close logger
  │
  ├─ except FileNotFoundError: Print error, exit 1
  ├─ except ValueError: Print error, exit 1
  ├─ except KeyboardInterrupt: Print message, exit 130
  └─ except Exception: Print traceback, exit 1
```

## Examples

### Handling Agent Errors

An agent that raises an exception:

```python
# Agent responds with error
response = agent.respond(message)  # Raises RuntimeError
```

Logged as:
```
ERROR: Agent failed to respond: Some runtime error
```

JSONL:
```json
{
  "level": "ERROR",
  "code": "AGT005",
  "message": "Agent failed to respond: Some runtime error",
  "context": {
    "agent_name": "Agent Alpha",
    "step": 3,
    "error": "Some runtime error"
  }
}
```

Simulation continues with next agent.

### Handling Configuration Errors

Invalid configuration:
```yaml
global_vars:
  interest_rate:
    type: float
    default: 5.0  # Exceeds max!
    max: 1.0
```

Result:
```
Configuration Error: Error parsing variable 'interest_rate': default (5.0) cannot be greater than max (1.0)
```

Simulation exits with code 1.
