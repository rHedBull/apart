# Simulation Pause/Resume Design

## Overview

Add ability to pause and resume running simulations for:
- Long-running simulations that span multiple sessions
- Resource management (API quota, system resources)

## Requirements

- Pause/resume via both CLI commands and API endpoints
- Mid-step pause behavior: user's choice (wait for step completion or `--force` immediate)
- Short auto-generated run IDs for identification (existing 8-char UUIDs)
- `apart list` command to show all runs with status
- Integrate with existing API server and job queue

## Status Model

Extend existing `SimulationStatus` enum in `src/server/models.py`:

```python
class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"       # NEW
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
```

New event types for EventBus:
- `simulation_paused` - emitted when pause completes
- `simulation_resumed` - emitted when resume starts

## API Endpoints

Add to existing `/api/v1/runs` router:

```
POST /api/v1/runs/{run_id}/pause
  Query params: ?force=true (optional - immediate pause, drops current step)
  Response: {"status": "paused", "paused_at_step": 15}

POST /api/v1/runs/{run_id}/resume
  Response: {"status": "running", "resuming_from_step": 15}
```

### Pause Flow

1. Validate run exists and status is `running`
2. Signal worker to pause via Redis pub/sub (`pause:{run_id}` channel)
3. Worker completes current step (unless `force=true`)
4. Worker emits `simulation_paused` event with current step
5. Endpoint returns success

### Resume Flow

1. Validate run exists and status is `paused`
2. Load state from `results/{run_id}/state.json`
3. Enqueue new job to continue from saved step
4. Emit `simulation_resumed` event
5. Return success

## CLI Tool

Create `src/cli.py` with proper CLI using `typer`:

```bash
# Run a simulation (submits to job queue)
apart run scenarios/config.yaml
apart run scenarios/config.yaml --name my-experiment
apart run scenarios/config.yaml --save-frequency 5

# Pause/resume
apart pause <run_id>
apart pause <run_id> --force

apart resume <run_id>

# List runs
apart list
apart list --status paused
apart list --status running

# Get details of a specific run
apart show <run_id>
```

CLI submits to job queue via API (requires server running).

## Pause Signaling

Use Redis pub/sub (already using Redis for job queue):

```python
# In Orchestrator.run() loop
for step in range(current_step, max_steps + 1):
    if self._check_pause_requested():  # Check Redis
        self._save_state()
        emit_event("simulation_paused", run_id=self.run_id, step=step)
        return  # Exit cleanly

    self._run_step(step)
```

For `--force` pause, set additional flag for immediate return.

## State Restoration

Orchestrator changes for resume support:

```python
class Orchestrator:
    def __init__(self, scenario_path, run_id=None, resume_state=None):
        if resume_state:
            self._restore_from_state(resume_state)
        else:
            self._initialize_fresh()
```

Resume job parameters:
```python
job_queue.enqueue(
    run_simulation_task,
    run_id=run_id,
    scenario_path=original_scenario_path,
    resume_from_step=last_step + 1,
    resume_state=loaded_state
)
```

## Files to Modify/Create

| File | Change |
|------|--------|
| `src/server/models.py` | Add `PAUSED` to `SimulationStatus` |
| `src/server/routes/v1.py` | Add `/pause` and `/resume` endpoints |
| `src/server/routes/v1.py` | Update `_get_run_status()` for new events |
| `src/server/worker_tasks.py` | Add pause checking, resume support |
| `src/core/orchestrator.py` | Add pause signal checking, state restoration |
| `src/cli.py` | **New** - CLI tool with subcommands |
| `pyproject.toml` | Add CLI entry point, typer dependency |

## Dependencies

- `typer` - CLI framework

## Event Flow

```
apart run scenario.yaml
  → POST /api/v1/runs → job queued → worker runs → events emitted

apart pause abc123
  → POST /api/v1/runs/abc123/pause → Redis pub/sub → worker pauses → simulation_paused event

apart resume abc123
  → POST /api/v1/runs/abc123/resume → load state → new job queued → simulation_resumed event
```
