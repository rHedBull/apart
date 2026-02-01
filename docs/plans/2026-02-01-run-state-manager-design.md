# Run State Manager - Implementation Plan

## Problem Statement

Run state is currently scattered across multiple sources with no single source of truth:

| Source | Location | What it tracks |
|--------|----------|---------------|
| EventBus | `src/server/event_bus.py` | Events (started, paused, completed, failed) |
| RQ Jobs | `src/server/job_queue.py` | Queue status (queued, started, finished, failed) |
| Results dir | `results/{run_id}/state.json` | Persisted snapshots |
| Database | `src/server/database.py` | SQLite records (optional mode) |
| `_get_run_status()` | `src/server/routes/v1.py:56-110` | Ad-hoc reconciliation |

This causes:
- Complex status derivation logic
- Race conditions between sources
- Inconsistent state after crashes
- No proper state machine enforcement

## Proposed Solution: RunStateManager

A centralized state manager with:
1. **Single source of truth** in Redis
2. **Defined state machine** with valid transitions
3. **Atomic state transitions** with optimistic locking
4. **Worker heartbeat tracking** for crash detection
5. **Event-driven updates** from existing event system

---

## State Machine Definition

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐   │
│ PENDING │───▶│ RUNNING │───▶│ PAUSED  │───▶│  RUNNING  │───┘
└─────────┘    └─────────┘    └─────────┘    └───────────┘
     │              │              │               │
     │              │              │               │
     │              ▼              ▼               ▼
     │         ┌─────────┐   ┌───────────┐   ┌─────────┐
     │         │COMPLETED│   │INTERRUPTED│   │ FAILED  │
     │         └─────────┘   └───────────┘   └─────────┘
     │              ▲              ▲               ▲
     │              │              │               │
     └──────────────┴──────────────┴───────────────┘
                  (timeout/cancel)
```

### Valid Transitions

```python
VALID_TRANSITIONS = {
    "pending":     ["running", "cancelled"],
    "running":     ["paused", "completed", "failed", "interrupted"],
    "paused":      ["running", "cancelled", "interrupted"],
    "completed":   [],  # terminal
    "failed":      [],  # terminal
    "interrupted": ["running"],  # can resume interrupted runs
    "cancelled":   [],  # terminal
}
```

---

## Data Model

### Redis Keys

```
apart:run:{run_id}:state     → JSON state object
apart:run:{run_id}:heartbeat → timestamp (TTL 30s)
apart:runs:index             → sorted set of run_ids by created_at
```

### State Object Schema

```python
@dataclass
class RunState:
    run_id: str
    status: SimulationStatus
    scenario_path: str
    scenario_name: str

    # Progress
    current_step: int
    total_steps: int | None

    # Timestamps
    created_at: datetime
    started_at: datetime | None
    paused_at: datetime | None
    completed_at: datetime | None

    # Worker tracking
    worker_id: str | None
    last_heartbeat: datetime | None

    # Metadata
    priority: str
    error: str | None
    danger_count: int

    # Version for optimistic locking
    version: int
```

---

## Implementation Components

### 1. New File: `src/server/run_state.py`

```python
class RunStateManager:
    """Centralized run state management with Redis backend."""

    def __init__(self, redis_conn: Redis):
        self._redis = redis_conn

    # State Operations
    def create_run(self, run_id: str, scenario_path: str, priority: str) -> RunState
    def get_state(self, run_id: str) -> RunState | None
    def transition(self, run_id: str, new_status: str, **kwargs) -> RunState
    def list_runs(self, status: str | None = None) -> list[RunState]

    # Worker Heartbeat
    def heartbeat(self, run_id: str, worker_id: str, step: int)
    def check_stale_runs(self, timeout_seconds: int = 60) -> list[str]

    # Atomic Operations
    def _atomic_transition(self, run_id: str, expected_version: int,
                          new_status: str, updates: dict) -> RunState
```

### 2. Integration Points

#### A. Job Queue (`src/server/job_queue.py`)

**Current:** Creates RQ job, status tracked by RQ
**Change:** Also create RunState entry

```python
# Line ~96 in enqueue_simulation()
def enqueue_simulation(...):
    # NEW: Create run state entry
    state_manager = get_state_manager()
    state_manager.create_run(run_id, scenario_path, priority)

    # Existing: enqueue RQ job
    job = queue.enqueue(...)
```

#### B. Worker Tasks (`src/server/worker_tasks.py`)

**Current:** Emits events, no heartbeat
**Change:** Update state + send heartbeats

```python
# Line ~52 in run_simulation_task()
def run_simulation_task(run_id, scenario_path, resume_from_step=None):
    state_manager = get_state_manager()

    # Transition to running
    state_manager.transition(run_id, "running", worker_id=get_worker_id())

    # Start heartbeat thread
    heartbeat_thread = start_heartbeat(run_id)

    try:
        orchestrator = Orchestrator(...)
        orchestrator.run(...)
        state_manager.transition(run_id, "completed")
    except Exception as e:
        state_manager.transition(run_id, "failed", error=str(e))
        raise
    finally:
        heartbeat_thread.stop()
```

#### C. Orchestrator (`src/core/orchestrator.py`)

**Current:** Emits events, checks pause signal
**Change:** Also update state via callback

```python
# Line ~468 in step loop
def run(self, start_step: int = 1):
    for step in range(start_step, self.max_steps + 1):
        # NEW: Update state with current step
        if self._state_callback:
            self._state_callback(step=step)

        # Existing: check pause
        if self._check_and_handle_pause():
            break
```

#### D. Pause/Resume Endpoints (`src/server/routes/v1.py`)

**Current:** Publishes Redis signal, derives status from events
**Change:** Use state manager for transitions

```python
# Line ~736 in pause_simulation()
@router.post("/{run_id}/pause")
async def pause_simulation(run_id: str):
    state_manager = get_state_manager()
    state = state_manager.get_state(run_id)

    if state.status != "running":
        raise HTTPException(409, f"Cannot pause: status is {state.status}")

    # Publish pause signal (existing)
    publish_pause_signal(run_id, force)

    # Don't transition yet - worker will transition when it sees signal
    return {"status": "pause_requested", ...}
```

#### E. List/Show Endpoints (`src/server/routes/v1.py`)

**Current:** Complex merging of EventBus + RQ + results dir
**Change:** Simple read from state manager

```python
# Line ~113 in list_runs()
@router.get("")
async def list_runs():
    state_manager = get_state_manager()
    runs = state_manager.list_runs()

    return {"runs": [run.to_api_dict() for run in runs]}
```

#### F. Event Bus (`src/server/event_bus.py`)

**Current:** Primary source of status info
**Change:** Events still emitted, but state manager is authoritative

```python
# Line ~258 in emit_event()
def emit_event(self, event: SimulationEvent):
    # Existing: persist and broadcast event
    self._persist_event(event)
    self._broadcast(event)

    # NEW: Sync state manager (optional, for consistency)
    # State manager is authoritative, but keep in sync
    if event.event_type == "simulation_completed":
        state_manager.transition(event.run_id, "completed")
```

---

## Migration Strategy

### Phase 1: Add State Manager (Non-Breaking)

1. Create `src/server/run_state.py` with RunStateManager
2. Add state creation in `enqueue_simulation()`
3. Add state reads in `list_runs()` / `get_run_detail()` with fallback to old logic
4. Add tests for state manager

### Phase 2: Worker Integration

1. Add state transitions in `worker_tasks.py`
2. Add heartbeat mechanism
3. Add stale run detection
4. Add callback for step updates from Orchestrator

### Phase 3: Cleanup

1. Remove complex status derivation in `_get_run_status()`
2. Simplify `list_runs()` to only use state manager
3. Keep EventBus for real-time streaming (separate concern)
4. Remove redundant RQ status checks

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/server/run_state.py` | **NEW** - RunStateManager class |
| `src/server/job_queue.py` | Add state creation in `enqueue_simulation()` |
| `src/server/worker_tasks.py` | Add state transitions + heartbeat |
| `src/core/orchestrator.py` | Add state callback for step updates |
| `src/server/routes/v1.py` | Simplify `list_runs()`, `get_run_detail()`, `pause`, `resume` |
| `src/server/event_bus.py` | Optional sync to state manager |
| `src/server/app.py` | Initialize state manager in lifespan |
| `src/cli.py` | No changes (uses API) |

---

## Testing Plan

### Unit Tests (`tests/unit/test_run_state.py`)

```python
class TestRunStateManager:
    def test_create_run()
    def test_valid_transition()
    def test_invalid_transition_raises()
    def test_optimistic_locking()
    def test_heartbeat_updates()
    def test_stale_run_detection()
    def test_list_runs_by_status()

class TestStateTransitions:
    def test_pending_to_running()
    def test_running_to_paused()
    def test_paused_to_running()
    def test_running_to_completed()
    def test_running_to_failed()
    def test_running_to_interrupted()
    def test_cannot_transition_from_terminal()
```

### Integration Tests (`tests/integration/test_run_lifecycle.py`)

```python
class TestRunLifecycle:
    def test_full_run_lifecycle()
    def test_pause_resume_lifecycle()
    def test_worker_crash_detection()
    def test_concurrent_status_updates()
```

---

## Estimated Effort

| Phase | Tasks | Effort |
|-------|-------|--------|
| Phase 1 | State manager + basic integration | 2-3 hours |
| Phase 2 | Worker + heartbeat | 2-3 hours |
| Phase 3 | Cleanup + full migration | 1-2 hours |
| Testing | Unit + integration tests | 2-3 hours |
| **Total** | | **7-11 hours** |

---

## Open Questions

1. **Database mode**: Should state manager also support SQLite backend for database mode?
2. **Event streaming**: Keep EventBus for SSE streaming, or also migrate to state manager?
3. **Historical runs**: How to handle runs from before migration (in results/ dir)?
4. **Cleanup policy**: Auto-delete old run states after X days?
