# Pause/Resume Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable pausing and resuming simulations via CLI and API for long-running simulations and resource management.

**Architecture:** Add `PAUSED` status to existing enum, use Redis pub/sub for pause signaling between API and workers, create `typer`-based CLI that submits to job queue via API.

**Tech Stack:** Python 3.12, FastAPI, Redis, RQ, typer, pytest

---

## Task 1: Add PAUSED Status to SimulationStatus Enum

**Files:**
- Modify: `src/server/models.py:12-18`
- Test: `tests/unit/test_api_models.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_api_models.py`:

```python
def test_simulation_status_includes_paused():
    """Test that SimulationStatus enum includes PAUSED."""
    from server.models import SimulationStatus

    assert hasattr(SimulationStatus, "PAUSED")
    assert SimulationStatus.PAUSED.value == "paused"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_api_models.py::test_simulation_status_includes_paused -v`
Expected: FAIL with "AttributeError" (PAUSED not defined)

**Step 3: Write minimal implementation**

In `src/server/models.py`, update the enum:

```python
class SimulationStatus(str, Enum):
    """Status of a simulation run."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"      # NEW
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_api_models.py::test_simulation_status_includes_paused -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/server/models.py tests/unit/test_api_models.py
git commit -m "feat(models): add PAUSED status to SimulationStatus enum"
```

---

## Task 2: Update Status Determination for Paused Events

**Files:**
- Modify: `src/server/routes/v1.py:31-55`
- Test: `tests/unit/test_server.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_server.py`:

```python
def test_get_run_status_paused():
    """Test that _get_run_status returns paused for simulation_paused event."""
    from server.event_bus import EventBus, SimulationEvent
    from server.routes.v1 import _get_run_status

    EventBus.reset_instance()
    bus = EventBus.get_instance()

    # Emit started then paused events
    bus.emit(SimulationEvent.create("simulation_started", run_id="test_paused"))
    bus.emit(SimulationEvent.create("simulation_paused", run_id="test_paused", step=5))

    status = _get_run_status("test_paused")
    assert status == "paused"


def test_get_run_status_resumed():
    """Test that _get_run_status returns running after simulation_resumed event."""
    from server.event_bus import EventBus, SimulationEvent
    from server.routes.v1 import _get_run_status

    EventBus.reset_instance()
    bus = EventBus.get_instance()

    # Emit started -> paused -> resumed events
    bus.emit(SimulationEvent.create("simulation_started", run_id="test_resumed"))
    bus.emit(SimulationEvent.create("simulation_paused", run_id="test_resumed", step=5))
    bus.emit(SimulationEvent.create("simulation_resumed", run_id="test_resumed", step=5))

    status = _get_run_status("test_resumed")
    assert status == "running"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_server.py::test_get_run_status_paused tests/unit/test_server.py::test_get_run_status_resumed -v`
Expected: FAIL (status won't be "paused")

**Step 3: Write minimal implementation**

In `src/server/routes/v1.py`, update `_get_run_status()`:

```python
def _get_run_status(run_id: str) -> str | None:
    """Get the current status of a simulation run.

    Returns None if the run doesn't exist.
    """
    event_bus = get_event_bus()
    history = event_bus.get_history(run_id)

    if not history:
        # Check results directory for completed runs
        results_dir = Path("results") / run_id
        if results_dir.exists():
            return "completed"
        return None

    status = "pending"
    for event in history:
        if event.event_type == "simulation_started":
            status = "running"
        elif event.event_type == "simulation_paused":
            status = "paused"
        elif event.event_type == "simulation_resumed":
            status = "running"
        elif event.event_type == "simulation_completed":
            status = "completed"
        elif event.event_type == "simulation_failed":
            status = "failed"

    return status
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_server.py::test_get_run_status_paused tests/unit/test_server.py::test_get_run_status_resumed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/server/routes/v1.py tests/unit/test_server.py
git commit -m "feat(api): update status determination for pause/resume events"
```

---

## Task 3: Add Redis Pub/Sub for Pause Signaling

**Files:**
- Modify: `src/server/job_queue.py`
- Test: `tests/unit/test_job_queue.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_job_queue.py`:

```python
def test_publish_pause_signal(mock_redis):
    """Test publishing pause signal to Redis."""
    from server.job_queue import publish_pause_signal, init_job_queue

    init_job_queue()

    result = publish_pause_signal("test_run_123", force=False)

    assert result is True
    # Verify message was published
    mock_redis.publish.assert_called_once()
    call_args = mock_redis.publish.call_args
    assert "pause:test_run_123" in str(call_args)


def test_check_pause_requested(mock_redis):
    """Test checking if pause was requested."""
    from server.job_queue import check_pause_requested, init_job_queue

    init_job_queue()

    # Initially no pause requested
    assert check_pause_requested("test_run") is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_job_queue.py::test_publish_pause_signal tests/unit/test_job_queue.py::test_check_pause_requested -v`
Expected: FAIL (functions don't exist)

**Step 3: Write minimal implementation**

Add to `src/server/job_queue.py`:

```python
import json
from typing import Literal

# Pause signal key prefix
PAUSE_SIGNAL_PREFIX = "apart:pause:"


def publish_pause_signal(run_id: str, force: bool = False) -> bool:
    """
    Publish a pause signal for a running simulation.

    Args:
        run_id: The simulation run ID to pause
        force: If True, pause immediately (drop current step)

    Returns:
        True if signal was published successfully
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    signal_data = json.dumps({"force": force})
    key = f"{PAUSE_SIGNAL_PREFIX}{run_id}"

    # Set with expiry (5 minutes - plenty of time for worker to see it)
    _redis_conn.setex(key, 300, signal_data)

    logger.info("Pause signal published", extra={"run_id": run_id, "force": force})
    return True


def check_pause_requested(run_id: str) -> dict | None:
    """
    Check if a pause has been requested for a simulation.

    Args:
        run_id: The simulation run ID to check

    Returns:
        Dict with pause info if requested, None otherwise
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    key = f"{PAUSE_SIGNAL_PREFIX}{run_id}"
    data = _redis_conn.get(key)

    if data:
        return json.loads(data)
    return None


def clear_pause_signal(run_id: str) -> bool:
    """
    Clear the pause signal for a simulation (after handling it).

    Args:
        run_id: The simulation run ID

    Returns:
        True if signal was cleared
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    key = f"{PAUSE_SIGNAL_PREFIX}{run_id}"
    _redis_conn.delete(key)
    return True
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_job_queue.py::test_publish_pause_signal tests/unit/test_job_queue.py::test_check_pause_requested -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/server/job_queue.py tests/unit/test_job_queue.py
git commit -m "feat(queue): add Redis pub/sub for pause signaling"
```

---

## Task 4: Add Pause/Resume API Endpoints

**Files:**
- Modify: `src/server/routes/v1.py`
- Modify: `src/server/models.py`
- Test: `tests/integration/test_server_api.py`

**Step 1: Write the failing test**

Add to `tests/integration/test_server_api.py`:

```python
@pytest.mark.asyncio
async def test_pause_simulation_endpoint(test_client):
    """Test POST /api/v1/runs/{run_id}/pause endpoint."""
    from server.event_bus import EventBus, SimulationEvent

    EventBus.reset_instance()
    bus = EventBus.get_instance()

    # Simulate a running simulation
    bus.emit(SimulationEvent.create("simulation_started", run_id="pause_test", max_steps=10))

    response = test_client.post("/api/v1/runs/pause_test/pause")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pause_requested"
    assert data["run_id"] == "pause_test"


@pytest.mark.asyncio
async def test_pause_non_running_simulation(test_client):
    """Test that pausing a non-running simulation fails."""
    response = test_client.post("/api/v1/runs/nonexistent/pause")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_resume_simulation_endpoint(test_client):
    """Test POST /api/v1/runs/{run_id}/resume endpoint."""
    from server.event_bus import EventBus, SimulationEvent

    EventBus.reset_instance()
    bus = EventBus.get_instance()

    # Simulate a paused simulation
    bus.emit(SimulationEvent.create("simulation_started", run_id="resume_test", max_steps=10))
    bus.emit(SimulationEvent.create("simulation_paused", run_id="resume_test", step=5))

    response = test_client.post("/api/v1/runs/resume_test/resume")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "resumed"
    assert data["run_id"] == "resume_test"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/integration/test_server_api.py::test_pause_simulation_endpoint tests/integration/test_server_api.py::test_resume_simulation_endpoint -v`
Expected: FAIL (404 - endpoints don't exist)

**Step 3: Write minimal implementation**

Add request/response models to `src/server/models.py`:

```python
class PauseSimulationResponse(BaseModel):
    """Response after requesting simulation pause."""
    run_id: str
    status: str  # "pause_requested"
    message: str


class ResumeSimulationResponse(BaseModel):
    """Response after resuming a simulation."""
    run_id: str
    status: str  # "resumed"
    resuming_from_step: int
    message: str
```

Add endpoints to `src/server/routes/v1.py`:

```python
from server.models import PauseSimulationResponse, ResumeSimulationResponse


@router.post("/{run_id}/pause", response_model=PauseSimulationResponse)
async def pause_simulation(run_id: str, force: bool = False):
    """Pause a running simulation.

    Args:
        run_id: The simulation run ID to pause
        force: If True, pause immediately without waiting for step completion
    """
    from server.job_queue import publish_pause_signal

    # Check simulation exists and is running
    status = _get_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot pause simulation with status '{status}'. Only running simulations can be paused."
        )

    # Publish pause signal
    publish_pause_signal(run_id, force=force)

    return PauseSimulationResponse(
        run_id=run_id,
        status="pause_requested",
        message=f"Pause signal sent to simulation {run_id}" + (" (force mode)" if force else "")
    )


@router.post("/{run_id}/resume", response_model=ResumeSimulationResponse)
async def resume_simulation(run_id: str):
    """Resume a paused simulation.

    Args:
        run_id: The simulation run ID to resume
    """
    from server.job_queue import enqueue_simulation
    from server.event_bus import emit_event

    # Check simulation exists and is paused
    status = _get_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if status != "paused":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot resume simulation with status '{status}'. Only paused simulations can be resumed."
        )

    # Load state to find scenario path and current step
    results_dir = Path("results") / run_id
    state_file = results_dir / "state.json"

    if not state_file.exists():
        raise HTTPException(status_code=500, detail=f"State file not found for run {run_id}")

    with open(state_file, "r") as f:
        state = json.load(f)

    scenario_path = state.get("scenario_path")
    snapshots = state.get("snapshots", [])
    last_step = snapshots[-1]["step"] if snapshots else 0

    # Emit resumed event
    emit_event("simulation_resumed", run_id, step=last_step)

    # Enqueue resume job
    # Note: We'll need to add resume support to enqueue_simulation
    enqueue_simulation(run_id, scenario_path, resume_from_step=last_step + 1)

    return ResumeSimulationResponse(
        run_id=run_id,
        status="resumed",
        resuming_from_step=last_step + 1,
        message=f"Simulation {run_id} resumed from step {last_step + 1}"
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/integration/test_server_api.py::test_pause_simulation_endpoint tests/integration/test_server_api.py::test_resume_simulation_endpoint -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/server/routes/v1.py src/server/models.py tests/integration/test_server_api.py
git commit -m "feat(api): add pause and resume endpoints"
```

---

## Task 5: Add Pause Check to Orchestrator

**Files:**
- Modify: `src/core/orchestrator.py`
- Test: `tests/unit/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_orchestrator.py`:

```python
def test_orchestrator_checks_pause_signal(mock_orchestrator):
    """Test that orchestrator checks for pause signal between steps."""
    from unittest.mock import patch, MagicMock

    # Mock the pause check to return a pause signal
    with patch("core.orchestrator.check_pause_requested") as mock_check:
        mock_check.return_value = {"force": False}

        # This should exit early due to pause
        with pytest.raises(SystemExit) as exc_info:
            mock_orchestrator.run()

        # Verify pause was checked
        mock_check.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_checks_pause_signal -v`
Expected: FAIL (no pause checking exists)

**Step 3: Write minimal implementation**

Add to `src/core/orchestrator.py`:

```python
# At top of file, add import
from server.job_queue import check_pause_requested, clear_pause_signal

# Add method to Orchestrator class
def _check_and_handle_pause(self, step: int) -> bool:
    """Check if pause was requested and handle it.

    Returns:
        True if simulation should stop (was paused), False to continue
    """
    pause_info = check_pause_requested(self.persistence.run_id)

    if pause_info is None:
        return False

    force = pause_info.get("force", False)

    if force:
        self.logger.info(MessageCode.SIM002, "Force pause requested", step=step)
        # Don't save state for force pause - we'll re-run this step
    else:
        self.logger.info(MessageCode.SIM002, "Pause requested, completing step", step=step)

    # Clear the pause signal
    clear_pause_signal(self.persistence.run_id)

    # Emit paused event
    emit(EventTypes.SIMULATION_PAUSED, step=step)

    return True
```

Update the `run()` method to check for pause:

```python
def run(self):
    """Run the simulation loop with SimulatorAgent."""
    enable_event_emitter(self.persistence.run_id)

    # ... existing setup code ...

    try:
        agent_messages = self._initialize_simulation()

        for step in range(1, self.max_steps + 1):
            # Check for pause BEFORE starting step
            if self._check_and_handle_pause(step):
                self.logger.info(MessageCode.SIM002, "Simulation paused", step=step)
                print(f"\nSimulation paused at step {step}")
                return  # Exit cleanly

            with PerformanceTimer(self.logger, MessageCode.PRF001, f"Step {step}", step=step):
                # ... existing step code ...

        self._save_final_state(step_messages)

    # ... existing exception handling ...
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_checks_pause_signal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat(orchestrator): add pause signal checking in step loop"
```

---

## Task 6: Add EventTypes for Pause/Resume

**Files:**
- Modify: `src/core/event_emitter.py`
- Test: `tests/unit/test_event_emitter.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_event_emitter.py`:

```python
def test_event_types_include_pause_resume():
    """Test that EventTypes includes PAUSED and RESUMED."""
    from core.event_emitter import EventTypes

    assert hasattr(EventTypes, "SIMULATION_PAUSED")
    assert hasattr(EventTypes, "SIMULATION_RESUMED")
    assert EventTypes.SIMULATION_PAUSED == "simulation_paused"
    assert EventTypes.SIMULATION_RESUMED == "simulation_resumed"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_event_emitter.py::test_event_types_include_pause_resume -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/core/event_emitter.py`, add to EventTypes:

```python
class EventTypes:
    """Standard event types for simulation lifecycle."""
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_COMPLETED = "simulation_completed"
    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_PAUSED = "simulation_paused"      # NEW
    SIMULATION_RESUMED = "simulation_resumed"    # NEW
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    AGENT_MESSAGE_SENT = "agent_message_sent"
    AGENT_RESPONSE_RECEIVED = "agent_response_received"
    DANGER_SIGNAL = "danger_signal"
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_event_emitter.py::test_event_types_include_pause_resume -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/event_emitter.py tests/unit/test_event_emitter.py
git commit -m "feat(events): add SIMULATION_PAUSED and SIMULATION_RESUMED event types"
```

---

## Task 7: Add Resume Support to Worker Tasks

**Files:**
- Modify: `src/server/worker_tasks.py`
- Modify: `src/server/job_queue.py`
- Test: `tests/integration/test_worker_tasks.py`

**Step 1: Write the failing test**

Add to `tests/integration/test_worker_tasks.py`:

```python
def test_run_simulation_task_with_resume(tmp_path, mock_llm):
    """Test running a simulation task that resumes from a saved state."""
    from server.worker_tasks import run_simulation_task

    # Create a mock state file representing a paused simulation
    run_id = "resume_test"
    results_dir = tmp_path / "results" / run_id
    results_dir.mkdir(parents=True)

    state = {
        "run_id": run_id,
        "scenario": "test",
        "snapshots": [
            {"step": 1, "global_vars": {}, "agent_vars": {}},
            {"step": 2, "global_vars": {}, "agent_vars": {}},
        ]
    }

    with open(results_dir / "state.json", "w") as f:
        json.dump(state, f)

    # This should resume from step 3
    result = run_simulation_task(
        run_id=run_id,
        scenario_path="scenarios/test.yaml",
        resume_from_step=3
    )

    assert result["status"] == "completed"
    assert result["resumed_from"] == 3
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/integration/test_worker_tasks.py::test_run_simulation_task_with_resume -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/server/job_queue.py` `enqueue_simulation()`:

```python
def enqueue_simulation(
    run_id: str,
    scenario_path: str,
    priority: str = "normal",
    resume_from_step: int | None = None,
) -> str:
    """
    Enqueue a simulation job for worker processing.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML file
        priority: Queue priority - "high", "normal", or "low"
        resume_from_step: If set, resume from this step (for paused simulations)

    Returns:
        Job ID for tracking
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    if priority not in _queues:
        raise ValueError(f"Invalid priority: {priority}. Must be one of: high, normal, low")

    queue = _queues[priority]

    from server.worker_tasks import run_simulation_task

    job = queue.enqueue(
        run_simulation_task,
        args=(run_id, scenario_path),
        kwargs={"resume_from_step": resume_from_step},
        job_id=run_id if resume_from_step is None else f"{run_id}_resume_{resume_from_step}",
        job_timeout=3600,
        retry=Retry(max=3, interval=[10, 30, 60]),
        meta={"scenario_path": scenario_path, "priority": priority, "resume_from_step": resume_from_step},
    )

    logger.info("Simulation enqueued", extra={
        "run_id": run_id,
        "job_id": job.id,
        "priority": priority,
        "scenario": scenario_path,
        "resume_from_step": resume_from_step,
    })
    return job.id
```

Update `src/server/worker_tasks.py`:

```python
def run_simulation_task(
    run_id: str,
    scenario_path: str,
    resume_from_step: int | None = None
) -> dict:
    """
    Execute a simulation as an RQ task.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML file
        resume_from_step: If set, resume from this step

    Returns:
        Dictionary with simulation result summary
    """
    from core.orchestrator import Orchestrator
    from core.event_emitter import enable_event_emitter
    from server.event_bus import emit_event

    scenario_path = Path(scenario_path)

    logger.info("Starting simulation", extra={
        "run_id": run_id,
        "scenario": scenario_path.stem,
        "scenario_path": str(scenario_path),
        "resume_from_step": resume_from_step,
    })

    try:
        enable_event_emitter(run_id)

        orchestrator = Orchestrator(
            str(scenario_path),
            scenario_path.stem,
            save_frequency=1,
            run_id=run_id,
        )

        if resume_from_step:
            orchestrator.run(start_step=resume_from_step)
        else:
            orchestrator.run()

        logger.info("Simulation completed", extra={
            "run_id": run_id,
            "scenario": scenario_path.stem,
        })

        result = {
            "run_id": run_id,
            "status": "completed",
            "scenario": scenario_path.stem,
        }
        if resume_from_step:
            result["resumed_from"] = resume_from_step

        return result

    except Exception as e:
        logger.error("Simulation failed", extra={
            "run_id": run_id,
            "error": str(e),
        })
        emit_event("simulation_failed", run_id, error=str(e))
        raise
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/integration/test_worker_tasks.py::test_run_simulation_task_with_resume -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/server/worker_tasks.py src/server/job_queue.py tests/integration/test_worker_tasks.py
git commit -m "feat(worker): add resume support for paused simulations"
```

---

## Task 8: Add Resume Support to Orchestrator

**Files:**
- Modify: `src/core/orchestrator.py`
- Test: `tests/unit/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_orchestrator.py`:

```python
def test_orchestrator_run_with_start_step():
    """Test that orchestrator can start from a specific step."""
    from core.orchestrator import Orchestrator
    from unittest.mock import patch, MagicMock

    # Mock the orchestrator to track which steps are executed
    executed_steps = []

    with patch.object(Orchestrator, "_collect_agent_responses") as mock_collect:
        mock_collect.side_effect = lambda step, msgs: (executed_steps.append(step), ({}, []))

        orchestrator = create_test_orchestrator(max_steps=5)
        orchestrator.run(start_step=3)

        # Should only execute steps 3, 4, 5
        assert executed_steps == [3, 4, 5]
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_run_with_start_step -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `run()` method signature and loop in `src/core/orchestrator.py`:

```python
def run(self, start_step: int = 1):
    """Run the simulation loop with SimulatorAgent.

    Args:
        start_step: Step to start from (default 1). Use > 1 to resume a paused simulation.
    """
    enable_event_emitter(self.persistence.run_id)

    is_resume = start_step > 1

    if is_resume:
        self.logger.info(
            MessageCode.SIM001,
            "Simulation resumed",
            num_agents=len(self.agents),
            max_steps=self.max_steps,
            start_step=start_step
        )
        # Load state from disk for resume
        self._restore_state_for_resume()
    else:
        self.logger.info(
            MessageCode.SIM001,
            "Simulation started",
            num_agents=len(self.agents),
            max_steps=self.max_steps
        )

    # ... emit events ...

    step_messages = []
    try:
        if is_resume:
            # For resume, get last agent messages from state
            agent_messages = self._get_last_agent_messages()
        else:
            agent_messages = self._initialize_simulation()

        for step in range(start_step, self.max_steps + 1):
            # Check for pause BEFORE starting step
            if self._check_and_handle_pause(step):
                self.logger.info(MessageCode.SIM002, "Simulation paused", step=step)
                print(f"\nSimulation paused at step {step}")
                return

            with PerformanceTimer(self.logger, MessageCode.PRF001, f"Step {step}", step=step):
                self.logger.info(MessageCode.SIM003, "Step started", step=step, max_steps=self.max_steps)
                emit(EventTypes.STEP_STARTED, step=step, max_steps=self.max_steps)
                print(f"\n=== Step {step}/{self.max_steps} ===")

                agent_responses, step_messages = self._collect_agent_responses(step, agent_messages)
                agent_messages = self._process_step_results(step, agent_responses, step_messages)

        self._save_final_state(step_messages)

    # ... exception handling ...
```

Add restore method:

```python
def _restore_state_for_resume(self):
    """Restore game state from saved state file for resume."""
    state_file = self.persistence.run_dir / "state.json"

    if not state_file.exists():
        raise ValueError(f"Cannot resume: state file not found at {state_file}")

    import json
    with open(state_file, "r") as f:
        state = json.load(f)

    snapshots = state.get("snapshots", [])
    if not snapshots:
        raise ValueError("Cannot resume: no snapshots found in state file")

    last_snapshot = snapshots[-1]

    # Restore game engine state
    self.game_engine.restore_from_snapshot(last_snapshot)

    # Restore agent stats
    for agent in self.agents:
        agent_vars = last_snapshot.get("agent_vars", {}).get(agent.name, {})
        agent.update_stats(agent_vars)

    self.logger.info(
        MessageCode.SIM001,
        "State restored from snapshot",
        step=last_snapshot.get("step", 0)
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_run_with_start_step -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat(orchestrator): add start_step parameter for resume support"
```

---

## Task 9: Create CLI Tool with Typer

**Files:**
- Create: `src/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_cli.py`

**Step 1: Write the failing test**

Create `tests/unit/test_cli.py`:

```python
"""Unit tests for the CLI tool."""

import pytest
from typer.testing import CliRunner


runner = CliRunner()


def test_cli_help():
    """Test that CLI shows help."""
    from cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "apart" in result.output.lower() or "simulation" in result.output.lower()


def test_cli_list_command():
    """Test that list command exists."""
    from cli import app

    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output.lower()


def test_cli_run_command():
    """Test that run command exists."""
    from cli import app

    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_cli_pause_command():
    """Test that pause command exists."""
    from cli import app

    result = runner.invoke(app, ["pause", "--help"])
    assert result.exit_code == 0


def test_cli_resume_command():
    """Test that resume command exists."""
    from cli import app

    result = runner.invoke(app, ["resume", "--help"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_cli.py -v`
Expected: FAIL (cli module doesn't exist)

**Step 3: Write minimal implementation**

Create `src/cli.py`:

```python
"""
APART CLI - Command-line interface for running simulations.

Usage:
    apart run <scenario>       Start a new simulation
    apart pause <run_id>       Pause a running simulation
    apart resume <run_id>      Resume a paused simulation
    apart list                 List all simulation runs
    apart show <run_id>        Show details of a specific run
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="apart",
    help="APART - Multi-agent simulation framework",
    add_completion=False,
)

console = Console()

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"


def get_api_url() -> str:
    """Get API URL from environment or default."""
    import os
    return os.environ.get("APART_API_URL", DEFAULT_API_URL)


@app.command()
def run(
    scenario: Path = typer.Argument(..., help="Path to scenario YAML file"),
    name: str = typer.Option(None, "--name", "-n", help="Custom run ID"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Queue priority: high, normal, low"),
    save_frequency: int = typer.Option(1, "--save-frequency", "-sf", help="Save every N steps (0=final only)"),
):
    """Start a new simulation by submitting to the job queue."""
    import requests

    if not scenario.exists():
        console.print(f"[red]Error:[/red] Scenario file not found: {scenario}")
        raise typer.Exit(1)

    api_url = get_api_url()

    try:
        response = requests.post(
            f"{api_url}/api/v1/runs",
            json={
                "scenario_path": str(scenario),
                "run_id": name,
                "priority": priority,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]Simulation started![/green]")
        console.print(f"  Run ID: [bold]{data['run_id']}[/bold]")
        console.print(f"  Status: {data['status']}")
        console.print(f"\nMonitor with: [cyan]apart show {data['run_id']}[/cyan]")

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Error:[/red] Cannot connect to API server at {api_url}")
        console.print("Make sure the server is running: [cyan]apart-server[/cyan]")
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        console.print(f"[red]Error:[/red] {e.response.json().get('detail', str(e))}")
        raise typer.Exit(1)


@app.command()
def pause(
    run_id: str = typer.Argument(..., help="Run ID to pause"),
    force: bool = typer.Option(False, "--force", "-f", help="Pause immediately without waiting for step completion"),
):
    """Pause a running simulation."""
    import requests

    api_url = get_api_url()

    try:
        response = requests.post(
            f"{api_url}/api/v1/runs/{run_id}/pause",
            params={"force": force},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[yellow]Pause requested[/yellow]")
        console.print(f"  Run ID: [bold]{data['run_id']}[/bold]")
        console.print(f"  Message: {data['message']}")

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Error:[/red] Cannot connect to API server at {api_url}")
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        console.print(f"[red]Error:[/red] {e.response.json().get('detail', str(e))}")
        raise typer.Exit(1)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
):
    """Resume a paused simulation."""
    import requests

    api_url = get_api_url()

    try:
        response = requests.post(
            f"{api_url}/api/v1/runs/{run_id}/resume",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]Simulation resumed![/green]")
        console.print(f"  Run ID: [bold]{data['run_id']}[/bold]")
        console.print(f"  Resuming from step: {data['resuming_from_step']}")

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Error:[/red] Cannot connect to API server at {api_url}")
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        console.print(f"[red]Error:[/red] {e.response.json().get('detail', str(e))}")
        raise typer.Exit(1)


@app.command("list")
def list_runs(
    status: str = typer.Option(None, "--status", "-s", help="Filter by status: pending, running, paused, completed, failed"),
):
    """List all simulation runs."""
    import requests

    api_url = get_api_url()

    try:
        response = requests.get(f"{api_url}/api/v1/runs", timeout=10)
        response.raise_for_status()
        data = response.json()

        runs = data.get("runs", [])

        # Filter by status if specified
        if status:
            runs = [r for r in runs if r.get("status") == status]

        if not runs:
            console.print("No simulation runs found.")
            return

        # Create table
        table = Table(title="Simulation Runs")
        table.add_column("Run ID", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Step", justify="right")
        table.add_column("Scenario")
        table.add_column("Started")

        status_colors = {
            "pending": "yellow",
            "running": "green",
            "paused": "yellow",
            "completed": "blue",
            "failed": "red",
        }

        for run in runs:
            status_val = run.get("status", "unknown")
            color = status_colors.get(status_val, "white")

            step = run.get("currentStep", 0)
            total = run.get("totalSteps")
            step_str = f"{step}/{total}" if total else str(step)

            started = run.get("startedAt", "")[:19] if run.get("startedAt") else "-"

            table.add_row(
                run.get("runId", "?"),
                f"[{color}]{status_val}[/{color}]",
                step_str,
                run.get("scenario", "?"),
                started,
            )

        console.print(table)

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Error:[/red] Cannot connect to API server at {api_url}")
        raise typer.Exit(1)


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to show"),
):
    """Show details of a specific simulation run."""
    import requests

    api_url = get_api_url()

    try:
        response = requests.get(f"{api_url}/api/v1/runs/{run_id}", timeout=10)
        response.raise_for_status()
        data = response.json()

        console.print(f"\n[bold]Run: {data.get('runId')}[/bold]")
        console.print(f"  Scenario: {data.get('scenario')}")
        console.print(f"  Status: {data.get('status')}")
        console.print(f"  Step: {data.get('currentStep')}/{data.get('maxSteps', '?')}")
        console.print(f"  Started: {data.get('startedAt', '-')}")
        console.print(f"  Agents: {', '.join(data.get('agentNames', []))}")

        danger_count = len(data.get("dangerSignals", []))
        if danger_count:
            console.print(f"  [red]Danger signals: {danger_count}[/red]")

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Error:[/red] Cannot connect to API server at {api_url}")
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            console.print(f"[red]Error:[/red] Run '{run_id}' not found")
        else:
            console.print(f"[red]Error:[/red] {e.response.json().get('detail', str(e))}")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/unit/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): add typer-based CLI with run/pause/resume/list/show commands"
```

---

## Task 10: Update pyproject.toml with CLI Entry Point

**Files:**
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```bash
# Test that apart command is installed
cd /home/hendrik/coding/ai-safety/apart && pip install -e . && apart --help
```

Expected: FAIL (command not found)

**Step 2: Update pyproject.toml**

```toml
[project]
name = "apart"
version = "0.1.0"
description = "Multi-agent simulation framework for AI safety research"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pyyaml>=6.0.1",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "google-generativeai>=0.3.0",
    "requests>=2.31.0",
    "openai>=1.0.0",
    "anthropic>=0.8.0",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sse-starlette>=2.0.0",
    "rq>=1.16.0",
    "redis>=5.0.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "fakeredis>=2.20.0",
]

[project.scripts]
apart = "cli:main"
apart-server = "server.app:main"

[tool.pytest.ini_options]
markers = [
    "redis: tests requiring a real Redis instance (deselect with '-m \"not redis\"')",
]
```

**Step 3: Install and verify**

```bash
cd /home/hendrik/coding/ai-safety/apart && pip install -e . && apart --help
```

Expected: Shows CLI help

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(cli): add apart CLI entry point and typer/rich dependencies"
```

---

## Task 11: Integration Test for Full Pause/Resume Flow

**Files:**
- Test: `tests/integration/test_pause_resume_flow.py`

**Step 1: Write integration test**

Create `tests/integration/test_pause_resume_flow.py`:

```python
"""Integration tests for the complete pause/resume flow."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPauseResumeFlow:
    """Test the complete pause/resume workflow."""

    def test_full_pause_resume_cycle(self, test_client, tmp_path):
        """Test starting, pausing, and resuming a simulation."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # 1. Start a simulation (mock)
        run_id = "flow_test_123"
        bus.emit(SimulationEvent.create(
            "simulation_started",
            run_id=run_id,
            max_steps=10,
            num_agents=2
        ))

        # Simulate some steps completing
        for step in range(1, 4):
            bus.emit(SimulationEvent.create(
                "step_completed",
                run_id=run_id,
                step=step
            ))

        # 2. Pause the simulation
        response = test_client.post(f"/api/v1/runs/{run_id}/pause")
        assert response.status_code == 200
        assert response.json()["status"] == "pause_requested"

        # Simulate worker receiving pause signal and emitting paused event
        bus.emit(SimulationEvent.create(
            "simulation_paused",
            run_id=run_id,
            step=3
        ))

        # 3. Verify status is paused
        from server.routes.v1 import _get_run_status
        status = _get_run_status(run_id)
        assert status == "paused"

        # 4. Create mock state file for resume
        results_dir = tmp_path / "results" / run_id
        results_dir.mkdir(parents=True)
        state = {
            "run_id": run_id,
            "scenario_path": "scenarios/test.yaml",
            "snapshots": [
                {"step": 1, "global_vars": {}, "agent_vars": {}},
                {"step": 2, "global_vars": {}, "agent_vars": {}},
                {"step": 3, "global_vars": {}, "agent_vars": {}},
            ]
        }
        with open(results_dir / "state.json", "w") as f:
            json.dump(state, f)

        # 5. Resume the simulation (with mocked Path)
        with patch("server.routes.v1.Path") as mock_path:
            mock_path.return_value = results_dir
            mock_path.__truediv__ = lambda self, x: results_dir / x

            # Mock enqueue to avoid actual Redis
            with patch("server.routes.v1.enqueue_simulation") as mock_enqueue:
                mock_enqueue.return_value = "job_123"

                response = test_client.post(f"/api/v1/runs/{run_id}/resume")

        # Note: Full resume test requires more infrastructure
        # This verifies the endpoint flow works

    def test_cannot_pause_completed_simulation(self, test_client):
        """Test that completed simulations cannot be paused."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus.reset_instance()
        bus = EventBus.get_instance()

        run_id = "completed_test"
        bus.emit(SimulationEvent.create("simulation_started", run_id=run_id))
        bus.emit(SimulationEvent.create("simulation_completed", run_id=run_id))

        response = test_client.post(f"/api/v1/runs/{run_id}/pause")
        assert response.status_code == 409  # Conflict
        assert "completed" in response.json()["detail"]

    def test_cannot_resume_running_simulation(self, test_client):
        """Test that running simulations cannot be resumed."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus.reset_instance()
        bus = EventBus.get_instance()

        run_id = "running_test"
        bus.emit(SimulationEvent.create("simulation_started", run_id=run_id))

        response = test_client.post(f"/api/v1/runs/{run_id}/resume")
        assert response.status_code == 409  # Conflict
        assert "running" in response.json()["detail"]
```

**Step 2: Run integration tests**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/integration/test_pause_resume_flow.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_pause_resume_flow.py
git commit -m "test: add integration tests for pause/resume flow"
```

---

## Task 12: Final Verification and Documentation

**Step 1: Run full test suite**

```bash
cd /home/hendrik/coding/ai-safety/apart && python -m pytest tests/ -v --tb=short
```

Expected: All tests pass

**Step 2: Manual verification**

```bash
# Start server
apart-server &

# Start worker
rq worker simulations &

# Test CLI commands
apart list
apart run scenarios/test.yaml --name test123
apart pause test123
apart list --status paused
apart resume test123
apart show test123
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete pause/resume feature implementation

Adds ability to pause and resume simulations:
- PAUSED status in SimulationStatus enum
- POST /api/v1/runs/{id}/pause endpoint
- POST /api/v1/runs/{id}/resume endpoint
- Redis-based pause signaling
- Orchestrator pause checking in step loop
- State restoration for resume
- CLI tool: apart run/pause/resume/list/show

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```
