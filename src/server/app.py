"""
FastAPI application for the Apart Dashboard.

Provides:
- REST API for simulation management
- SSE event streaming for real-time updates
- Optional Redis job queue for distributed deployments
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from server.event_bus import get_event_bus
from server.models import (
    SimulationSummary,
    SimulationDetails,
    SimulationStatus,
    StartSimulationRequest,
    StartSimulationResponse,
)

# Check if job queue mode is enabled
USE_JOB_QUEUE = os.environ.get("APART_USE_JOB_QUEUE", "").lower() in ("1", "true", "yes")


# ============================================================================
# Execution infrastructure
# ============================================================================

# Fixed pool size prevents unbounded thread creation
MAX_CONCURRENT_SIMULATIONS = 4
_executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_SIMULATIONS,
    thread_name_prefix="sim-"
)

# Semaphore to limit concurrent simulations (matches executor size)
_sim_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SIMULATIONS)

# Track running simulations: run_id -> {"future": Future, "status": str}
_running_simulations: dict[str, dict] = {}
_simulations_lock = Lock()

# Shutdown coordination
_shutdown_requested = False


def _graceful_shutdown():
    """Shut down the executor gracefully."""
    global _shutdown_requested
    _shutdown_requested = True

    # Cancel pending futures (running ones will complete)
    with _simulations_lock:
        for run_id, sim_info in _running_simulations.items():
            future = sim_info.get("future")
            if future and not future.done():
                future.cancel()

    # Wait for running tasks to complete (with timeout)
    _executor.shutdown(wait=True, cancel_futures=True)


def _initialize_database():
    """Initialize database if database mode is enabled."""
    if os.environ.get("APART_USE_DATABASE", "").lower() in ("1", "true", "yes"):
        from server.database import init_db
        from server.event_bus import EventBus
        EventBus._use_database = True
        init_db()


def _generate_mock_data_if_empty():
    """Generate mock data if results/ directory is empty.

    This provides sample data for development/demo purposes.
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Check if there are any run directories
    existing_runs = [
        d for d in results_dir.iterdir()
        if d.is_dir() and (d.name.startswith("run_") or d.name.startswith("mock_run_"))
    ]

    if existing_runs:
        return  # Already have data

    print("No simulation runs found. Generating mock data for development...")

    try:
        import subprocess
        import sys

        script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_mock_data.py"
        if script_path.exists():
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            if result.returncode == 0:
                print("Mock data generated successfully.")
            else:
                print(f"Failed to generate mock data: {result.stderr}")
        else:
            print(f"Mock data script not found at {script_path}")
    except Exception as e:
        print(f"Error generating mock data: {e}")


def _initialize_job_queue():
    """Initialize Redis job queue if job queue mode is enabled."""
    if USE_JOB_QUEUE:
        from server.job_queue import init_job_queue
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        init_job_queue(redis_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    _initialize_database()
    _initialize_job_queue()
    _generate_mock_data_if_empty()
    yield
    # Shutdown
    _graceful_shutdown()


app = FastAPI(
    title="Apart Dashboard API",
    description="API for monitoring AI safety simulations",
    version="0.1.0",
    docs_url="/api/docs",
    lifespan=lifespan,
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/health/detailed")
async def detailed_health():
    """Detailed health check with system metrics."""
    from server.event_bus import EventBus

    event_bus = get_event_bus()

    with _simulations_lock:
        active_sims = sum(1 for s in _running_simulations.values() if s.get("status") == "running")
        total_tracked = len(_running_simulations)

    result = {
        "status": "healthy" if not _shutdown_requested else "shutting_down",
        "active_simulations": active_sims,
        "total_tracked_simulations": total_tracked,
        "max_concurrent_simulations": MAX_CONCURRENT_SIMULATIONS,
        "event_bus_subscribers": len(event_bus._subscribers),
        "total_run_ids": len(event_bus.get_all_run_ids()),
        "persistence_mode": "database" if EventBus._use_database else "jsonl",
    }

    # Add database stats if using database mode
    if EventBus._use_database:
        try:
            from server.database import get_db
            db = get_db()
            result["database_stats"] = db.get_stats()
        except Exception:
            result["database_stats"] = {"error": "unavailable"}

    return result


@app.get("/api/runs")
async def list_runs():
    """List all simulation runs by scanning results/ directory and merging with EventBus data.

    Returns runs from both:
    - results/ directory (completed/historical runs)
    - EventBus (in-memory, currently active runs)
    """
    import json
    from pathlib import Path

    event_bus = get_event_bus()
    runs_by_id: dict[str, dict] = {}

    # 1. Scan results/ directory for persisted runs
    results_dir = Path("results")
    if results_dir.exists():
        for run_dir in results_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Support both real runs (run_*) and mock runs (mock_run_*)
            if not (run_dir.name.startswith("run_") or run_dir.name.startswith("mock_run_")):
                continue

            state_file = run_dir / "state.json"
            if not state_file.exists():
                continue

            try:
                with open(state_file, "r") as f:
                    state = json.load(f)

                run_id = state.get("run_id", run_dir.name)
                scenario = state.get("scenario", "Unknown")
                started_at = state.get("started_at")
                snapshots = state.get("snapshots", [])

                # Determine status and step from snapshots
                current_step = 0
                total_steps = None
                danger_count = 0

                if snapshots:
                    last_snapshot = snapshots[-1]
                    current_step = last_snapshot.get("step", 0)
                    # Check for danger signals in snapshots
                    for snapshot in snapshots:
                        game_state = snapshot.get("game_state", {})
                        if isinstance(game_state, dict):
                            dangers = game_state.get("danger_signals", [])
                            danger_count += len(dangers) if isinstance(dangers, list) else 0

                # Default to completed if we have snapshots
                status = "completed" if snapshots else "pending"

                runs_by_id[run_id] = {
                    "runId": run_id,
                    "scenario": scenario,
                    "status": status,
                    "currentStep": current_step,
                    "totalSteps": total_steps,
                    "startedAt": started_at,
                    "completedAt": None,  # Not tracked in state.json
                    "dangerCount": danger_count,
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip corrupted files
                continue

    # 2. Merge with EventBus data (for real-time status updates)
    for run_id in event_bus.get_all_run_ids():
        history = event_bus.get_history(run_id)

        status = "pending"
        current_step = 0
        total_steps = None
        started_at = None
        completed_at = None
        scenario_name = None
        danger_count = 0

        for event in history:
            if event.event_type == "simulation_started":
                status = "running"
                started_at = event.timestamp
                total_steps = event.data.get("max_steps")
                scenario_name = event.data.get("scenario_name")
            elif event.event_type == "step_completed":
                current_step = event.step or 0
            elif event.event_type == "danger_signal":
                danger_count += 1
            elif event.event_type == "simulation_completed":
                status = "completed"
                completed_at = event.timestamp
            elif event.event_type == "simulation_failed":
                status = "failed"
                completed_at = event.timestamp

        # Update or create entry (EventBus has more recent data)
        if run_id in runs_by_id:
            # Merge: EventBus has live status info
            runs_by_id[run_id].update({
                "status": status,
                "currentStep": current_step,
                "totalSteps": total_steps or runs_by_id[run_id].get("totalSteps"),
                "completedAt": completed_at,
                "dangerCount": max(danger_count, runs_by_id[run_id].get("dangerCount", 0)),
            })
            if scenario_name:
                runs_by_id[run_id]["scenario"] = scenario_name
        else:
            runs_by_id[run_id] = {
                "runId": run_id,
                "scenario": scenario_name or run_id,
                "status": status,
                "currentStep": current_step,
                "totalSteps": total_steps,
                "startedAt": started_at,
                "completedAt": completed_at,
                "dangerCount": danger_count,
            }

    # Sort by start time (most recent first)
    runs_list = sorted(
        runs_by_id.values(),
        key=lambda r: r.get("startedAt") or "",
        reverse=True
    )

    return {"runs": runs_list}


@app.get("/api/runs/{run_id}")
async def get_run_detail(run_id: str):
    """Get full state data for a specific run from disk.

    This reads the state.json file directly, providing historical data
    even when the EventBus doesn't have the events in memory.
    """
    import json
    from pathlib import Path

    results_dir = Path("results")
    run_dir = results_dir / run_id
    state_file = run_dir / "state.json"

    if not state_file.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        # Transform to frontend-expected format
        snapshots = state.get("snapshots", [])

        # Extract messages from all snapshots
        messages = []
        for snapshot in snapshots:
            step = snapshot.get("step", 0)
            for msg in snapshot.get("messages", []):
                if msg.get("from") == "orchestrator":
                    messages.append({
                        "step": step,
                        "timestamp": state.get("started_at", ""),
                        "agentName": msg.get("to", "unknown"),
                        "direction": "sent",
                        "content": msg.get("content", ""),
                    })
                else:
                    messages.append({
                        "step": step,
                        "timestamp": state.get("started_at", ""),
                        "agentName": msg.get("from", "unknown"),
                        "direction": "received",
                        "content": msg.get("content", ""),
                    })

        # Extract danger signals
        danger_signals = []
        for snapshot in snapshots:
            step = snapshot.get("step", 0)
            game_state = snapshot.get("game_state", {})
            for signal in game_state.get("danger_signals", []):
                danger_signals.append({
                    "step": signal.get("step", step),
                    "timestamp": signal.get("timestamp", ""),
                    "category": signal.get("category", "unknown"),
                    "agentName": signal.get("agent_name"),
                    "metric": signal.get("metric", ""),
                    "value": signal.get("value", 0),
                    "threshold": signal.get("threshold"),
                })

        # Extract variable history
        global_vars_history = []
        agent_vars_history = {}
        agent_names = set()

        for snapshot in snapshots:
            step = snapshot.get("step", 0)

            if "global_vars" in snapshot:
                global_vars_history.append({
                    "step": step,
                    "values": snapshot["global_vars"],
                })

            if "agent_vars" in snapshot:
                for agent_name, vars in snapshot["agent_vars"].items():
                    agent_names.add(agent_name)
                    if agent_name not in agent_vars_history:
                        agent_vars_history[agent_name] = []
                    agent_vars_history[agent_name].append({
                        "step": step,
                        "values": vars,
                    })

        # Determine status
        current_step = snapshots[-1]["step"] if snapshots else 0

        # Check EventBus for live status
        event_bus = get_event_bus()
        history = event_bus.get_history(run_id)
        status = "completed"  # Default for disk-only runs
        for event in history:
            if event.event_type == "simulation_started":
                status = "running"
            elif event.event_type == "simulation_completed":
                status = "completed"
            elif event.event_type == "simulation_failed":
                status = "failed"

        # Try to get spatial graph from first snapshot's game_state or from EventBus
        spatial_graph = None
        for event in history:
            if event.event_type == "simulation_started" and event.data.get("spatial_graph"):
                spatial_graph = event.data["spatial_graph"]
                break

        # Also check snapshots for spatial data hints
        if not spatial_graph and snapshots:
            first_snapshot = snapshots[0]
            agent_vars = first_snapshot.get("agent_vars", {})
            # If agents have location data, we likely have a spatial scenario
            has_locations = any(
                "location" in vars
                for vars in agent_vars.values()
            )
            if has_locations:
                # Return a default spatial graph for mock data
                spatial_graph = {
                    "nodes": [
                        {"id": "taiwan", "name": "Taiwan", "type": "nation", "properties": {}, "conditions": []},
                        {"id": "china", "name": "China", "type": "nation", "properties": {}, "conditions": []},
                        {"id": "usa", "name": "United States", "type": "nation", "properties": {}, "conditions": []},
                        {"id": "taiwan_strait", "name": "Taiwan Strait", "type": "sea_zone", "properties": {}, "conditions": []},
                        {"id": "pacific", "name": "Pacific Ocean", "type": "sea_zone", "properties": {}, "conditions": []},
                        {"id": "taipei", "name": "Taipei", "type": "city", "properties": {}, "conditions": []},
                        {"id": "beijing", "name": "Beijing", "type": "city", "properties": {}, "conditions": []},
                    ],
                    "edges": [
                        {"from": "taiwan", "to": "taiwan_strait", "type": "maritime", "directed": False, "properties": {"distance_km": 100}},
                        {"from": "china", "to": "taiwan_strait", "type": "maritime", "directed": False, "properties": {"distance_km": 150}},
                        {"from": "taiwan_strait", "to": "pacific", "type": "maritime", "directed": False, "properties": {"distance_km": 500}},
                        {"from": "usa", "to": "pacific", "type": "maritime", "directed": False, "properties": {"distance_km": 8000}},
                        {"from": "taipei", "to": "taiwan", "type": "land", "directed": False, "properties": {}},
                        {"from": "beijing", "to": "china", "type": "land", "directed": False, "properties": {}},
                    ],
                    "blocked_edge_types": [],
                }

        return {
            "runId": run_id,
            "scenario": state.get("scenario", run_id),
            "status": status,
            "currentStep": current_step,
            "maxSteps": len(snapshots) if snapshots else None,
            "startedAt": state.get("started_at"),
            "agentNames": list(agent_names),
            "spatialGraph": spatial_graph,
            "messages": messages,
            "dangerSignals": danger_signals,
            "globalVarsHistory": global_vars_history,
            "agentVarsHistory": agent_vars_history,
        }

    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Error reading run data: {str(e)}")


@app.get("/api/simulations", response_model=list[SimulationSummary])
async def list_simulations():
    """List all simulation runs."""
    event_bus = get_event_bus()
    run_ids = event_bus.get_all_run_ids()

    summaries = []
    for run_id in run_ids:
        history = event_bus.get_history(run_id)

        # Determine status from events
        status = SimulationStatus.PENDING
        current_step = 0
        max_steps = None
        agent_count = 0
        scenario_name = None
        started_at = None
        completed_at = None

        for event in history:
            if event.event_type == "simulation_started":
                status = SimulationStatus.RUNNING
                started_at = event.timestamp
                max_steps = event.data.get("max_steps")
                agent_count = event.data.get("num_agents", 0)
            elif event.event_type == "step_completed":
                current_step = event.step or 0
            elif event.event_type == "simulation_completed":
                status = SimulationStatus.COMPLETED
                completed_at = event.timestamp
            elif event.event_type == "simulation_failed":
                status = SimulationStatus.FAILED
                completed_at = event.timestamp

        summaries.append(SimulationSummary(
            run_id=run_id,
            status=status,
            scenario_name=scenario_name,
            started_at=started_at,
            completed_at=completed_at,
            current_step=current_step,
            max_steps=max_steps,
            agent_count=agent_count
        ))

    return summaries


@app.get("/api/simulations/{run_id}", response_model=SimulationDetails)
async def get_simulation(run_id: str):
    """Get details for a specific simulation."""
    event_bus = get_event_bus()
    history = event_bus.get_history(run_id)

    if not history:
        raise HTTPException(status_code=404, detail=f"Simulation {run_id} not found")

    # Extract details from events
    status = SimulationStatus.PENDING
    current_step = 0
    max_steps = None
    agent_count = 0
    agent_names = []
    started_at = None
    completed_at = None
    error_message = None

    for event in history:
        if event.event_type == "simulation_started":
            status = SimulationStatus.RUNNING
            started_at = event.timestamp
            max_steps = event.data.get("max_steps")
            agent_names = event.data.get("agent_names", [])
            agent_count = len(agent_names)
        elif event.event_type == "step_completed":
            current_step = event.step or 0
        elif event.event_type == "simulation_completed":
            status = SimulationStatus.COMPLETED
            completed_at = event.timestamp
        elif event.event_type == "simulation_failed":
            status = SimulationStatus.FAILED
            completed_at = event.timestamp
            error_message = event.data.get("error")

    return SimulationDetails(
        run_id=run_id,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        current_step=current_step,
        max_steps=max_steps,
        agent_count=agent_count,
        agents=[{"name": name, "llm_provider": None, "llm_model": None} for name in agent_names],
        error_message=error_message
    )


def _run_simulation_sync(run_id: str, scenario_path: Path) -> None:
    """
    Synchronous simulation runner for executor.

    This function runs in a thread pool worker and handles the full
    simulation lifecycle including event emission.
    """
    from core.orchestrator import Orchestrator
    from core.event_emitter import enable_event_emitter
    from server.event_bus import emit_event

    try:
        # Enable event emission
        enable_event_emitter(run_id)

        # Run simulation
        orchestrator = Orchestrator(
            str(scenario_path),
            scenario_path.stem,
            save_frequency=1,
            run_id=run_id
        )
        orchestrator.run()
    except Exception as e:
        emit_event("simulation_failed", run_id, error=str(e))
    finally:
        # Clean up tracking
        with _simulations_lock:
            if run_id in _running_simulations:
                _running_simulations[run_id]["status"] = "completed"


@app.post("/api/simulations", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    """Start a new simulation.

    When APART_USE_JOB_QUEUE is enabled, jobs are queued to Redis for worker processing.
    Otherwise, uses a bounded thread pool to prevent resource exhaustion.
    """
    import uuid

    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")

    # Validate scenario path
    scenario_path = Path(request.scenario_path)
    if not scenario_path.exists():
        raise HTTPException(status_code=400, detail=f"Scenario not found: {request.scenario_path}")

    # Generate run ID
    run_id = request.run_id or str(uuid.uuid4())[:8]

    # Job queue mode: enqueue to Redis
    if USE_JOB_QUEUE:
        from server.job_queue import enqueue_simulation

        priority = request.priority.value if request.priority else "normal"
        job_id = enqueue_simulation(run_id, str(scenario_path), priority)

        return StartSimulationResponse(
            run_id=run_id,
            status=SimulationStatus.PENDING,
            message=f"Simulation queued (job_id: {job_id})"
        )

    # ThreadPool mode: run locally
    # Check if we can accept more simulations (non-blocking check)
    if _sim_semaphore.locked():
        # All slots are in use - check if we should queue or reject
        active_count = sum(
            1 for s in _running_simulations.values()
            if s.get("status") == "running"
        )
        if active_count >= MAX_CONCURRENT_SIMULATIONS:
            raise HTTPException(
                status_code=429,
                detail=f"Too many concurrent simulations (max {MAX_CONCURRENT_SIMULATIONS})"
            )

    # Submit to executor
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(_executor, _run_simulation_sync, run_id, scenario_path)

    with _simulations_lock:
        _running_simulations[run_id] = {"future": future, "status": "running"}

    return StartSimulationResponse(
        run_id=run_id,
        status=SimulationStatus.RUNNING,
        message=f"Simulation started: {scenario_path.name}"
    )


@app.get("/api/events/stream")
async def event_stream(run_id: Optional[str] = None, history: bool = False):
    """SSE event stream for real-time updates."""
    event_bus = get_event_bus()

    async def generate():
        # Send connection event
        yield f"data: {{\"event_type\": \"connected\", \"message\": \"Connected to event stream\"}}\n\n"

        # Use the async iterator subscribe method
        async for event in event_bus.subscribe(run_id=run_id, include_history=history):
            yield event.to_sse()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/events/stream/{run_id}")
async def event_stream_for_run(run_id: str, history: bool = False):
    """SSE event stream for a specific run."""
    return await event_stream(run_id=run_id, history=history)


# ============================================================================
# Job Queue Endpoints (only active when APART_USE_JOB_QUEUE=1)
# ============================================================================


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a queued job.

    Only available when job queue mode is enabled.
    """
    if not USE_JOB_QUEUE:
        raise HTTPException(status_code=501, detail="Job queue not enabled")

    from server.job_queue import get_job_status

    try:
        return get_job_status(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.get("/api/queue/stats")
async def queue_stats():
    """Get job queue statistics.

    Only available when job queue mode is enabled.
    """
    if not USE_JOB_QUEUE:
        raise HTTPException(status_code=501, detail="Job queue not enabled")

    from server.job_queue import get_queue_stats

    return get_queue_stats()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job.

    Only works for jobs that haven't started yet.
    Only available when job queue mode is enabled.
    """
    if not USE_JOB_QUEUE:
        raise HTTPException(status_code=501, detail="Job queue not enabled")

    from server.job_queue import cancel_job as do_cancel

    cancelled = do_cancel(job_id)
    if cancelled:
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} cannot be cancelled (may be running or completed)"
        )
