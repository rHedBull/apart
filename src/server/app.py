"""
FastAPI application for the Apart Dashboard.

Provides:
- REST API for simulation management
- SSE event streaming for real-time updates
"""

import asyncio
import atexit
import signal
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
    import os
    if os.environ.get("APART_USE_DATABASE", "").lower() in ("1", "true", "yes"):
        from server.database import init_db
        from server.event_bus import EventBus
        EventBus._use_database = True
        init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    _initialize_database()
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

    Uses a bounded thread pool to prevent resource exhaustion.
    Limits concurrent simulations via semaphore.
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
