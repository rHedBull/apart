"""
FastAPI application for the Apart Dashboard.

Provides:
- REST API for simulation management
- SSE event streaming for real-time updates
"""

import asyncio
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.event_bus import get_event_bus
from server.models import (
    SimulationSummary,
    SimulationDetails,
    SimulationStatus,
    StartSimulationRequest,
    StartSimulationResponse,
)


# Track running simulations
_running_simulations: dict[str, dict] = {}
_simulations_lock = threading.Lock()


app = FastAPI(
    title="Apart Dashboard API",
    description="API for monitoring AI safety simulations",
    version="0.1.0",
    docs_url="/api/docs",
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


@app.post("/api/simulations", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    """Start a new simulation."""
    from core.orchestrator import Orchestrator
    from core.event_emitter import enable_event_emitter
    import uuid

    # Validate scenario path
    scenario_path = Path(request.scenario_path)
    if not scenario_path.exists():
        raise HTTPException(status_code=400, detail=f"Scenario not found: {request.scenario_path}")

    # Generate run ID
    run_id = request.run_id or str(uuid.uuid4())[:8]

    def run_simulation():
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
            from server.event_bus import emit_event
            emit_event("simulation_failed", run_id, error=str(e))

    # Start simulation in background thread
    thread = threading.Thread(target=run_simulation, daemon=True)
    thread.start()

    with _simulations_lock:
        _running_simulations[run_id] = {"thread": thread, "status": "running"}

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
