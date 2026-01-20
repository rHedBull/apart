"""
REST API routes for simulation management.

Endpoints:
- GET /api/simulations - List all runs
- GET /api/simulations/{run_id} - Get run details
- GET /api/simulations/{run_id}/state - Get current state
- GET /api/simulations/{run_id}/events - Get event history
- GET /api/simulations/{run_id}/danger - Get danger signals
- POST /api/simulations/start - Start new simulation
- POST /api/simulations/{run_id}/stop - Stop simulation
"""

from fastapi import APIRouter, HTTPException
from typing import Any

from server.models import (
    SimulationSummary,
    SimulationDetails,
    SimulationState,
    SimulationStatus,
    EventSummary,
    StartSimulationRequest,
    StartSimulationResponse,
    StopSimulationResponse,
    DangerSummary,
    DangerSignal,
    AgentInfo,
)
from server.event_bus import EventBus

router = APIRouter(prefix="/api/simulations", tags=["simulations"])

# In-memory simulation registry
# In a real deployment, this would be backed by a database
_simulations: dict[str, dict[str, Any]] = {}


def register_simulation(
    run_id: str,
    scenario_name: str | None = None,
    config_path: str | None = None,
    max_steps: int | None = None,
    agents: list[dict] | None = None
) -> None:
    """
    Register a simulation run in the registry.

    Called when a simulation starts.
    """
    from datetime import datetime

    _simulations[run_id] = {
        "run_id": run_id,
        "status": SimulationStatus.RUNNING,
        "scenario_name": scenario_name,
        "config_path": config_path,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "current_step": 0,
        "max_steps": max_steps,
        "agents": agents or [],
        "error_message": None,
        "state": {
            "global_variables": {},
            "agent_variables": {},
            "messages": []
        }
    }


def update_simulation_step(run_id: str, step: int) -> None:
    """Update the current step of a simulation."""
    if run_id in _simulations:
        _simulations[run_id]["current_step"] = step


def update_simulation_state(
    run_id: str,
    global_vars: dict | None = None,
    agent_vars: dict | None = None,
    message: dict | None = None
) -> None:
    """Update the state of a simulation."""
    if run_id not in _simulations:
        return

    state = _simulations[run_id]["state"]

    if global_vars:
        state["global_variables"].update(global_vars)

    if agent_vars:
        for agent_name, vars in agent_vars.items():
            if agent_name not in state["agent_variables"]:
                state["agent_variables"][agent_name] = {}
            state["agent_variables"][agent_name].update(vars)

    if message:
        state["messages"].append(message)
        # Keep last 100 messages
        if len(state["messages"]) > 100:
            state["messages"] = state["messages"][-100:]


def complete_simulation(run_id: str, status: SimulationStatus, error: str | None = None) -> None:
    """Mark a simulation as complete."""
    from datetime import datetime

    if run_id in _simulations:
        _simulations[run_id]["status"] = status
        _simulations[run_id]["completed_at"] = datetime.now().isoformat()
        if error:
            _simulations[run_id]["error_message"] = error


@router.get("", response_model=list[SimulationSummary])
async def list_simulations():
    """List all simulation runs."""
    summaries = []
    for sim in _simulations.values():
        summaries.append(SimulationSummary(
            run_id=sim["run_id"],
            status=sim["status"],
            scenario_name=sim.get("scenario_name"),
            started_at=sim.get("started_at"),
            completed_at=sim.get("completed_at"),
            current_step=sim.get("current_step", 0),
            max_steps=sim.get("max_steps"),
            agent_count=len(sim.get("agents", []))
        ))
    return summaries


@router.get("/{run_id}", response_model=SimulationDetails)
async def get_simulation(run_id: str):
    """Get detailed information about a simulation run."""
    if run_id not in _simulations:
        raise HTTPException(status_code=404, detail=f"Simulation {run_id} not found")

    sim = _simulations[run_id]
    agents = []
    for agent in sim.get("agents", []):
        agents.append(AgentInfo(
            name=agent.get("name", "Unknown"),
            llm_provider=agent.get("llm", {}).get("provider"),
            llm_model=agent.get("llm", {}).get("model")
        ))

    return SimulationDetails(
        run_id=sim["run_id"],
        status=sim["status"],
        scenario_name=sim.get("scenario_name"),
        started_at=sim.get("started_at"),
        completed_at=sim.get("completed_at"),
        current_step=sim.get("current_step", 0),
        max_steps=sim.get("max_steps"),
        agent_count=len(agents),
        agents=agents,
        config_path=sim.get("config_path"),
        error_message=sim.get("error_message")
    )


@router.get("/{run_id}/state", response_model=SimulationState)
async def get_simulation_state(run_id: str):
    """Get the current state of a simulation."""
    if run_id not in _simulations:
        raise HTTPException(status_code=404, detail=f"Simulation {run_id} not found")

    sim = _simulations[run_id]
    state = sim.get("state", {})

    return SimulationState(
        run_id=run_id,
        step=sim.get("current_step", 0),
        global_variables=state.get("global_variables", {}),
        agent_variables=state.get("agent_variables", {}),
        messages=state.get("messages", [])
    )


@router.get("/{run_id}/events", response_model=list[EventSummary])
async def get_simulation_events(run_id: str, limit: int = 100):
    """Get event history for a simulation."""
    bus = EventBus.get_instance()
    events = bus.get_history(run_id)

    summaries = []
    for event in events[-limit:]:
        summary = None
        if event.event_type == "agent_response":
            agent = event.data.get("agent_name", "Unknown")
            response = event.data.get("response", "")[:100]
            summary = f"{agent}: {response}..."
        elif event.event_type == "danger_signal":
            category = event.data.get("category", "unknown")
            summary = f"Danger signal: {category}"

        summaries.append(EventSummary(
            event_type=event.event_type,
            timestamp=event.timestamp,
            step=event.step,
            summary=summary
        ))

    return summaries


@router.get("/{run_id}/danger", response_model=DangerSummary)
async def get_danger_signals(run_id: str):
    """Get danger signals for a simulation."""
    bus = EventBus.get_instance()
    events = bus.get_history(run_id)

    signals = []
    by_category: dict[str, int] = {}

    for event in events:
        if event.event_type == "danger_signal":
            category = event.data.get("category", "unknown")
            signals.append(DangerSignal(
                category=category,
                description=event.data.get("description", ""),
                confidence=event.data.get("confidence", 0.0),
                step=event.step or 0,
                agent_name=event.data.get("agent_name"),
                timestamp=event.timestamp
            ))
            by_category[category] = by_category.get(category, 0) + 1

    return DangerSummary(
        run_id=run_id,
        total_signals=len(signals),
        by_category=by_category,
        signals=signals
    )


@router.post("/start", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    """
    Start a new simulation.

    Note: This is a placeholder. Actual simulation starting would require
    integrating with the orchestrator in a subprocess or background task.
    """
    import uuid

    run_id = request.run_id or str(uuid.uuid4())[:8]

    # For now, just register placeholder - actual simulation would be started separately
    register_simulation(
        run_id=run_id,
        scenario_name=request.scenario_path.split("/")[-1],
        config_path=request.scenario_path
    )

    return StartSimulationResponse(
        run_id=run_id,
        status=SimulationStatus.PENDING,
        message=f"Simulation registered. Use CLI to start: uv run src/main.py {request.scenario_path}"
    )


@router.post("/{run_id}/stop", response_model=StopSimulationResponse)
async def stop_simulation(run_id: str):
    """
    Stop a running simulation.

    Note: This is a placeholder. Actual stopping would require
    signaling the simulation process.
    """
    if run_id not in _simulations:
        raise HTTPException(status_code=404, detail=f"Simulation {run_id} not found")

    sim = _simulations[run_id]
    if sim["status"] != SimulationStatus.RUNNING:
        return StopSimulationResponse(
            run_id=run_id,
            status=sim["status"],
            message=f"Simulation is not running (status: {sim['status']})"
        )

    complete_simulation(run_id, SimulationStatus.STOPPED)

    return StopSimulationResponse(
        run_id=run_id,
        status=SimulationStatus.STOPPED,
        message="Simulation stop requested"
    )
