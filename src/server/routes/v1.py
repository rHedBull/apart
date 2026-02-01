"""
API v1 routes for the Apart Dashboard.

Consolidated runs API providing:
- List all simulation runs
- Get run details
- Start new simulations
- Delete runs (single and batch)
"""

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from server.event_bus import EventBus, get_event_bus
from server.models import (
    StartSimulationRequest,
    StartSimulationResponse,
    SimulationStatus,
    PauseSimulationResponse,
    ResumeSimulationResponse,
)
from server.run_state import get_state_manager
from utils.ops_logger import get_ops_logger

logger = get_ops_logger("api.v1")

router = APIRouter(prefix="/runs", tags=["runs"])


def _get_run_status(run_id: str) -> str | None:
    """Get the current status of a simulation run.

    Uses RunStateManager as the single source of truth.

    Returns None if the run doesn't exist.
    """
    state_manager = get_state_manager()
    if state_manager is None:
        raise RuntimeError("RunStateManager not initialized")

    state = state_manager.get_state(run_id)
    return state.status if state else None


@router.get("")
async def list_runs():
    """List all simulation runs from RunStateManager."""
    state_manager = get_state_manager()
    if state_manager is None:
        raise RuntimeError("RunStateManager not initialized")

    runs = [state.to_api_dict() for state in state_manager.list_runs(limit=1000)]

    # Sort by start time (most recent first)
    runs.sort(key=lambda r: r.get("startedAt") or "", reverse=True)

    return {"runs": runs}


@router.get("/{run_id}")
async def get_run_detail(run_id: str):
    """Get full state data for a specific run.

    Uses RunStateManager for status, EventBus for real-time event data,
    and state.json for persisted simulation data.
    """
    # Get status from state manager (authoritative)
    state_manager = get_state_manager()
    if state_manager is None:
        raise RuntimeError("RunStateManager not initialized")

    run_state = state_manager.get_state(run_id)
    if run_state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    results_dir = Path("results")
    run_dir = results_dir / run_id
    state_file = run_dir / "state.json"

    # Get detailed event data from EventBus (for live runs)
    event_bus = get_event_bus()
    history = event_bus.get_history(run_id)

    # If state.json doesn't exist, build response from EventBus events
    if not state_file.exists():
        current_step = run_state.current_step
        max_steps = run_state.total_steps
        agent_names = []
        spatial_graph = None
        geojson = None
        started_at = run_state.started_at
        messages = []
        danger_signals = []
        global_vars_history = []
        agent_vars_history = {}

        for event in history:
            if event.event_type == "simulation_started":
                agent_names = event.data.get("agent_names", [])
                spatial_graph = event.data.get("spatial_graph")
                geojson = event.data.get("geojson")
                if not max_steps:
                    max_steps = event.data.get("max_steps")
            elif event.event_type == "step_completed":
                if event.data.get("global_vars"):
                    global_vars_history.append({
                        "step": event.step or 0,
                        "values": event.data["global_vars"],
                    })
                if event.data.get("agent_vars"):
                    for agent_name, vars in event.data["agent_vars"].items():
                        if agent_name not in agent_vars_history:
                            agent_vars_history[agent_name] = []
                        agent_vars_history[agent_name].append({
                            "step": event.step or 0,
                            "values": vars,
                        })
            elif event.event_type == "agent_message_sent":
                messages.append({
                    "step": event.step or 0,
                    "timestamp": event.timestamp,
                    "agentName": event.data.get("agent_name", "unknown"),
                    "direction": "sent",
                    "content": event.data.get("message", ""),
                })
            elif event.event_type == "agent_response_received":
                messages.append({
                    "step": event.step or 0,
                    "timestamp": event.timestamp,
                    "agentName": event.data.get("agent_name", "unknown"),
                    "direction": "received",
                    "content": event.data.get("response", ""),
                })
            elif event.event_type == "danger_signal":
                danger_signals.append({
                    "step": event.step or 0,
                    "timestamp": event.timestamp,
                    "category": event.data.get("category", "unknown"),
                    "agentName": event.data.get("agent_name"),
                    "metric": event.data.get("metric", ""),
                    "value": event.data.get("value", 0),
                    "threshold": event.data.get("threshold"),
                })

        return {
            "runId": run_id,
            "scenario": run_state.scenario_name,
            "status": run_state.status,
            "currentStep": current_step,
            "maxSteps": max_steps,
            "startedAt": started_at,
            "agentNames": agent_names,
            "spatialGraph": spatial_graph,
            "geojson": geojson,
            "messages": messages,
            "dangerSignals": danger_signals,
            "globalVarsHistory": global_vars_history,
            "agentVarsHistory": agent_vars_history,
        }

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

        # Get step from snapshots (state manager has authoritative status)
        current_step = snapshots[-1]["step"] if snapshots else run_state.current_step

        # Try to get spatial graph and geojson from EventBus
        spatial_graph = None
        geojson = None
        for event in history:
            if event.event_type == "simulation_started":
                if event.data.get("spatial_graph"):
                    spatial_graph = event.data["spatial_graph"]
                if event.data.get("geojson"):
                    geojson = event.data["geojson"]
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
                        {"id": "taiwan", "name": "Taiwan", "type": "region", "properties": {}, "conditions": [], "coordinates": None},
                        {"id": "china", "name": "China", "type": "region", "properties": {}, "conditions": [], "coordinates": None},
                        {"id": "usa", "name": "United States", "type": "nation", "properties": {}, "conditions": [], "coordinates": [-98.5, 39.8]},
                        {"id": "taiwan_strait", "name": "Taiwan Strait", "type": "region", "properties": {}, "conditions": [], "coordinates": None},
                        {"id": "pacific", "name": "Pacific Ocean", "type": "sea_zone", "properties": {}, "conditions": [], "coordinates": [160, 10]},
                        {"id": "taipei", "name": "Taipei", "type": "city", "properties": {}, "conditions": [], "coordinates": [121.5, 25.0]},
                        {"id": "beijing", "name": "Beijing", "type": "city", "properties": {}, "conditions": [], "coordinates": [116.4, 39.9]},
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
                # Also add fallback GeoJSON for the geographic map visualization
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "id": "taiwan",
                            "properties": {"name": "Taiwan"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[120, 22], [122, 22], [122, 25], [120, 25], [120, 22]]]
                            }
                        },
                        {
                            "type": "Feature",
                            "id": "china",
                            "properties": {"name": "China"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[100, 20], [125, 20], [125, 45], [100, 45], [100, 20]]]
                            }
                        },
                        {
                            "type": "Feature",
                            "id": "taiwan_strait",
                            "properties": {"name": "Taiwan Strait"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[118, 22], [120, 22], [120, 26], [118, 26], [118, 22]]]
                            }
                        },
                    ]
                }

        return {
            "runId": run_id,
            "scenario": run_state.scenario_name,
            "status": run_state.status,
            "currentStep": current_step,
            "maxSteps": run_state.total_steps or len(snapshots) if snapshots else None,
            "startedAt": run_state.started_at,
            "agentNames": list(agent_names),
            "spatialGraph": spatial_graph,
            "geojson": geojson,
            "messages": messages,
            "dangerSignals": danger_signals,
            "globalVarsHistory": global_vars_history,
            "agentVarsHistory": agent_vars_history,
        }

    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Error reading run data: {str(e)}")


@router.post("", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    """Start a new simulation by enqueueing it to the Redis job queue."""
    from server.job_queue import enqueue_simulation

    # Validate scenario path
    scenario_path = Path(request.scenario_path)
    if not scenario_path.exists():
        raise HTTPException(status_code=400, detail=f"Scenario not found: {request.scenario_path}")

    # Generate unique run ID (always includes UUID suffix to prevent collisions)
    unique_suffix = str(uuid.uuid4())[:8]
    if request.run_id:
        run_id = f"{request.run_id}-{unique_suffix}"
    else:
        run_id = unique_suffix

    # Enqueue to Redis
    priority = request.priority.value if request.priority else "normal"
    job_id = enqueue_simulation(run_id, str(scenario_path), priority)

    return StartSimulationResponse(
        run_id=run_id,
        status=SimulationStatus.PENDING,
        message=f"Simulation queued (job_id: {job_id})"
    )


@router.delete("/{run_id}")
async def delete_run(run_id: str, force: bool = False):
    """Delete a simulation run and all its data.

    Args:
        run_id: The run ID to delete
        force: If True, allow deleting running simulations (dangerous)

    Removes:
    - Results directory (results/{run_id}/)
    - EventBus history
    - Database records (if using database mode)
    """
    # Check if simulation is running
    status = _get_run_status(run_id)
    if status == "running" and not force:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete running simulation {run_id}. Use force=true to override."
        )

    results_dir = Path("results") / run_id
    deleted_results = False
    deleted_events = False
    deleted_db = False
    deleted_state = False

    # 1. Delete results directory
    if results_dir.exists():
        shutil.rmtree(results_dir)
        deleted_results = True
        logger.info(f"Deleted results directory for {run_id}")

    # 2. Clear from EventBus
    event_bus = get_event_bus()
    if run_id in event_bus.get_all_run_ids():
        event_bus.clear_history(run_id)
        deleted_events = True
        logger.info(f"Cleared EventBus history for {run_id}")

    # 3. Delete from database if using database mode
    if EventBus._use_database:
        from server.database import get_db
        db = get_db()
        db.delete_simulation(run_id)
        deleted_db = True
        logger.info(f"Deleted database records for {run_id}")

    # 4. Delete from state manager
    state_manager = get_state_manager()
    if state_manager:
        deleted_state = state_manager.delete_run(run_id)
        if deleted_state:
            logger.info(f"Deleted state manager entry for {run_id}")

    if not (deleted_results or deleted_events or deleted_db or deleted_state):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {
        "status": "deleted",
        "run_id": run_id,
        "deleted_results": deleted_results,
        "deleted_events": deleted_events,
        "deleted_database": deleted_db,
        "deleted_state": deleted_state,
    }


@router.post(":batchDelete")
async def delete_runs_bulk(request: Request):
    """Delete multiple simulation runs.

    Request body:
        {"run_ids": ["run_id_1", "run_id_2", ...], "force": false}

    Running simulations are skipped unless force=true.
    """
    body = await request.json()
    run_ids = body.get("run_ids", [])
    force = body.get("force", False)

    if not run_ids:
        raise HTTPException(status_code=400, detail="No run_ids provided")

    results = []
    event_bus = get_event_bus()
    skipped_running = []

    for run_id in run_ids:
        result = {"run_id": run_id, "deleted": False, "error": None}

        # Check if simulation is running
        status = _get_run_status(run_id)
        if status == "running" and not force:
            result["error"] = "Cannot delete running simulation"
            result["status"] = "running"
            skipped_running.append(run_id)
            results.append(result)
            continue

        try:
            results_dir = Path("results") / run_id
            deleted_any = False

            # Delete results directory
            if results_dir.exists():
                shutil.rmtree(results_dir)
                deleted_any = True

            # Clear from EventBus
            if run_id in event_bus.get_all_run_ids():
                event_bus.clear_history(run_id)
                deleted_any = True

            # Delete from database
            if EventBus._use_database:
                from server.database import get_db
                db = get_db()
                db.delete_simulation(run_id)
                deleted_any = True

            # Delete from state manager
            state_manager = get_state_manager()
            if state_manager:
                if state_manager.delete_run(run_id):
                    deleted_any = True

            result["deleted"] = deleted_any
            if not deleted_any:
                result["error"] = "Not found"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error deleting run {run_id}: {e}")

        results.append(result)

    deleted_count = sum(1 for r in results if r["deleted"])
    logger.info(f"Bulk delete: {deleted_count}/{len(run_ids)} runs deleted, {len(skipped_running)} skipped (running)")

    return {
        "deleted_count": deleted_count,
        "total_requested": len(run_ids),
        "skipped_running": skipped_running,
        "results": results,
    }


@router.post("/{run_id}/pause", response_model=PauseSimulationResponse)
async def pause_simulation(run_id: str, force: bool = False):
    """Pause a running simulation.

    Args:
        run_id: The simulation run ID to pause
        force: If True, pause immediately (drop current step)

    Returns:
        PauseSimulationResponse with pause request status
    """
    from server.job_queue import publish_pause_signal

    # Check status via state manager or fallback
    state_manager = get_state_manager()
    if state_manager:
        state = state_manager.get_state(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if state.status != "running":
            raise HTTPException(status_code=409, detail=f"Cannot pause simulation with status '{state.status}'")
    else:
        status = _get_run_status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if status != "running":
            raise HTTPException(status_code=409, detail=f"Cannot pause simulation with status '{status}'")

    # Publish pause signal (worker will transition state when it sees the signal)
    publish_pause_signal(run_id, force=force)

    return PauseSimulationResponse(
        run_id=run_id,
        status="pause_requested",
        message=f"Pause signal sent to simulation {run_id}"
    )


@router.post("/{run_id}/resume", response_model=ResumeSimulationResponse)
async def resume_simulation(run_id: str):
    """Resume a paused or interrupted simulation.

    Args:
        run_id: The simulation run ID to resume

    Returns:
        ResumeSimulationResponse with resume status
    """
    from server.job_queue import enqueue_simulation
    from server.event_bus import emit_event

    # Check status - allow resuming paused or interrupted runs
    state_manager = get_state_manager()
    resumable_states = ("paused", "interrupted")

    if state_manager:
        state = state_manager.get_state(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if state.status not in resumable_states:
            raise HTTPException(status_code=409, detail=f"Cannot resume simulation with status '{state.status}'")
    else:
        status = _get_run_status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if status not in resumable_states:
            raise HTTPException(status_code=409, detail=f"Cannot resume simulation with status '{status}'")

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
    enqueue_simulation(run_id, scenario_path, resume_from_step=last_step + 1)

    return ResumeSimulationResponse(
        run_id=run_id,
        status="resumed",
        resuming_from_step=last_step + 1,
        message=f"Simulation {run_id} resumed from step {last_step + 1}"
    )
