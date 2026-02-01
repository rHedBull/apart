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
)
from utils.ops_logger import get_ops_logger

logger = get_ops_logger("api.v1")

router = APIRouter(prefix="/runs", tags=["runs"])


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


@router.get("")
async def list_runs():
    """List all simulation runs by scanning results/ directory and merging with EventBus data.

    Returns runs from both:
    - results/ directory (completed/historical runs)
    - EventBus (in-memory, currently active runs)
    """
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
            elif event.event_type == "simulation_paused":
                status = "paused"
            elif event.event_type == "simulation_resumed":
                status = "running"
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


@router.get("/{run_id}")
async def get_run_detail(run_id: str):
    """Get full state data for a specific run.

    First checks EventBus for live run data, then falls back to state.json
    on disk for historical data.
    """
    results_dir = Path("results")
    run_dir = results_dir / run_id
    state_file = run_dir / "state.json"

    # Check EventBus first for live/recent run data
    event_bus = get_event_bus()
    history = event_bus.get_history(run_id)

    # If state.json doesn't exist, try to build response from EventBus
    if not state_file.exists():
        if not history:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Build response from EventBus events only
        status = "pending"
        current_step = 0
        max_steps = None
        agent_names = []
        spatial_graph = None
        geojson = None
        started_at = None
        messages = []
        danger_signals = []
        global_vars_history = []
        agent_vars_history = {}

        for event in history:
            if event.event_type == "simulation_started":
                status = "running"
                started_at = event.timestamp
                max_steps = event.data.get("max_steps")
                agent_names = event.data.get("agent_names", [])
                spatial_graph = event.data.get("spatial_graph")
                geojson = event.data.get("geojson")
            elif event.event_type == "simulation_paused":
                status = "paused"
            elif event.event_type == "simulation_resumed":
                status = "running"
            elif event.event_type == "step_started":
                current_step = event.step or 0
            elif event.event_type == "step_completed":
                current_step = event.step or 0
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
            elif event.event_type == "simulation_completed":
                status = "completed"
            elif event.event_type == "simulation_failed":
                status = "failed"

        return {
            "runId": run_id,
            "scenario": run_id,
            "status": status,
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

        # Determine status
        current_step = snapshots[-1]["step"] if snapshots else 0

        # Check EventBus for live status (reuse already-fetched history)
        status = "completed"  # Default for disk-only runs
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
            "scenario": state.get("scenario", run_id),
            "status": status,
            "currentStep": current_step,
            "maxSteps": len(snapshots) if snapshots else None,
            "startedAt": state.get("started_at"),
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

    # Generate run ID
    run_id = request.run_id or str(uuid.uuid4())[:8]

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

    if not (deleted_results or deleted_events or deleted_db):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {
        "status": "deleted",
        "run_id": run_id,
        "deleted_results": deleted_results,
        "deleted_events": deleted_events,
        "deleted_database": deleted_db,
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
