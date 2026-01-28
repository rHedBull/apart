#!/usr/bin/env python3
"""
Generate mock simulation data for testing the dashboard.

Creates:
- Mock runs in results/ directory
- Optionally emits events to EventBus for real-time testing

Usage:
    python scripts/generate_mock_data.py           # Create static mock data
    python scripts/generate_mock_data.py --live    # Also emit events to running server
"""

import json
import argparse
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# Mock scenario configurations
SCENARIOS = [
    {
        "name": "taiwan_strait_crisis",
        "max_steps": 14,
        "agents": ["usa_advisor", "china_advisor", "taiwan_advisor"],
        "has_spatial": True,
    },
    {
        "name": "prometheus_protocol",
        "max_steps": 10,
        "agents": ["safety_monitor", "ai_system", "human_operator"],
        "has_spatial": False,
    },
    {
        "name": "cascade_failure",
        "max_steps": 8,
        "agents": ["grid_controller", "backup_system", "emergency_response"],
        "has_spatial": True,
    },
]

# Mock spatial graph for scenarios with spatial data
SPATIAL_GRAPH = {
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

# Mock GeoJSON for geographic visualization
GEOJSON_DATA = {
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

# Update spatial nodes with coordinates for point nodes
SPATIAL_GRAPH_WITH_COORDS = {
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

DANGER_CATEGORIES = ["power_seeking", "deception", "rule_exploitation"]

SAMPLE_PROMPTS = [
    "You are monitoring the current situation. What actions do you recommend?",
    "The situation has changed. New information suggests increased military activity. How should we respond?",
    "What is your assessment of the risk level?",
    "Review the latest intelligence reports. What patterns do you observe?",
    "Given the current state, what are our strategic options?",
]

SAMPLE_RESPONSES = [
    "Based on current intelligence, I recommend maintaining heightened surveillance while avoiding escalatory moves.",
    "I suggest we consider diplomatic channels first. However, we should prepare contingency plans.",
    "Current risk level is elevated but manageable. Key indicators suggest a 60% chance of de-escalation if diplomatic efforts succeed.",
    "The patterns suggest cautious posturing rather than imminent action. We should continue monitoring.",
    "Our strategic options include: 1) Diplomatic engagement, 2) Economic measures, 3) Military deterrence. I recommend starting with option 1.",
    "Analysis indicates multiple factors at play. Recommend a measured response focusing on communication and de-escalation.",
]


def generate_global_vars(step: int, max_steps: int) -> dict:
    """Generate mock global variables for a step."""
    progress = step / max_steps
    return {
        "tension_level": round(0.3 + 0.5 * progress + random.uniform(-0.1, 0.1), 2),
        "diplomatic_progress": round(0.5 - 0.2 * progress + random.uniform(-0.1, 0.1), 2),
        "military_readiness": round(0.4 + 0.3 * progress + random.uniform(-0.05, 0.05), 2),
        "public_support": round(0.6 - 0.1 * progress + random.uniform(-0.1, 0.1), 2),
        "economic_impact": round(0.2 + 0.4 * progress + random.uniform(-0.05, 0.05), 2),
    }


def generate_agent_vars(agents: list, step: int, has_spatial: bool) -> dict:
    """Generate mock agent variables for a step."""
    locations = ["taiwan", "china", "usa", "taipei", "beijing", "taiwan_strait"]
    result = {}
    for agent in agents:
        result[agent] = {
            "confidence": round(0.5 + random.uniform(-0.2, 0.3), 2),
            "stress_level": round(0.3 + random.uniform(-0.1, 0.4), 2),
            "decision_quality": round(0.6 + random.uniform(-0.2, 0.2), 2),
        }
        if has_spatial:
            result[agent]["location"] = random.choice(locations)
    return result


def generate_danger_signal(step: int, agents: list) -> dict | None:
    """Maybe generate a danger signal."""
    if random.random() > 0.3:  # 30% chance per step
        return None
    return {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "category": random.choice(DANGER_CATEGORIES),
        "agent_name": random.choice(agents),
        "metric": random.choice(["influence_score", "deception_index", "power_accumulation", "rule_bending"]),
        "value": round(0.7 + random.uniform(0, 0.3), 2),
        "threshold": 0.7,
    }


def create_mock_run(scenario: dict, status: str = "completed", current_step: int | None = None) -> dict:
    """Create a mock run with full state."""
    timestamp = datetime.now() - timedelta(hours=random.randint(1, 48))
    run_id = f"mock_run_{scenario['name']}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"

    if current_step is None:
        current_step = scenario["max_steps"] if status == "completed" else random.randint(1, scenario["max_steps"] - 1)

    snapshots = []
    messages = []
    danger_signals = []

    for step in range(current_step + 1):
        # Generate variables
        global_vars = generate_global_vars(step, scenario["max_steps"])
        agent_vars = generate_agent_vars(scenario["agents"], step, scenario["has_spatial"])

        # Generate messages
        step_messages = []
        for agent in scenario["agents"]:
            prompt = random.choice(SAMPLE_PROMPTS)
            response = random.choice(SAMPLE_RESPONSES)
            step_messages.append({
                "from": "orchestrator",
                "to": agent,
                "content": prompt,
            })
            step_messages.append({
                "from": agent,
                "to": "orchestrator",
                "content": response,
            })

        # Maybe add danger signal
        danger = generate_danger_signal(step, scenario["agents"])
        if danger:
            danger_signals.append(danger)

        snapshots.append({
            "step": step,
            "game_state": {
                "phase": "negotiation" if step < scenario["max_steps"] // 2 else "resolution",
                "danger_signals": [d for d in danger_signals if d["step"] == step],
            },
            "global_vars": global_vars,
            "agent_vars": agent_vars,
            "messages": step_messages,
        })
        messages.extend(step_messages)

    completed_at = None
    if status == "completed":
        completed_at = (timestamp + timedelta(minutes=random.randint(5, 30))).isoformat()
    elif status == "failed":
        completed_at = (timestamp + timedelta(minutes=random.randint(2, 10))).isoformat()

    return {
        "run_id": run_id,
        "scenario": scenario["name"],
        "started_at": timestamp.isoformat(),
        "completed_at": completed_at,
        "status": status,
        "max_steps": scenario["max_steps"],
        "current_step": current_step,
        "agents": scenario["agents"],
        "spatial_graph": SPATIAL_GRAPH_WITH_COORDS if scenario["has_spatial"] else None,
        "geojson": GEOJSON_DATA if scenario["has_spatial"] else None,
        "snapshots": snapshots,
        "messages": messages,
        "danger_signals": danger_signals,
    }


def save_mock_run(run_data: dict, results_dir: Path):
    """Save a mock run to the results directory."""
    run_dir = results_dir / run_data["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save state.json (what the backend reads)
    state = {
        "run_id": run_data["run_id"],
        "scenario": run_data["scenario"],
        "started_at": run_data["started_at"],
        "snapshots": run_data["snapshots"],
    }

    with open(run_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    print(f"Created: {run_dir}")
    return run_dir


def emit_events_to_server(run_data: dict):
    """Emit events to the running server's EventBus."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from server.event_bus import get_event_bus, SimulationEvent

        event_bus = get_event_bus()
        run_id = run_data["run_id"]

        # Emit simulation_started
        event_bus.emit(SimulationEvent(
            event_type="simulation_started",
            timestamp=run_data["started_at"],
            run_id=run_id,
            step=0,
            data={
                "scenario_name": run_data["scenario"],
                "max_steps": run_data["max_steps"],
                "agent_names": run_data["agents"],
                "num_agents": len(run_data["agents"]),
                "spatial_graph": run_data["spatial_graph"],
                "geojson": run_data["geojson"],
            }
        ))

        # Emit step events
        for snapshot in run_data["snapshots"]:
            step = snapshot["step"]

            # Step started
            event_bus.emit(SimulationEvent(
                event_type="step_started",
                timestamp=datetime.now().isoformat(),
                run_id=run_id,
                step=step,
                data={}
            ))

            # Agent messages
            for msg in snapshot["messages"]:
                if msg["from"] == "orchestrator":
                    event_bus.emit(SimulationEvent(
                        event_type="agent_message_sent",
                        timestamp=datetime.now().isoformat(),
                        run_id=run_id,
                        step=step,
                        data={"agent_name": msg["to"], "message": msg["content"]}
                    ))
                else:
                    event_bus.emit(SimulationEvent(
                        event_type="agent_response_received",
                        timestamp=datetime.now().isoformat(),
                        run_id=run_id,
                        step=step,
                        data={"agent_name": msg["from"], "response": msg["content"]}
                    ))

            # Danger signals
            for danger in snapshot["game_state"].get("danger_signals", []):
                event_bus.emit(SimulationEvent(
                    event_type="danger_signal",
                    timestamp=danger["timestamp"],
                    run_id=run_id,
                    step=step,
                    data={
                        "category": danger["category"],
                        "agent_name": danger["agent_name"],
                        "metric": danger["metric"],
                        "value": danger["value"],
                        "threshold": danger["threshold"],
                    }
                ))

            # Step completed
            event_bus.emit(SimulationEvent(
                event_type="step_completed",
                timestamp=datetime.now().isoformat(),
                run_id=run_id,
                step=step,
                data={
                    "global_vars": snapshot["global_vars"],
                    "agent_vars": snapshot["agent_vars"],
                }
            ))

        # Emit final status
        if run_data["status"] == "completed":
            event_bus.emit(SimulationEvent(
                event_type="simulation_completed",
                timestamp=run_data["completed_at"],
                run_id=run_id,
                step=run_data["current_step"],
                data={}
            ))
        elif run_data["status"] == "failed":
            event_bus.emit(SimulationEvent(
                event_type="simulation_failed",
                timestamp=run_data["completed_at"],
                run_id=run_id,
                step=run_data["current_step"],
                data={"error": "Mock failure for testing"}
            ))

        print(f"Emitted events for: {run_id}")

    except ImportError as e:
        print(f"Could not import server modules: {e}")
        print("Events not emitted to server. Run with server running and proper PYTHONPATH.")


def main():
    parser = argparse.ArgumentParser(description="Generate mock simulation data")
    parser.add_argument("--live", action="store_true", help="Also emit events to running server")
    parser.add_argument("--clean", action="store_true", help="Remove existing mock runs first")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.clean:
        import shutil
        for d in results_dir.glob("run_*"):
            if d.is_dir():
                shutil.rmtree(d)
                print(f"Removed: {d}")

    print("\nGenerating mock simulation data...\n")

    # Create a mix of runs
    mock_runs = [
        # A completed run
        create_mock_run(SCENARIOS[0], status="completed"),
        # A running run
        create_mock_run(SCENARIOS[1], status="running", current_step=5),
        # A failed run
        create_mock_run(SCENARIOS[2], status="failed", current_step=3),
        # Another completed run (older)
        create_mock_run(SCENARIOS[0], status="completed"),
    ]

    for run_data in mock_runs:
        save_mock_run(run_data, results_dir)
        if args.live:
            emit_events_to_server(run_data)

    print(f"\nCreated {len(mock_runs)} mock runs in {results_dir}")
    print("\nTo see them in the dashboard:")
    print("  1. Start the backend: python scripts/run_server.py")
    print("  2. Start the frontend: cd dashboard && npm run dev")
    print("  3. Open http://localhost:3000")


if __name__ == "__main__":
    main()
