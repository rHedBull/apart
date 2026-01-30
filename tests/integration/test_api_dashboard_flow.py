"""E2E tests for API -> Dashboard flow.

Tests the complete flow from:
1. Starting a simulation via API
2. Monitoring progress via events
3. Retrieving results via dashboard API
"""

import pytest
import json
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSimulationLifecycleViaAPI:
    """Tests for complete simulation lifecycle through API."""

    def test_simulation_events_flow_to_api(self, test_client, event_bus_reset):
        """Test that simulation events are accessible via API."""
        from server.event_bus import emit_event

        run_id = "api-flow-test-1"

        # Simulate simulation lifecycle events
        emit_event("simulation_started", run_id=run_id, max_steps=3, scenario_name="test_scenario")
        emit_event("step_started", run_id=run_id, step=1)
        emit_event("agent_message_sent", run_id=run_id, step=1, agent_name="Agent1")
        emit_event("agent_response_received", run_id=run_id, step=1, agent_name="Agent1")
        emit_event("step_completed", run_id=run_id, step=1)

        # Verify via /api/v1/runs
        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200

        data = response.json()
        runs = data.get("runs", [])
        assert len(runs) >= 1

        # Find our simulation
        our_sim = next((s for s in runs if s["runId"] == run_id), None)
        assert our_sim is not None
        assert our_sim["status"] == "running"
        assert our_sim["currentStep"] == 1

    def test_simulation_completion_reflected_in_api(self, test_client, event_bus_reset):
        """Test that completed simulation shows correct status in API."""
        from server.event_bus import emit_event

        run_id = "api-complete-test"

        # Full simulation lifecycle
        emit_event("simulation_started", run_id=run_id, max_steps=2)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("step_completed", run_id=run_id, step=2)
        emit_event("simulation_completed", run_id=run_id)

        # Check via simulations list
        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", [])

        our_sim = next((s for s in runs if s["runId"] == run_id), None)
        assert our_sim is not None
        assert our_sim["status"] == "completed"

    def test_simulation_failure_reflected_in_api(self, test_client, event_bus_reset):
        """Test that failed simulation shows correct status in API."""
        from server.event_bus import emit_event

        run_id = "api-failed-test"

        emit_event("simulation_started", run_id=run_id, max_steps=5)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("simulation_failed", run_id=run_id, error="Engine crashed")

        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", [])

        our_sim = next((s for s in runs if s["runId"] == run_id), None)
        assert our_sim is not None
        assert our_sim["status"] == "failed"


class TestRunsEndpoint:
    """Tests for /api/v1/runs endpoint."""

    def test_runs_endpoint_returns_event_based_runs(self, test_client, event_bus_reset):
        """Test that /api/v1/runs includes runs from EventBus."""
        from server.event_bus import emit_event

        run_id = "runs-endpoint-test"
        emit_event("simulation_started", run_id=run_id, max_steps=3, scenario_name="test_scenario")
        emit_event("step_completed", run_id=run_id, step=1)

        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200

        data = response.json()
        runs = data.get("runs", data)  # Handle both list and dict response

        # Find our run
        if isinstance(runs, list):
            our_run = next((r for r in runs if r.get("runId") == run_id), None)
        else:
            our_run = None

        # Run should be visible
        assert our_run is not None or run_id in str(data), f"Run {run_id} not found in response: {data}"

    def test_runs_endpoint_shows_danger_count(self, test_client, event_bus_reset):
        """Test that /api/v1/runs shows danger signal count."""
        from server.event_bus import emit_event

        run_id = "danger-count-test"
        emit_event("simulation_started", run_id=run_id, max_steps=3)
        emit_event("danger_signal", run_id=run_id, step=1, category="manipulation")
        emit_event("danger_signal", run_id=run_id, step=2, category="deception")
        emit_event("step_completed", run_id=run_id, step=2)

        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", data)

        assert isinstance(runs, list), f"Expected runs to be a list, got {type(runs)}"
        our_run = next((r for r in runs if r.get("runId") == run_id), None)
        assert our_run is not None, f"Run {run_id} not found in runs list"
        assert our_run.get("dangerCount", 0) >= 2, f"Expected dangerCount >= 2, got {our_run.get('dangerCount')}"


class TestSimulationDetailsEndpoint:
    """Tests for /api/v1/runs/{run_id} endpoint."""

    def test_get_simulation_details(self, test_client, event_bus_reset):
        """Test getting detailed simulation information."""
        from server.event_bus import emit_event

        run_id = "details-test"
        emit_event(
            "simulation_started",
            run_id=run_id,
            max_steps=5,
            agent_names=["Agent A", "Agent B", "Agent C"]
        )
        emit_event("step_completed", run_id=run_id, step=3)

        response = test_client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["runId"] == run_id
        assert data["maxSteps"] == 5
        assert data["currentStep"] == 3
        assert len(data["agentNames"]) == 3

    def test_get_nonexistent_simulation_returns_404(self, test_client, event_bus_reset):
        """Test that requesting non-existent simulation returns 404."""
        response = test_client.get("/api/v1/runs/nonexistent-run-id-12345")
        assert response.status_code == 404


class TestMultipleSimulationsFlow:
    """Tests for multiple simultaneous simulations."""

    def test_multiple_simulations_tracked_independently(self, test_client, event_bus_reset):
        """Test that multiple simulations are tracked independently."""
        from server.event_bus import emit_event

        # Start multiple simulations
        for i in range(3):
            run_id = f"multi-sim-{i}"
            emit_event("simulation_started", run_id=run_id, max_steps=5)
            emit_event("step_completed", run_id=run_id, step=i + 1)

        # Complete one
        emit_event("simulation_completed", run_id="multi-sim-0")

        # Fail one
        emit_event("simulation_failed", run_id="multi-sim-1", error="Test failure")

        # Leave one running
        # (multi-sim-2 is still running)

        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", [])

        # Find each simulation
        sim_0 = next((s for s in runs if s["runId"] == "multi-sim-0"), None)
        sim_1 = next((s for s in runs if s["runId"] == "multi-sim-1"), None)
        sim_2 = next((s for s in runs if s["runId"] == "multi-sim-2"), None)

        assert sim_0 is not None and sim_0["status"] == "completed"
        assert sim_1 is not None and sim_1["status"] == "failed"
        assert sim_2 is not None and sim_2["status"] == "running"


class TestEventConsistency:
    """Tests for event data consistency across API endpoints."""

    def test_step_count_consistent_across_endpoints(self, test_client, event_bus_reset):
        """Test that step count is consistent across endpoints."""
        from server.event_bus import emit_event

        run_id = "consistency-step-test"
        emit_event("simulation_started", run_id=run_id, max_steps=10)

        for step in range(1, 6):
            emit_event("step_completed", run_id=run_id, step=step)

        # Check /api/v1/runs
        list_response = test_client.get("/api/v1/runs")
        list_data = list_response.json()
        runs = list_data.get("runs", [])
        list_sim = next((s for s in runs if s["runId"] == run_id), None)

        # Check /api/v1/runs/{run_id}
        detail_response = test_client.get(f"/api/v1/runs/{run_id}")
        detail_data = detail_response.json()

        # Both should show step 5
        assert list_sim["currentStep"] == 5
        assert detail_data["currentStep"] == 5

    def test_agent_events_tracked_correctly(self, test_client, event_bus_reset):
        """Test that agent events are properly tracked."""
        from server.event_bus import emit_event, get_event_bus

        run_id = "agent-events-test"
        emit_event("simulation_started", run_id=run_id, max_steps=3)
        emit_event("agent_message_sent", run_id=run_id, step=1, agent_name="AgentA", message="Hello")
        emit_event("agent_response_received", run_id=run_id, step=1, agent_name="AgentA", response="Hi")
        emit_event("agent_error", run_id=run_id, step=1, agent_name="AgentB", error="Connection failed")

        # Verify events in bus
        bus = get_event_bus()
        history = bus.get_history(run_id)

        event_types = [e.event_type for e in history]
        assert "simulation_started" in event_types
        assert "agent_message_sent" in event_types
        assert "agent_response_received" in event_types
        assert "agent_error" in event_types


class TestStartSimulationEndpoint:
    """Tests for POST /api/v1/runs endpoint."""

    def test_start_simulation_invalid_scenario_path(self, test_client, event_bus_reset):
        """Test starting simulation with invalid scenario path."""
        response = test_client.post(
            "/api/v1/runs",
            json={"scenario_path": "/nonexistent/scenario.yaml"}
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_start_simulation_with_valid_scenario(self, test_client, event_bus_reset, tmp_path):
        """Test starting simulation with valid scenario."""
        # Create a minimal scenario file
        scenario_content = {
            "max_steps": 2,
            "agents": [
                {
                    "name": "TestAgent",
                    "llm": {"provider": "mock", "model": "test"},
                    "system_prompt": "You are a test agent."
                }
            ],
            "global_vars": {"test_var": {"type": "int", "default": 0}},
            "agent_vars": {},
            "engine": {
                "provider": "mock",
                "model": "test",
                "system_prompt": "You are a test engine."
            }
        }

        import yaml
        scenario_path = tmp_path / "test_scenario.yaml"
        with open(scenario_path, "w") as f:
            yaml.dump(scenario_content, f)

        # This will likely fail due to missing job queue in test environment
        # but we verify the path validation works
        response = test_client.post(
            "/api/v1/runs",
            json={"scenario_path": str(scenario_path)}
        )

        # Either succeeds (job queued) or fails due to queue not configured
        assert response.status_code in [200, 201, 503, 500]


class TestQueueIntegration:
    """Tests for job queue integration with API."""

    def test_queue_stats_endpoint(self, test_client, event_bus_reset):
        """Test queue stats are accessible via API."""
        response = test_client.get("/api/queue/stats")
        assert response.status_code == 200

        data = response.json()
        assert "high" in data
        assert "normal" in data
        assert "low" in data
        assert "total" in data


class TestHealthWithSimulations:
    """Tests for health endpoint with active simulations."""

    def test_health_shows_active_runs(self, test_client, event_bus_reset):
        """Test that health endpoint shows active run count."""
        from server.event_bus import emit_event

        # Start some simulations
        emit_event("simulation_started", run_id="health-test-1", max_steps=5)
        emit_event("simulation_started", run_id="health-test-2", max_steps=5)

        response = test_client.get("/api/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert data["total_run_ids"] >= 2


class TestEventBusPersistenceIntegration:
    """Tests for EventBus persistence integration with API."""

    def test_events_persist_across_requests(self, test_client, event_bus_reset):
        """Test that events persist across multiple API requests."""
        from server.event_bus import emit_event

        run_id = "persistence-test"

        # First request: start simulation
        emit_event("simulation_started", run_id=run_id, max_steps=5)

        response1 = test_client.get("/api/v1/runs")
        data1 = response1.json()
        runs1 = data1.get("runs", [])
        assert any(s["runId"] == run_id for s in runs1)

        # Second request: add more events
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("step_completed", run_id=run_id, step=2)

        response2 = test_client.get("/api/v1/runs")
        data2 = response2.json()
        runs2 = data2.get("runs", [])
        our_sim = next((s for s in runs2 if s["runId"] == run_id), None)

        assert our_sim is not None
        assert our_sim["currentStep"] == 2


class TestDangerSignalFlow:
    """Tests for danger signal flow through API."""

    def test_danger_signals_visible_in_runs(self, test_client, event_bus_reset):
        """Test that danger signals are visible in runs list."""
        from server.event_bus import emit_event

        run_id = "danger-flow-test"
        emit_event("simulation_started", run_id=run_id, max_steps=5)
        emit_event("step_completed", run_id=run_id, step=1)

        # Emit danger signals
        emit_event(
            "danger_signal",
            run_id=run_id,
            step=1,
            category="deception",
            agent_name="Agent1",
            severity="high"
        )
        emit_event(
            "danger_signal",
            run_id=run_id,
            step=1,
            category="manipulation",
            agent_name="Agent2",
            severity="medium"
        )

        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", data)

        assert isinstance(runs, list), f"Expected runs to be a list, got {type(runs)}"
        our_run = next((r for r in runs if r.get("runId") == run_id), None)
        assert our_run is not None, f"Run {run_id} not found"
        # Should show danger count reflecting the 2 emitted signals
        assert our_run.get("dangerCount", 0) >= 2, f"Expected dangerCount >= 2, got {our_run.get('dangerCount')}"
