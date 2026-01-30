"""Integration tests for the server API endpoints."""

import pytest
import time
from pathlib import Path


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, test_client):
        """Basic health check should return healthy."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_detailed_health(self, test_client, event_bus_reset):
        """Detailed health check should return system metrics."""
        response = test_client.get("/api/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "queue_stats" in data
        assert "event_bus_subscribers" in data
        assert "total_run_ids" in data


class TestSimulationEndpoints:
    """Tests for simulation management endpoints."""

    def test_list_simulations_empty(self, test_client, event_bus_reset):
        """List simulations should return empty list initially."""
        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert data["runs"] == []

    def test_list_simulations_with_events(self, test_client, event_bus_reset):
        """List simulations should return runs that have events."""
        from server.event_bus import emit_event

        # Emit some events for a run
        emit_event("simulation_started", run_id="test-run-1", max_steps=5, num_agents=2)
        emit_event("step_completed", run_id="test-run-1", step=1)

        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200

        data = response.json()
        runs = data.get("runs", [])
        assert len(runs) >= 1
        our_run = next((r for r in runs if r["runId"] == "test-run-1"), None)
        assert our_run is not None
        assert our_run["status"] == "running"

    def test_get_simulation_not_found(self, test_client, event_bus_reset):
        """Get non-existent simulation should return 404."""
        response = test_client.get("/api/v1/runs/nonexistent")
        assert response.status_code == 404

    def test_get_simulation_details(self, test_client, event_bus_reset):
        """Get simulation details should return event-derived info."""
        from server.event_bus import emit_event

        emit_event(
            "simulation_started",
            run_id="detail-test",
            max_steps=10,
            agent_names=["Agent A", "Agent B"]
        )
        emit_event("step_completed", run_id="detail-test", step=3)

        response = test_client.get("/api/v1/runs/detail-test")
        assert response.status_code == 200

        data = response.json()
        assert data["runId"] == "detail-test"
        assert data["status"] == "running"
        assert data["maxSteps"] == 10
        assert data["currentStep"] == 3
        assert len(data["agentNames"]) == 2

    def test_start_simulation_invalid_path(self, test_client, event_bus_reset):
        """Start simulation with invalid path should return 400."""
        response = test_client.post(
            "/api/v1/runs",
            json={"scenario_path": "/nonexistent/path.yaml"}
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


class TestQueueEndpoints:
    """Tests for job queue endpoints."""

    def test_queue_stats(self, test_client, event_bus_reset):
        """Queue stats endpoint should return queue information."""
        response = test_client.get("/api/queue/stats")
        assert response.status_code == 200

        data = response.json()
        # Should have stats for each priority queue plus total
        assert "high" in data
        assert "normal" in data
        assert "low" in data
        assert "total" in data


class TestRunIdConsistency:
    """Tests for run ID consistency across the system."""

    def test_events_have_consistent_run_id(self, test_client, event_bus_reset):
        """All events for a run should have the same run_id."""
        from server.event_bus import emit_event, get_event_bus

        run_id = "consistency-test"
        emit_event("simulation_started", run_id=run_id)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("agent_response", run_id=run_id, step=1, agent_name="A")

        bus = get_event_bus()
        history = bus.get_history(run_id)

        assert len(history) == 3
        for event in history:
            assert event.run_id == run_id
