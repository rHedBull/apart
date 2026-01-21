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
        assert "active_simulations" in data
        assert "max_concurrent_simulations" in data
        assert data["max_concurrent_simulations"] == 4


class TestSimulationEndpoints:
    """Tests for simulation management endpoints."""

    def test_list_simulations_empty(self, test_client, event_bus_reset):
        """List simulations should return empty list initially."""
        response = test_client.get("/api/simulations")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_simulations_with_events(self, test_client, event_bus_reset):
        """List simulations should return runs that have events."""
        from server.event_bus import emit_event

        # Emit some events for a run
        emit_event("simulation_started", run_id="test-run-1", max_steps=5, num_agents=2)
        emit_event("step_completed", run_id="test-run-1", step=1)

        response = test_client.get("/api/simulations")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["run_id"] == "test-run-1"
        assert data[0]["status"] == "running"

    def test_get_simulation_not_found(self, test_client, event_bus_reset):
        """Get non-existent simulation should return 404."""
        response = test_client.get("/api/simulations/nonexistent")
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

        response = test_client.get("/api/simulations/detail-test")
        assert response.status_code == 200

        data = response.json()
        assert data["run_id"] == "detail-test"
        assert data["status"] == "running"
        assert data["max_steps"] == 10
        assert data["current_step"] == 3
        assert data["agent_count"] == 2

    def test_start_simulation_invalid_path(self, test_client, event_bus_reset):
        """Start simulation with invalid path should return 400."""
        response = test_client.post(
            "/api/simulations",
            json={"scenario_path": "/nonexistent/path.yaml"}
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


class TestConcurrencyLimits:
    """Tests for concurrency limiting behavior."""

    def test_detailed_health_shows_limits(self, test_client, event_bus_reset):
        """Health endpoint should show concurrency limits."""
        response = test_client.get("/api/health/detailed")
        data = response.json()

        assert data["max_concurrent_simulations"] == 4
        assert data["active_simulations"] == 0


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
