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
        """List simulations should return runs tracked in state manager."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create run in state manager (required for new architecture)
        create_test_run("test-run-1", scenario_name="test-scenario", max_steps=5)

        # Emit events for additional detail
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
        from tests.integration.conftest import create_test_run

        # Create run in state manager first
        create_test_run("detail-test", scenario_name="detail-scenario", max_steps=10, current_step=3)

        # Emit events for additional detail (agent names, etc.)
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


class TestPauseResumeEndpoints:
    """Tests for pause/resume simulation endpoints."""

    def test_pause_simulation_endpoint(self, test_client, event_bus_reset):
        """Test POST /api/v1/runs/{run_id}/pause endpoint."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create a running simulation in state manager
        create_test_run("pause_test", max_steps=10)
        emit_event("simulation_started", run_id="pause_test", max_steps=10)

        response = test_client.post("/api/v1/runs/pause_test/pause")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopping"  # State transitions to "stopping" when pause requested
        assert data["run_id"] == "pause_test"

    def test_pause_non_running_simulation(self, test_client, event_bus_reset):
        """Test that pausing a non-running simulation fails with 404."""
        response = test_client.post("/api/v1/runs/nonexistent/pause")
        assert response.status_code == 404

    def test_pause_completed_simulation(self, test_client, event_bus_reset):
        """Test that pausing a completed simulation fails with 409."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create a completed simulation in state manager
        create_test_run("completed_sim", status="completed", max_steps=5)
        emit_event("simulation_started", run_id="completed_sim", max_steps=5)
        emit_event("simulation_completed", run_id="completed_sim")

        response = test_client.post("/api/v1/runs/completed_sim/pause")
        assert response.status_code == 409

    def test_resume_simulation_endpoint(self, test_client, event_bus_reset, tmp_path):
        """Test POST /api/v1/runs/{run_id}/resume endpoint."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run
        from unittest.mock import patch, MagicMock
        import json
        import os

        run_id = "resume_test"

        # Create a paused simulation in state manager
        create_test_run(run_id, status="paused", max_steps=10)
        emit_event("simulation_started", run_id=run_id, max_steps=10)
        emit_event("simulation_paused", run_id=run_id, step=5)

        # Create actual state file in the results directory
        results_dir = Path("results") / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        state_file = results_dir / "state.json"
        state_file.write_text(json.dumps({
            "run_id": run_id,
            "scenario_path": "/path/to/scenario.yaml",
            "snapshots": [{"step": 5, "game_state": {}}]
        }))

        try:
            # Mock enqueue_simulation at the job_queue module level
            with patch("server.job_queue.enqueue_simulation") as mock_enqueue:
                mock_enqueue.return_value = "job_123"

                response = test_client.post(f"/api/v1/runs/{run_id}/resume")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "resumed"
            assert data["run_id"] == run_id
            assert data["resuming_from_step"] == 6
        finally:
            # Cleanup
            import shutil
            if results_dir.exists():
                shutil.rmtree(results_dir)

    def test_resume_non_paused_simulation(self, test_client, event_bus_reset):
        """Test that resuming a non-paused simulation fails with 409."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create a running (not paused) simulation
        create_test_run("running_sim", status="running", max_steps=10)
        emit_event("simulation_started", run_id="running_sim", max_steps=10)

        response = test_client.post("/api/v1/runs/running_sim/resume")
        assert response.status_code == 409

    def test_resume_nonexistent_simulation(self, test_client, event_bus_reset):
        """Test that resuming a non-existent simulation fails with 404."""
        response = test_client.post("/api/v1/runs/nonexistent/resume")
        assert response.status_code == 404
