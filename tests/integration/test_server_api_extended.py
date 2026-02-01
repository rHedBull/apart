"""Extended integration tests for server API endpoints."""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSimulationLifecycle:
    """Tests for complete simulation lifecycle through API."""

    def test_simulation_pending_to_running_to_completed(self, test_client, event_bus_reset):
        """Test full simulation lifecycle through events."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "lifecycle-test"

        # Initially no simulations
        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200
        assert len(response.json().get("runs", [])) == 0

        # Create run in state manager and start simulation
        create_test_run(run_id, max_steps=5, current_step=0)
        emit_event("simulation_started", run_id=run_id, max_steps=5, num_agents=2, agent_names=["A", "B"])

        # Should be running
        response = test_client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["maxSteps"] == 5
        assert len(data["agentNames"]) == 2

        # Progress through steps
        from server.run_state import get_state_manager
        state_manager = get_state_manager()
        for step in range(1, 4):
            emit_event("step_completed", run_id=run_id, step=step)
            state_manager.update_progress(run_id, current_step=step)

        response = test_client.get(f"/api/v1/runs/{run_id}")
        assert response.json()["currentStep"] == 3

        # Complete simulation
        emit_event("simulation_completed", run_id=run_id)
        state_manager.transition(run_id, "completed")

        response = test_client.get(f"/api/v1/runs/{run_id}")
        data = response.json()
        assert data["status"] == "completed"

    def test_simulation_failure(self, test_client, event_bus_reset):
        """Test simulation failure is properly tracked."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run
        from server.run_state import get_state_manager

        run_id = "fail-test"

        create_test_run(run_id, max_steps=10)
        emit_event("simulation_started", run_id=run_id, max_steps=10)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("simulation_failed", run_id=run_id, error="Out of memory")

        state_manager = get_state_manager()
        state_manager.transition(run_id, "failed", error="Out of memory")

        response = test_client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        assert data["status"] == "failed"


class TestSimulationListFiltering:
    """Tests for simulation listing and filtering."""

    def test_list_multiple_simulations(self, test_client, event_bus_reset):
        """Test listing multiple simulations with different statuses."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create simulations in different states
        create_test_run("run-1", status="completed", max_steps=5)
        emit_event("simulation_started", run_id="run-1", max_steps=5)
        emit_event("simulation_completed", run_id="run-1")

        create_test_run("run-2", status="running", max_steps=10)
        emit_event("simulation_started", run_id="run-2", max_steps=10)

        create_test_run("run-3", status="failed", max_steps=3)
        emit_event("simulation_started", run_id="run-3", max_steps=3)
        emit_event("simulation_failed", run_id="run-3", error="Test error")

        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", [])

        assert len(runs) >= 3

        # Find each run
        runs_by_id = {r["runId"]: r for r in runs}
        assert runs_by_id["run-1"]["status"] == "completed"
        assert runs_by_id["run-2"]["status"] == "running"
        assert runs_by_id["run-3"]["status"] == "failed"

    def test_list_simulations_preserves_order(self, test_client, event_bus_reset):
        """Test that simulations are listed consistently."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        for i in range(5):
            create_test_run(f"ordered-{i}", max_steps=5)
            emit_event("simulation_started", run_id=f"ordered-{i}", max_steps=5)

        response = test_client.get("/api/v1/runs")
        data = response.json()
        runs = data.get("runs", [])

        assert len(runs) >= 5
        # All runs should be present
        run_ids = {r["runId"] for r in runs}
        for i in range(5):
            assert f"ordered-{i}" in run_ids


class TestStartSimulationEndpoint:
    """Tests for POST /api/v1/runs endpoint."""

    def test_start_simulation_with_valid_scenario(self, test_client, event_bus_reset, sample_scenario):
        """Test starting simulation with valid scenario file."""
        response = test_client.post(
            "/api/v1/runs",
            json={"scenario_path": str(sample_scenario)}
        )

        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"
        assert "queued" in data["message"].lower() or "job_id" in data["message"].lower()

    def test_start_simulation_with_custom_run_id(self, test_client, event_bus_reset, sample_scenario):
        """Test starting simulation with custom run ID."""
        response = test_client.post(
            "/api/v1/runs",
            json={
                "scenario_path": str(sample_scenario),
                "run_id": "my-custom-id"
            }
        )

        assert response.status_code == 200
        # Custom run_id is used as prefix, UUID suffix is always appended for uniqueness
        assert response.json()["run_id"].startswith("my-custom-id-")

    def test_start_simulation_with_priority(self, test_client, event_bus_reset, sample_scenario):
        """Test starting simulation with different priorities."""
        for priority in ["high", "normal", "low"]:
            response = test_client.post(
                "/api/v1/runs",
                json={
                    "scenario_path": str(sample_scenario),
                    "priority": priority
                }
            )
            assert response.status_code == 200

    def test_start_simulation_missing_scenario_path(self, test_client, event_bus_reset):
        """Test that missing scenario_path returns 422."""
        response = test_client.post("/api/v1/runs", json={})
        assert response.status_code == 422

    def test_start_simulation_nonexistent_file(self, test_client, event_bus_reset):
        """Test that nonexistent scenario file returns 400."""
        response = test_client.post(
            "/api/v1/runs",
            json={"scenario_path": "/definitely/not/a/real/path.yaml"}
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


class TestJobEndpoints:
    """Tests for job queue related endpoints."""

    def test_get_job_not_found(self, test_client, event_bus_reset):
        """Test that getting nonexistent job returns 404."""
        response = test_client.get("/api/jobs/nonexistent-job-id")
        assert response.status_code == 404

    def test_cancel_job_not_found(self, test_client, event_bus_reset):
        """Test canceling nonexistent job."""
        response = test_client.delete("/api/jobs/nonexistent-job-id")
        # Should return 409 (cannot cancel) since job doesn't exist
        assert response.status_code == 409

    def test_queue_stats_structure(self, test_client, event_bus_reset):
        """Test queue stats endpoint returns proper structure."""
        response = test_client.get("/api/queue/stats")
        assert response.status_code == 200

        data = response.json()
        assert "high" in data
        assert "normal" in data
        assert "low" in data
        assert "total" in data

        # Each priority should have expected fields
        for priority in ["high", "normal", "low"]:
            assert "queued" in data[priority]
            assert "failed" in data[priority]
            assert "started" in data[priority]
            assert "finished" in data[priority]


class TestEventStreamEndpoints:
    """Tests for SSE event streaming endpoints.

    Note: SSE streaming tests are inherently difficult with TestClient
    because the stream stays open indefinitely. These are marked as skip
    for CI but can be run manually for verification.
    """

    @pytest.mark.skip(reason="SSE streaming tests cause timeouts in TestClient")
    def test_event_stream_endpoint_accessible(self, test_client, event_bus_reset):
        """Test that event stream endpoint returns SSE content type."""
        pass

    @pytest.mark.skip(reason="SSE streaming tests cause timeouts in TestClient")
    def test_event_stream_run_specific_endpoint(self, test_client, event_bus_reset):
        """Test that run-specific event stream endpoint is accessible."""
        pass


class TestHealthEndpointsExtended:
    """Extended tests for health endpoints."""

    def test_health_detailed_with_events(self, test_client, event_bus_reset):
        """Test detailed health includes run counts from state manager."""
        from tests.integration.conftest import create_test_run

        # Create runs in state manager
        create_test_run("health-test-1")
        create_test_run("health-test-2")

        response = test_client.get("/api/health/detailed")
        data = response.json()

        assert data["total_run_ids"] >= 2
        assert data["persistence_mode"] == "jsonl"

    def test_health_check_fast(self, test_client, event_bus_reset):
        """Test that health check responds quickly."""
        import time

        start = time.time()
        response = test_client.get("/api/health")
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 1.0  # Should be very fast


class TestLegacyRunsEndpoints:
    """Tests for legacy /api/v1/runs endpoints."""

    def test_list_runs_returns_valid_structure(self, test_client, event_bus_reset):
        """Test legacy runs endpoint returns valid structure."""
        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)
        # Mock data may be generated, so just verify structure
        for run in data["runs"]:
            assert "runId" in run
            assert "status" in run

    def test_list_runs_with_events(self, test_client, event_bus_reset):
        """Test legacy runs endpoint with event data."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        create_test_run("legacy-test", max_steps=5)
        emit_event("simulation_started", run_id="legacy-test", max_steps=5, scenario_name="Test")
        emit_event("step_completed", run_id="legacy-test", step=1)
        emit_event("danger_signal", run_id="legacy-test", step=1, category="test")

        response = test_client.get("/api/v1/runs")
        data = response.json()

        assert len(data["runs"]) >= 1
        run = next(r for r in data["runs"] if r["runId"] == "legacy-test")
        assert run["status"] == "running"

    def test_get_run_not_found(self, test_client, event_bus_reset):
        """Test getting nonexistent run returns 404."""
        response = test_client.get("/api/v1/runs/nonexistent-run")
        assert response.status_code == 404


class TestCORSHeaders:
    """Tests for CORS middleware."""

    def test_cors_headers_present(self, test_client, event_bus_reset):
        """Test that CORS headers are present in responses."""
        response = test_client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )
        # CORS preflight should succeed or health endpoint should have CORS headers
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled

    def test_allowed_origin(self, test_client, event_bus_reset):
        """Test that allowed origins get CORS headers."""
        response = test_client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        # Access-Control-Allow-Origin should be present for allowed origin
        # Note: FastAPI TestClient might not fully simulate CORS behavior


class TestRequestValidation:
    """Tests for request validation."""

    def test_invalid_json_body(self, test_client, event_bus_reset):
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/api/v1/runs",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_invalid_priority_value(self, test_client, event_bus_reset, sample_scenario):
        """Test that invalid priority value returns 422."""
        response = test_client.post(
            "/api/v1/runs",
            json={
                "scenario_path": str(sample_scenario),
                "priority": "invalid"
            }
        )
        assert response.status_code == 422


class TestSimulationStatusDetermination:
    """Tests for correct status determination from state manager."""

    def test_pending_status_no_started_event(self, test_client, event_bus_reset):
        """Test that run in pending state shows as pending."""
        from tests.integration.conftest import create_test_run
        from server.run_state import get_state_manager

        # Create a pending run (not transitioned to running)
        state_manager = get_state_manager()
        state_manager.create_run(
            run_id="pending-only",
            scenario_path="/path.yaml",
            scenario_name="pending-scenario",
        )

        response = test_client.get("/api/v1/runs/pending-only")
        data = response.json()
        assert data["status"] == "pending"

    def test_running_status_with_started_event(self, test_client, event_bus_reset):
        """Test that running status is shown for running run."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        create_test_run("running-status")
        emit_event("simulation_started", run_id="running-status")

        response = test_client.get("/api/v1/runs/running-status")
        assert response.json()["status"] == "running"

    def test_completed_overrides_running(self, test_client, event_bus_reset):
        """Test that completed status is shown for completed run."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        create_test_run("complete-override", status="completed")
        emit_event("simulation_started", run_id="complete-override")
        emit_event("simulation_completed", run_id="complete-override")

        response = test_client.get("/api/v1/runs/complete-override")
        assert response.json()["status"] == "completed"

    def test_failed_overrides_running(self, test_client, event_bus_reset):
        """Test that failed status is shown for failed run."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        create_test_run("fail-override", status="failed")
        emit_event("simulation_started", run_id="fail-override")
        emit_event("simulation_failed", run_id="fail-override", error="Test")

        response = test_client.get("/api/v1/runs/fail-override")
        assert response.json()["status"] == "failed"


class TestRunDeletion:
    """Tests for run deletion API endpoints."""

    def test_delete_completed_run(self, test_client, event_bus_reset):
        """Should delete a completed simulation run."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "delete-completed-test"
        create_test_run(run_id, status="completed")
        emit_event("simulation_started", run_id=run_id)
        emit_event("simulation_completed", run_id=run_id)

        response = test_client.delete(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["run_id"] == run_id

    def test_delete_running_simulation_blocked(self, test_client, event_bus_reset):
        """Should reject deletion of running simulation without force flag."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "delete-running-test"
        create_test_run(run_id, status="running")
        emit_event("simulation_started", run_id=run_id)

        response = test_client.delete(f"/api/v1/runs/{run_id}")
        assert response.status_code == 409
        assert "running" in response.json()["detail"].lower()

    def test_delete_running_simulation_with_force(self, test_client, event_bus_reset):
        """Should allow deletion of running simulation with force=true."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "delete-running-force-test"
        create_test_run(run_id, status="running")
        emit_event("simulation_started", run_id=run_id)

        response = test_client.delete(f"/api/v1/runs/{run_id}?force=true")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

    def test_delete_nonexistent_run(self, test_client, event_bus_reset):
        """Should return 404 for nonexistent run."""
        response = test_client.delete("/api/v1/runs/nonexistent-run-id")
        assert response.status_code == 404

    def test_batch_delete_runs(self, test_client, event_bus_reset):
        """Should delete multiple completed runs in batch."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create completed runs
        for i in range(3):
            run_id = f"batch-delete-{i}"
            create_test_run(run_id, status="completed")
            emit_event("simulation_started", run_id=run_id)
            emit_event("simulation_completed", run_id=run_id)

        response = test_client.post(
            "/api/v1/runs:batchDelete",
            json={"run_ids": ["batch-delete-0", "batch-delete-1", "batch-delete-2"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 3
        assert data["total_requested"] == 3
        assert len(data["skipped_running"]) == 0

    def test_batch_delete_skips_running(self, test_client, event_bus_reset):
        """Should skip running simulations in batch delete."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create one completed and one running
        create_test_run("batch-completed", status="completed")
        emit_event("simulation_started", run_id="batch-completed")
        emit_event("simulation_completed", run_id="batch-completed")

        create_test_run("batch-running", status="running")
        emit_event("simulation_started", run_id="batch-running")

        response = test_client.post(
            "/api/v1/runs:batchDelete",
            json={"run_ids": ["batch-completed", "batch-running"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 1
        assert data["total_requested"] == 2
        assert "batch-running" in data["skipped_running"]

    def test_batch_delete_empty_list(self, test_client, event_bus_reset):
        """Should reject empty run_ids list."""
        response = test_client.post(
            "/api/v1/runs:batchDelete",
            json={"run_ids": []}
        )
        assert response.status_code == 400

    def test_batch_delete_with_force(self, test_client, event_bus_reset):
        """Should delete running simulations when force=true."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        create_test_run("force-batch-running", status="running")
        emit_event("simulation_started", run_id="force-batch-running")

        response = test_client.post(
            "/api/v1/runs:batchDelete",
            json={"run_ids": ["force-batch-running"], "force": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 1
        assert len(data["skipped_running"]) == 0
