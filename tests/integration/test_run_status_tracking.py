"""
Tests for run status tracking via RunStateManager.

These tests verify that simulation run status is correctly tracked
throughout the simulation lifecycle using RunStateManager as the
single source of truth:
- When simulations complete successfully
- When simulations fail with errors
- When worker processes crash or timeout (stale run detection)

Architecture: RunStateManager (Redis-backed) is the authoritative source
for run status. EventBus is used for detailed event data only.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestRunStatusAfterCompletion:
    """Tests verifying run status is updated when simulations complete."""

    def test_status_becomes_completed_after_transition(
        self, test_client, event_bus_reset
    ):
        """Status should be 'completed' after RunStateManager transition."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "status-test-completed"

        # Create run and transition to completed via state manager
        create_test_run(run_id, status="completed", max_steps=3)

        # Emit simulation lifecycle events (for detailed event data)
        emit_event("simulation_started", run_id=run_id, max_steps=3, num_agents=2)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("step_completed", run_id=run_id, step=2)
        emit_event("step_completed", run_id=run_id, step=3)
        emit_event("simulation_completed", run_id=run_id, step=3, total_steps=3)

        # Check status via API
        response = test_client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["runId"] == run_id
        assert data["status"] == "completed"

    def test_status_becomes_failed_after_transition(
        self, test_client, event_bus_reset
    ):
        """Status should be 'failed' after RunStateManager transition."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "status-test-failed"

        # Create run and transition to failed via state manager
        create_test_run(run_id, status="failed", max_steps=5)

        emit_event("simulation_started", run_id=run_id, max_steps=5, num_agents=1)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("simulation_failed", run_id=run_id, error="Test error message")

        response = test_client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "failed"

    def test_list_simulations_shows_correct_status_after_completion(
        self, test_client, event_bus_reset
    ):
        """List simulations endpoint should show correct status for all runs."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create runs with different statuses via state manager
        create_test_run("run-running", status="running", max_steps=10, current_step=1)
        create_test_run("run-completed", status="completed", max_steps=3)
        create_test_run("run-failed", status="failed", max_steps=5)

        # Emit events for detail data
        emit_event("simulation_started", run_id="run-running", max_steps=10, num_agents=1)
        emit_event("step_completed", run_id="run-running", step=1)

        emit_event("simulation_started", run_id="run-completed", max_steps=3, num_agents=1)
        emit_event("simulation_completed", run_id="run-completed", step=3, total_steps=3)

        emit_event("simulation_started", run_id="run-failed", max_steps=5, num_agents=1)
        emit_event("simulation_failed", run_id="run-failed", error="Crashed")

        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200

        data = response.json()
        runs_list = data.get("runs", [])
        runs = {r["runId"]: r["status"] for r in runs_list}

        assert runs.get("run-running") == "running"
        assert runs.get("run-completed") == "completed"
        assert runs.get("run-failed") == "failed"


class TestWorkerTaskStatusTracking:
    """Tests for status tracking when running via RQ worker task."""

    def test_worker_task_success_emits_completion_event(self, tmp_path):
        """Worker task should result in status being 'completed'."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "test_scenario.yaml"
        scenario_path.write_text("name: test\nmax_steps: 1\n")

        emitted_events = []

        def capture_emit(event_type, run_id, step=None, **data):
            emitted_events.append({
                "event_type": event_type,
                "run_id": run_id,
                "step": step,
                "data": data
            })

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"), \
             patch("server.event_bus.emit_event", side_effect=capture_emit):

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            # Simulate what the real orchestrator.run() would do
            def mock_run():
                # Real orchestrator emits SIMULATION_COMPLETED at the end
                capture_emit("simulation_completed", "worker-run-123", step=1, total_steps=1)

            mock_orchestrator.run.side_effect = mock_run

            result = run_simulation_task("worker-run-123", str(scenario_path))

            assert result["status"] == "completed"

            # Verify simulation_completed event was emitted
            completion_events = [e for e in emitted_events if e["event_type"] == "simulation_completed"]
            assert len(completion_events) >= 1

    def test_worker_task_failure_emits_failed_event(self, tmp_path):
        """Worker task should emit simulation_failed event on error."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "failing_scenario.yaml"
        scenario_path.write_text("name: test\n")

        emitted_events = []

        def capture_emit(event_type, run_id, step=None, **data):
            emitted_events.append({
                "event_type": event_type,
                "run_id": run_id,
                "data": data
            })

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"), \
             patch("server.event_bus.emit_event", side_effect=capture_emit):

            mock_orchestrator = MagicMock()
            mock_orchestrator.run.side_effect = RuntimeError("Simulation crashed")
            mock_orch_class.return_value = mock_orchestrator

            with pytest.raises(RuntimeError):
                run_simulation_task("failing-worker-run", str(scenario_path))

            # Verify simulation_failed event was emitted
            failure_events = [e for e in emitted_events if e["event_type"] == "simulation_failed"]
            assert len(failure_events) == 1
            assert failure_events[0]["data"].get("error") == "Simulation crashed"


class TestStaleRunDetection:
    """Tests for detecting and handling stale runs via RunStateManager."""

    def test_run_without_heartbeat_detected_as_stale(self, test_client, event_bus_reset):
        """A running run without heartbeat should be detected as stale."""
        from server.run_state import get_state_manager
        from tests.integration.conftest import create_test_run

        run_id = "stale-run-test"
        create_test_run(run_id, status="running", max_steps=5)

        state_manager = get_state_manager()

        # Without sending a heartbeat, the run should be stale
        assert state_manager.is_heartbeat_stale(run_id) is True

    def test_multiple_runs_track_status_independently(self, test_client, event_bus_reset):
        """Each run should track its status independently via RunStateManager."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        # Create two runs with different statuses
        create_test_run("run-A", status="completed", max_steps=3)
        create_test_run("run-B", status="running", max_steps=3, current_step=1)

        emit_event("simulation_started", run_id="run-A", max_steps=3, num_agents=1)
        emit_event("simulation_completed", run_id="run-A", step=3, total_steps=3)

        emit_event("simulation_started", run_id="run-B", max_steps=3, num_agents=1)

        # Check run-A is completed
        response_a = test_client.get("/api/v1/runs/run-A")
        assert response_a.json()["status"] == "completed"

        # Check run-B is still running
        response_b = test_client.get("/api/v1/runs/run-B")
        assert response_b.json()["status"] == "running"


class TestSingleSourceOfTruth:
    """Tests verifying RunStateManager is the single source of truth for status."""

    def test_runstatemanager_is_single_source_of_truth(self, test_client, event_bus_reset):
        """RunStateManager should be the only source of truth for run status."""
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run
        from server.run_state import get_state_manager

        run_id = "single-source-test"

        # Create run in state manager
        create_test_run(run_id, status="running", max_steps=5)

        # Emit events to EventBus
        emit_event("simulation_started", run_id=run_id, max_steps=5, num_agents=2)

        # Both endpoints should see running status from state manager
        response1 = test_client.get(f"/api/v1/runs/{run_id}")
        assert response1.status_code == 200
        assert response1.json()["status"] == "running"

        response2 = test_client.get("/api/v1/runs")
        runs = {r["runId"]: r for r in response2.json().get("runs", [])}
        assert run_id in runs
        assert runs[run_id]["status"] == "running"

        # Transition via state manager
        state_manager = get_state_manager()
        state_manager.transition(run_id, "completed")

        # Both endpoints should now show completed
        response3 = test_client.get(f"/api/v1/runs/{run_id}")
        assert response3.json()["status"] == "completed"

        response4 = test_client.get("/api/v1/runs")
        runs = {r["runId"]: r for r in response4.json().get("runs", [])}
        assert runs[run_id]["status"] == "completed"

    def test_state_manager_required_for_run_visibility(self, test_client, event_bus_reset):
        """Runs must be in RunStateManager to be visible via API."""
        from tests.integration.conftest import create_test_run

        run_id = "visibility-test"

        # Create run in state manager
        create_test_run(run_id, max_steps=3)

        # Should be visible via API
        response = test_client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["maxSteps"] == 3


class TestRunsApiStatusConsistency:
    """Tests for /api/v1/runs endpoint status consistency."""

    def test_runs_api_reflects_state_manager_status(self, test_client, event_bus_reset):
        """
        /api/v1/runs should show status from RunStateManager.
        """
        from server.event_bus import emit_event
        from tests.integration.conftest import create_test_run

        run_id = "api-runs-test"

        # Create completed run in state manager
        create_test_run(run_id, status="completed", max_steps=2)

        # Emit full simulation lifecycle
        emit_event("simulation_started", run_id=run_id, max_steps=2, num_agents=1,
                   scenario_name="test-scenario")
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("step_completed", run_id=run_id, step=2)
        emit_event("simulation_completed", run_id=run_id, step=2, total_steps=2)

        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200

        runs = response.json().get("runs", [])
        our_run = next((r for r in runs if r["runId"] == run_id), None)

        assert our_run is not None
        assert our_run["status"] == "completed"


class TestCrossProcessEventDelivery:
    """
    Tests for event delivery across process boundaries.

    With RunStateManager, state is stored in Redis which is shared between
    API server and worker processes. Events are still stored in EventBus
    for detailed history but status comes from RunStateManager.
    """

    def test_state_manager_shared_across_processes(self):
        """
        RunStateManager uses Redis as backend, enabling cross-process state sharing.

        This test verifies the state manager can be accessed from different
        contexts (simulating API server and worker processes).
        """
        from fakeredis import FakeRedis
        from server.run_state import RunStateManager

        # Create shared Redis (simulating real Redis shared between processes)
        shared_redis = FakeRedis(decode_responses=True)

        # "API server" initializes state manager
        RunStateManager.reset_instance()
        api_manager = RunStateManager.initialize(shared_redis)

        # API creates a run
        api_manager.create_run(
            run_id="cross-process-test",
            scenario_path="/path.yaml",
            scenario_name="test-scenario",
        )

        # Verify pending status
        state = api_manager.get_state("cross-process-test")
        assert state.status == "pending"

        # "Worker" (using same Redis instance) transitions the run
        api_manager.transition("cross-process-test", "running", worker_id="worker-1")

        # "API" can see the updated status
        state = api_manager.get_state("cross-process-test")
        assert state.status == "running"
        assert state.worker_id == "worker-1"

        # Worker completes
        api_manager.transition("cross-process-test", "completed")

        # API sees completed
        state = api_manager.get_state("cross-process-test")
        assert state.status == "completed"

        RunStateManager.reset_instance()

    def test_heartbeat_stale_detection_works(self):
        """
        Test that heartbeat expiration detection works.

        When a worker crashes, its heartbeat TTL expires and the run
        is detected as stale.
        """
        from fakeredis import FakeRedis
        from server.run_state import RunStateManager, KEY_PREFIX, HEARTBEAT_SUFFIX

        shared_redis = FakeRedis(decode_responses=True)

        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(shared_redis)

        # Create and start a run
        manager.create_run("stale-detection", "/path.yaml", "scenario")
        manager.transition("stale-detection", "running", worker_id="dying-worker")

        # Worker sends heartbeat
        manager.heartbeat("stale-detection", "dying-worker", step=1)

        # Heartbeat exists - not stale
        assert manager.is_heartbeat_stale("stale-detection") is False

        # Simulate heartbeat TTL expiration by deleting the key
        heartbeat_key = f"{KEY_PREFIX}stale-detection{HEARTBEAT_SUFFIX}"
        shared_redis.delete(heartbeat_key)

        # Now should be stale
        assert manager.is_heartbeat_stale("stale-detection") is True

        # check_stale_runs should find it
        stale_runs = manager.check_stale_runs()
        assert "stale-detection" in stale_runs

        RunStateManager.reset_instance()
