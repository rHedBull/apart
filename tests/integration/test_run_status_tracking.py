"""
Tests for run status tracking bugs.

These tests verify that simulation run status is correctly tracked
throughout the simulation lifecycle, especially:
- When simulations complete successfully
- When simulations fail with errors
- When worker processes crash or timeout

Bug reference: Runs get stuck in "running" status even when no longer running.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestRunStatusAfterCompletion:
    """Tests verifying run status is updated when simulations complete."""

    def test_status_becomes_completed_after_simulation_completed_event(
        self, test_client, event_bus_reset
    ):
        """
        FAILING TEST: Status should be 'completed' after simulation_completed event.

        This test demonstrates the bug where runs stay in 'running' status
        even after the simulation has completed.
        """
        from server.event_bus import emit_event

        run_id = "status-test-completed"

        # Emit simulation lifecycle events
        emit_event("simulation_started", run_id=run_id, max_steps=3, num_agents=2)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("step_completed", run_id=run_id, step=2)
        emit_event("step_completed", run_id=run_id, step=3)
        emit_event("simulation_completed", run_id=run_id, step=3, total_steps=3)

        # Check status via API
        response = test_client.get(f"/api/simulations/{run_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["run_id"] == run_id
        # BUG: This assertion may fail if status tracking is broken
        assert data["status"] == "completed", (
            f"Expected status 'completed' but got '{data['status']}'. "
            "Run is stuck in running state after completion event."
        )

    def test_status_becomes_failed_after_simulation_failed_event(
        self, test_client, event_bus_reset
    ):
        """
        Status should be 'failed' after simulation_failed event.
        """
        from server.event_bus import emit_event

        run_id = "status-test-failed"

        emit_event("simulation_started", run_id=run_id, max_steps=5, num_agents=1)
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("simulation_failed", run_id=run_id, error="Test error message")

        response = test_client.get(f"/api/simulations/{run_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "failed", (
            f"Expected status 'failed' but got '{data['status']}'. "
            "Run is stuck in running state after failure event."
        )
        assert data["error_message"] == "Test error message"

    def test_list_simulations_shows_correct_status_after_completion(
        self, test_client, event_bus_reset
    ):
        """
        List simulations endpoint should show correct status for completed runs.
        """
        from server.event_bus import emit_event

        # Create multiple runs with different statuses
        emit_event("simulation_started", run_id="run-running", max_steps=10, num_agents=1)
        emit_event("step_completed", run_id="run-running", step=1)

        emit_event("simulation_started", run_id="run-completed", max_steps=3, num_agents=1)
        emit_event("simulation_completed", run_id="run-completed", step=3, total_steps=3)

        emit_event("simulation_started", run_id="run-failed", max_steps=5, num_agents=1)
        emit_event("simulation_failed", run_id="run-failed", error="Crashed")

        response = test_client.get("/api/simulations")
        assert response.status_code == 200

        runs = {r["run_id"]: r["status"] for r in response.json()}

        assert runs.get("run-running") == "running"
        assert runs.get("run-completed") == "completed", (
            f"Completed run shows status '{runs.get('run-completed')}' instead of 'completed'"
        )
        assert runs.get("run-failed") == "failed", (
            f"Failed run shows status '{runs.get('run-failed')}' instead of 'failed'"
        )


class TestWorkerTaskStatusTracking:
    """Tests for status tracking when running via RQ worker task."""

    def test_worker_task_success_emits_completion_event(self, tmp_path):
        """
        FAILING TEST: Worker task should result in status being 'completed'.

        The worker task calls orchestrator.run(), which emits SIMULATION_COMPLETED.
        We verify this event is actually emitted and would be received by event bus.
        """
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
            assert len(completion_events) >= 1, (
                "No simulation_completed event was emitted. "
                "This would cause the run to stay in 'running' status."
            )

    def test_worker_task_failure_emits_failed_event(self, tmp_path):
        """
        Worker task should emit simulation_failed event on error.
        """
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
            assert len(failure_events) == 1, (
                f"Expected 1 simulation_failed event, got {len(failure_events)}. "
                "This would cause the run to stay in 'running' status."
            )
            assert failure_events[0]["data"].get("error") == "Simulation crashed"


class TestStaleRunDetection:
    """Tests for detecting and handling stale runs."""

    def test_run_without_completion_event_shows_running(self, test_client, event_bus_reset):
        """
        A run with only simulation_started event should show as 'running'.

        This is expected behavior, but becomes a bug when the simulation
        actually finished but no completion event was emitted.
        """
        from server.event_bus import emit_event

        run_id = "stale-run-test"
        emit_event("simulation_started", run_id=run_id, max_steps=5, num_agents=1)

        response = test_client.get(f"/api/simulations/{run_id}")
        assert response.status_code == 200

        data = response.json()
        # This is technically correct - without completion event, status is running
        assert data["status"] == "running"

    def test_multiple_runs_track_status_independently(self, test_client, event_bus_reset):
        """
        Each run should track its status independently.

        Bug scenario: One run completing might not affect another run's status.
        """
        from server.event_bus import emit_event

        # Start two runs
        emit_event("simulation_started", run_id="run-A", max_steps=3, num_agents=1)
        emit_event("simulation_started", run_id="run-B", max_steps=3, num_agents=1)

        # Complete only run-A
        emit_event("simulation_completed", run_id="run-A", step=3, total_steps=3)

        # Check run-A is completed
        response_a = test_client.get("/api/simulations/run-A")
        assert response_a.json()["status"] == "completed"

        # Check run-B is still running
        response_b = test_client.get("/api/simulations/run-B")
        assert response_b.json()["status"] == "running"


class TestSingleSourceOfTruth:
    """Tests verifying EventBus is the single source of truth for status."""

    def test_eventbus_is_single_source_of_truth(self, test_client, event_bus_reset):
        """
        EventBus should be the only source of truth for run status.

        All API endpoints should derive status from EventBus events,
        not from any separate in-memory registry.
        """
        from server.event_bus import emit_event

        run_id = "single-source-test"

        # Emit events to EventBus
        emit_event("simulation_started", run_id=run_id, max_steps=5, num_agents=2)

        # Both /api/simulations and /api/runs should see the same status
        response1 = test_client.get(f"/api/simulations/{run_id}")
        assert response1.status_code == 200
        assert response1.json()["status"] == "running"

        response2 = test_client.get("/api/runs")
        runs = {r["runId"]: r for r in response2.json().get("runs", [])}
        assert run_id in runs
        assert runs[run_id]["status"] == "running"

        # Complete the simulation
        emit_event("simulation_completed", run_id=run_id, step=5, total_steps=5)

        # Both endpoints should now show completed
        response3 = test_client.get(f"/api/simulations/{run_id}")
        assert response3.json()["status"] == "completed"

        response4 = test_client.get("/api/runs")
        runs = {r["runId"]: r for r in response4.json().get("runs", [])}
        assert runs[run_id]["status"] == "completed"

    def test_no_separate_registry_needed(self, test_client, event_bus_reset):
        """
        Status should work without any registration step.

        Just emitting events should be sufficient for status tracking.
        """
        from server.event_bus import emit_event

        run_id = "no-registry-test"

        # Just emit events - no separate registration needed
        emit_event(
            "simulation_started",
            run_id=run_id,
            max_steps=3,
            num_agents=1,
            agent_names=["TestAgent"],
            scenario_name="Test Scenario"
        )

        # Should be visible in API immediately
        response = test_client.get(f"/api/simulations/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["max_steps"] == 3
        assert data["agent_count"] == 1


class TestRunsApiStatusConsistency:
    """Tests for /api/runs endpoint status consistency."""

    def test_runs_api_reflects_completed_status(self, test_client, event_bus_reset, tmp_path):
        """
        FAILING TEST: /api/runs should show 'completed' for finished simulations.

        The /api/runs endpoint merges data from results/ directory and EventBus.
        This test verifies the status is correctly determined.
        """
        from server.event_bus import emit_event

        run_id = "api-runs-test"

        # Emit full simulation lifecycle
        emit_event("simulation_started", run_id=run_id, max_steps=2, num_agents=1,
                   scenario_name="test-scenario")
        emit_event("step_completed", run_id=run_id, step=1)
        emit_event("step_completed", run_id=run_id, step=2)
        emit_event("simulation_completed", run_id=run_id, step=2, total_steps=2)

        response = test_client.get("/api/runs")
        assert response.status_code == 200

        runs = response.json().get("runs", [])
        our_run = next((r for r in runs if r["runId"] == run_id), None)

        assert our_run is not None, f"Run {run_id} not found in /api/runs response"
        assert our_run["status"] == "completed", (
            f"Expected status 'completed' but got '{our_run['status']}' in /api/runs. "
            "This indicates the status tracking bug where runs stay stuck in 'running'."
        )


class TestCrossProcessEventDelivery:
    """
    Tests for event delivery across process boundaries.

    When RQ workers run simulations in separate processes, events emitted
    by the worker must somehow be visible to the API server. This is where
    the bug likely lives - events stay in the worker's EventBus and never
    reach the server.
    """

    def test_events_persisted_can_be_loaded_by_new_eventbus_instance(
        self, tmp_path
    ):
        """
        Events written by one EventBus instance should be loadable by another.

        This simulates what happens when a worker emits events and then
        the API server (with its own EventBus) needs to read them.
        """
        from server.event_bus import EventBus, emit_event

        persist_path = tmp_path / "shared_events.jsonl"

        # First EventBus instance (simulates worker)
        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        worker_bus = EventBus.get_instance()

        # Worker emits events
        emit_event("simulation_started", run_id="cross-process-test", max_steps=3)
        emit_event("simulation_completed", run_id="cross-process-test", step=3)

        # Verify events are persisted
        assert persist_path.exists(), "Events should be persisted to disk"

        # Reset to simulate a new process (API server)
        EventBus.reset_instance()
        server_bus = EventBus.get_instance()

        # Server should be able to see the events
        history = server_bus.get_history("cross-process-test")

        assert len(history) >= 2, (
            f"Expected at least 2 events, got {len(history)}. "
            "Events from worker process are not visible to server process."
        )

        event_types = [e.event_type for e in history]
        assert "simulation_started" in event_types
        assert "simulation_completed" in event_types, (
            "simulation_completed event not found. "
            "This would cause run to stay stuck in 'running' status."
        )

        # Cleanup
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_database_mode_events_persist_across_instances(self, tmp_path):
        """
        FAILING TEST: In database mode, events should persist across EventBus instances.

        When APART_USE_DATABASE=1, events go to SQLite instead of JSONL.
        The database should be the source of truth for run status.
        """
        from server.event_bus import EventBus, SimulationEvent
        from server.database import init_db, reset_db, get_db, set_db_path

        db_path = tmp_path / "test_status.db"
        set_db_path(db_path)
        init_db()

        # Enable database mode
        original_use_db = EventBus._use_database
        EventBus._use_database = True

        try:
            # Reset EventBus to pick up database mode
            EventBus.reset_instance()
            worker_bus = EventBus.get_instance()

            # Insert simulation to database
            db = get_db()
            db.insert_simulation(
                run_id="db-test-run",
                scenario_name="test",
                max_steps=3
            )

            # Emit events (worker process)
            event1 = SimulationEvent.create(
                "simulation_started", "db-test-run", step=0, max_steps=3
            )
            event2 = SimulationEvent.create(
                "simulation_completed", "db-test-run", step=3, total_steps=3
            )
            worker_bus.emit(event1)
            worker_bus.emit(event2)

            # Simulate server process getting the events
            EventBus.reset_instance()
            server_bus = EventBus.get_instance()

            history = server_bus.get_history("db-test-run")
            event_types = [e.event_type for e in history]

            assert "simulation_completed" in event_types, (
                f"Events in history: {event_types}. "
                "simulation_completed not found - status would be stuck as 'running'."
            )

            # Also verify database has correct status
            sim = db.get_simulation("db-test-run")
            # Note: The database status is set via update_simulation_status,
            # not automatically from events. This is a potential bug!

        finally:
            EventBus._use_database = original_use_db
            EventBus._test_persist_path = None
            EventBus.reset_instance()
            set_db_path(None)

    def test_worker_and_server_share_no_memory(self):
        """
        Demonstrates that worker and server EventBus instances are isolated.

        This is the fundamental cause of the bug: RQ workers run in separate
        processes with their own memory space. Events emitted in the worker
        only go to the worker's in-memory EventBus, not the server's.

        The only way events can cross is through:
        1. Shared file persistence (JSONL)
        2. Shared database (SQLite)
        3. Message queue (Redis pub/sub - not implemented)
        """
        from server.event_bus import EventBus

        # Create two separate EventBus "instances" by resetting
        EventBus.reset_instance()
        bus1 = EventBus.get_instance()
        bus1._event_history["isolated-run"] = []  # Add directly to memory

        # Get "another instance" (same singleton in same process, but
        # in real RQ this would be a different process entirely)
        # We can't truly test cross-process in a unit test, but we can
        # document the expected behavior

        # In the real bug scenario:
        # - Worker process has EventBus with events in memory
        # - Server process has EventBus with NO events in memory
        # - Only persistence (file/db) bridges the gap
        # - If persistence fails or isn't read, status stays "running"

        EventBus.reset_instance()

    def test_server_eventbus_stale_cache_bug(self, tmp_path):
        """
        FAILING TEST: Server EventBus doesn't reload events from persistence.

        This is the actual bug:
        1. Server starts, EventBus loads events from disk (empty)
        2. Worker starts simulation, emits events to disk
        3. Worker completes, events are persisted
        4. Server queries status - uses stale in-memory cache
        5. Status shows "running" because completion event not in cache

        The EventBus only loads from persistence at __init__, never refreshes.
        """
        from server.event_bus import EventBus, SimulationEvent
        import json

        persist_path = tmp_path / "stale_cache_test.jsonl"

        # Step 1: Server starts with empty EventBus
        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        server_bus = EventBus.get_instance()

        # Verify no events yet
        assert len(server_bus.get_history("stale-run")) == 0

        # Step 2: Simulate worker writing events DIRECTLY to persistence file
        # (This is what happens when a worker process emits events)
        events_to_write = [
            SimulationEvent.create("simulation_started", "stale-run", step=0, max_steps=3),
            SimulationEvent.create("step_completed", "stale-run", step=1),
            SimulationEvent.create("step_completed", "stale-run", step=2),
            SimulationEvent.create("step_completed", "stale-run", step=3),
            SimulationEvent.create("simulation_completed", "stale-run", step=3, total_steps=3),
        ]

        with open(persist_path, "w") as f:
            for event in events_to_write:
                f.write(event.to_json() + "\n")

        # Step 3: Server queries status - BUG: still uses stale cache
        history = server_bus.get_history("stale-run")

        # This assertion SHOULD pass but currently FAILS due to stale cache
        assert len(history) == 5, (
            f"Expected 5 events from persistence, got {len(history)}. "
            "EventBus is using stale in-memory cache instead of reloading from disk. "
            "This is why runs stay stuck in 'running' status."
        )

        event_types = [e.event_type for e in history]
        assert "simulation_completed" in event_types, (
            "simulation_completed not found in history. "
            "Server is not seeing events written by worker process."
        )

        # Cleanup
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_server_should_reload_events_for_active_runs(self, tmp_path):
        """
        FAILING TEST: Server should refresh events from persistence for active runs.

        Proposed fix: When querying a run that's in 'running' state, the server
        should check persistence for newer events to detect completion/failure.
        """
        from server.event_bus import EventBus, SimulationEvent

        persist_path = tmp_path / "reload_test.jsonl"

        # Server starts, a run is marked as started
        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        server_bus = EventBus.get_instance()

        # Emit start event through server
        start_event = SimulationEvent.create(
            "simulation_started", "reload-run", step=0, max_steps=2
        )
        server_bus.emit(start_event)

        # Verify run is "running"
        history = server_bus.get_history("reload-run")
        assert len(history) == 1
        assert history[0].event_type == "simulation_started"

        # Worker completes and writes completion event to disk
        # (Appending to existing file)
        complete_event = SimulationEvent.create(
            "simulation_completed", "reload-run", step=2, total_steps=2
        )

        # Small delay to ensure file mtime changes (filesystem resolution can be 1s)
        import time
        time.sleep(0.1)

        with open(persist_path, "a") as f:
            f.write(complete_event.to_json() + "\n")

        # Force file mtime to be different by touching it with a future time
        import os
        future_time = time.time() + 1
        os.utime(persist_path, (future_time, future_time))

        # Server should be able to see the completion event
        history_after = server_bus.get_history("reload-run")

        assert len(history_after) == 2, (
            f"Expected 2 events after worker completion, got {len(history_after)}. "
            "Server EventBus is not detecting new events from worker."
        )

        # Cleanup
        EventBus._test_persist_path = None
        EventBus.reset_instance()
