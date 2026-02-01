"""
Integration tests for RunStateManager with real Redis (FakeRedis).

Tests cover:
- Full state lifecycle with actual Redis operations
- Concurrent access and optimistic locking
- Heartbeat TTL expiration
- Stale run detection
- Cross-process simulation (API/worker sharing Redis)
"""

import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from fakeredis import FakeRedis


class TestRunStateIntegration:
    """Integration tests with real Redis operations."""

    @pytest.fixture
    def redis_conn(self):
        """Create a FakeRedis connection for testing."""
        return FakeRedis(decode_responses=True)

    @pytest.fixture
    def state_manager(self, redis_conn):
        """Create a RunStateManager with FakeRedis."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(redis_conn)
        yield manager
        RunStateManager.reset_instance()

    def test_full_lifecycle_pending_to_completed(self, state_manager):
        """Test complete run lifecycle: pending → running → completed."""
        # Create run
        state = state_manager.create_run(
            run_id="lifecycle-1",
            scenario_path="/path/scenario.yaml",
            scenario_name="lifecycle-test",
        )
        assert state.status == "pending"
        assert state.version == 1

        # Transition to running
        state = state_manager.transition("lifecycle-1", "running", worker_id="worker-1")
        assert state.status == "running"
        assert state.worker_id == "worker-1"
        assert state.started_at is not None
        assert state.version == 2

        # Update progress
        state = state_manager.update_progress("lifecycle-1", current_step=5, danger_count=2)
        assert state.current_step == 5
        assert state.danger_count == 2

        # Transition to completed
        state = state_manager.transition("lifecycle-1", "completed")
        assert state.status == "completed"
        assert state.completed_at is not None

        # Verify terminal state - cannot transition further
        from server.run_state import InvalidTransitionError
        with pytest.raises(InvalidTransitionError):
            state_manager.transition("lifecycle-1", "running")

    def test_full_lifecycle_pending_to_failed(self, state_manager):
        """Test run lifecycle with failure: pending → running → failed."""
        state = state_manager.create_run(
            run_id="fail-lifecycle",
            scenario_path="/path/scenario.yaml",
            scenario_name="fail-test",
        )

        state_manager.transition("fail-lifecycle", "running", worker_id="worker-2")
        state = state_manager.transition(
            "fail-lifecycle",
            "failed",
            error="Simulation crashed with OOM"
        )

        assert state.status == "failed"
        assert state.error == "Simulation crashed with OOM"
        assert state.completed_at is not None

    def test_pause_and_resume(self, state_manager):
        """Test pause/resume cycle: running → paused → running."""
        state_manager.create_run(
            run_id="pause-test",
            scenario_path="/path.yaml",
            scenario_name="pause-scenario",
        )
        state_manager.transition("pause-test", "running", worker_id="w1")

        # Pause
        state = state_manager.transition("pause-test", "paused")
        assert state.status == "paused"
        assert state.paused_at is not None

        # Resume
        state = state_manager.transition("pause-test", "running", worker_id="w2")
        assert state.status == "running"
        assert state.worker_id == "w2"
        assert state.paused_at is None  # Cleared on resume

    def test_interrupted_and_resume(self, state_manager):
        """Test interrupted state and resumption."""
        state_manager.create_run(
            run_id="interrupt-test",
            scenario_path="/path.yaml",
            scenario_name="interrupt-scenario",
        )
        state_manager.transition("interrupt-test", "running", worker_id="w1")

        # Mark as interrupted (e.g., worker crashed)
        state = state_manager.transition(
            "interrupt-test",
            "interrupted",
            error="Worker heartbeat expired"
        )
        assert state.status == "interrupted"
        assert state.error == "Worker heartbeat expired"

        # Resume with new worker
        state = state_manager.transition("interrupt-test", "running", worker_id="w2")
        assert state.status == "running"
        assert state.worker_id == "w2"

    def test_list_runs_pagination(self, state_manager):
        """Test list_runs with pagination."""
        # Create multiple runs
        for i in range(15):
            state_manager.create_run(
                run_id=f"paginate-{i:02d}",
                scenario_path="/path.yaml",
                scenario_name=f"scenario-{i}",
            )

        # Get first page
        page1 = state_manager.list_runs(limit=5, offset=0)
        assert len(page1) == 5

        # Get second page
        page2 = state_manager.list_runs(limit=5, offset=5)
        assert len(page2) == 5

        # Ensure no overlap
        page1_ids = {s.run_id for s in page1}
        page2_ids = {s.run_id for s in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_list_runs_filter_by_status(self, state_manager):
        """Test filtering runs by status."""
        # Create runs with different statuses
        state_manager.create_run("filter-pending", "/p.yaml", "s1")

        state_manager.create_run("filter-running", "/p.yaml", "s2")
        state_manager.transition("filter-running", "running", worker_id="w1")

        state_manager.create_run("filter-completed", "/p.yaml", "s3")
        state_manager.transition("filter-completed", "running", worker_id="w1")
        state_manager.transition("filter-completed", "completed")

        # Filter by status
        pending = state_manager.list_runs(status="pending")
        running = state_manager.list_runs(status="running")
        completed = state_manager.list_runs(status="completed")

        assert len(pending) == 1
        assert pending[0].run_id == "filter-pending"

        assert len(running) == 1
        assert running[0].run_id == "filter-running"

        assert len(completed) == 1
        assert completed[0].run_id == "filter-completed"

    def test_delete_run(self, state_manager):
        """Test deleting a run removes all associated data."""
        state_manager.create_run("delete-me", "/p.yaml", "delete-scenario")

        # Verify it exists
        assert state_manager.get_state("delete-me") is not None

        # Delete
        deleted = state_manager.delete_run("delete-me")
        assert deleted is True

        # Verify it's gone
        assert state_manager.get_state("delete-me") is None

        # Verify not in list
        runs = state_manager.list_runs()
        assert all(r.run_id != "delete-me" for r in runs)

    def test_duplicate_run_id_raises(self, state_manager):
        """Test creating a run with existing ID raises error."""
        state_manager.create_run("dup-test", "/p.yaml", "scenario")

        with pytest.raises(ValueError, match="already exists"):
            state_manager.create_run("dup-test", "/p.yaml", "scenario2")


class TestHeartbeatIntegration:
    """Integration tests for heartbeat functionality."""

    @pytest.fixture
    def redis_conn(self):
        """Create FakeRedis for testing."""
        return FakeRedis(decode_responses=True)

    @pytest.fixture
    def state_manager(self, redis_conn):
        """Create RunStateManager."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(redis_conn)
        yield manager
        RunStateManager.reset_instance()

    def test_heartbeat_updates_state(self, state_manager):
        """Test heartbeat updates last_heartbeat and current_step."""
        state_manager.create_run("hb-test", "/p.yaml", "hb-scenario")
        state_manager.transition("hb-test", "running", worker_id="w1")

        # Send heartbeat
        result = state_manager.heartbeat("hb-test", "w1", step=5)
        assert result is True

        # Verify state updated
        state = state_manager.get_state("hb-test")
        assert state.last_heartbeat is not None
        assert state.current_step == 5

    def test_heartbeat_with_different_worker_updates_worker_id(self, state_manager):
        """Test heartbeat from different worker updates worker_id."""
        state_manager.create_run("hb-worker", "/p.yaml", "scenario")
        state_manager.transition("hb-worker", "running", worker_id="w1")

        # Heartbeat from different worker
        state_manager.heartbeat("hb-worker", "w2", step=3)

        state = state_manager.get_state("hb-worker")
        assert state.worker_id == "w2"

    def test_is_heartbeat_stale_without_heartbeat(self, state_manager, redis_conn):
        """Test is_heartbeat_stale returns True when no heartbeat exists."""
        state_manager.create_run("no-hb", "/p.yaml", "scenario")
        state_manager.transition("no-hb", "running", worker_id="w1")

        # No heartbeat sent yet - should be stale
        assert state_manager.is_heartbeat_stale("no-hb") is True

    def test_is_heartbeat_stale_with_recent_heartbeat(self, state_manager):
        """Test is_heartbeat_stale returns False after recent heartbeat."""
        state_manager.create_run("recent-hb", "/p.yaml", "scenario")
        state_manager.transition("recent-hb", "running", worker_id="w1")

        # Send heartbeat
        state_manager.heartbeat("recent-hb", "w1", step=1)

        # Should not be stale
        assert state_manager.is_heartbeat_stale("recent-hb") is False


class TestStaleRunDetection:
    """Integration tests for stale run detection."""

    @pytest.fixture
    def redis_conn(self):
        """Create FakeRedis."""
        return FakeRedis(decode_responses=True)

    @pytest.fixture
    def state_manager(self, redis_conn):
        """Create RunStateManager."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(redis_conn)
        yield manager
        RunStateManager.reset_instance()

    def test_check_stale_runs_finds_runs_without_heartbeat(self, state_manager):
        """Test check_stale_runs finds running runs without heartbeat."""
        # Create a running run without heartbeat
        state_manager.create_run("stale-1", "/p.yaml", "scenario")
        state_manager.transition("stale-1", "running", worker_id="w1")

        # Create a running run WITH heartbeat
        state_manager.create_run("fresh-1", "/p.yaml", "scenario")
        state_manager.transition("fresh-1", "running", worker_id="w2")
        state_manager.heartbeat("fresh-1", "w2", step=1)

        # Check stale runs
        stale = state_manager.check_stale_runs()

        assert "stale-1" in stale
        assert "fresh-1" not in stale

    def test_check_stale_runs_ignores_non_running(self, state_manager):
        """Test check_stale_runs ignores non-running runs."""
        # Pending run (no heartbeat expected)
        state_manager.create_run("pending-1", "/p.yaml", "scenario")

        # Completed run (no heartbeat expected)
        state_manager.create_run("completed-1", "/p.yaml", "scenario")
        state_manager.transition("completed-1", "running", worker_id="w1")
        state_manager.transition("completed-1", "completed")

        stale = state_manager.check_stale_runs()

        assert "pending-1" not in stale
        assert "completed-1" not in stale

    def test_mark_interrupted(self, state_manager):
        """Test mark_interrupted transitions run to interrupted status."""
        state_manager.create_run("to-interrupt", "/p.yaml", "scenario")
        state_manager.transition("to-interrupt", "running", worker_id="w1")

        state = state_manager.mark_interrupted(
            "to-interrupt",
            reason="Worker died"
        )

        assert state is not None
        assert state.status == "interrupted"
        assert state.error == "Worker died"

    def test_mark_interrupted_returns_none_for_invalid_transition(self, state_manager):
        """Test mark_interrupted returns None for invalid transitions."""
        # Pending run can't be interrupted
        state_manager.create_run("pending-interrupt", "/p.yaml", "scenario")

        state = state_manager.mark_interrupted("pending-interrupt", "Test")

        assert state is None

        # Verify still pending
        current = state_manager.get_state("pending-interrupt")
        assert current.status == "pending"


class TestConcurrentAccess:
    """Tests for concurrent access and optimistic locking."""

    @pytest.fixture
    def redis_conn(self):
        """Create FakeRedis."""
        return FakeRedis(decode_responses=True)

    @pytest.fixture
    def state_manager(self, redis_conn):
        """Create RunStateManager."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(redis_conn)
        yield manager
        RunStateManager.reset_instance()

    def test_concurrent_progress_updates(self, state_manager):
        """Test concurrent progress updates are handled safely."""
        state_manager.create_run("concurrent-1", "/p.yaml", "scenario")
        state_manager.transition("concurrent-1", "running", worker_id="w1")

        results = []
        errors = []

        def update_progress(step):
            try:
                state = state_manager.update_progress("concurrent-1", current_step=step)
                results.append(state.current_step)
            except Exception as e:
                errors.append(e)

        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_progress, i) for i in range(10)]
            for future in as_completed(futures):
                pass  # Wait for all

        # Should have completed without errors (may have retries internally)
        assert len(errors) == 0

        # Final state should have some step value
        final = state_manager.get_state("concurrent-1")
        assert final.current_step in range(10)

    def test_concurrent_heartbeats(self, state_manager):
        """Test concurrent heartbeats from multiple 'workers'."""
        state_manager.create_run("hb-concurrent", "/p.yaml", "scenario")
        state_manager.transition("hb-concurrent", "running", worker_id="w1")

        success_count = [0]  # Use list to allow mutation in nested function
        lock = threading.Lock()

        def send_heartbeat(worker_id, step):
            result = state_manager.heartbeat("hb-concurrent", worker_id, step=step)
            if result:
                with lock:
                    success_count[0] += 1

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(send_heartbeat, f"w{i}", i)
                for i in range(10)
            ]
            for future in as_completed(futures):
                pass

        # Most heartbeats should succeed (some might fail due to contention)
        assert success_count[0] >= 5


class TestCrossProcessScenario:
    """Tests simulating API server and worker sharing Redis."""

    @pytest.fixture
    def shared_redis(self):
        """Create shared Redis instance."""
        return FakeRedis(decode_responses=True)

    def test_api_creates_worker_transitions(self, shared_redis):
        """Simulate API creating run, worker transitioning it."""
        from server.run_state import RunStateManager

        # "API server" creates run
        RunStateManager.reset_instance()
        api_manager = RunStateManager.initialize(shared_redis)

        api_manager.create_run("cross-1", "/p.yaml", "scenario")

        # Verify pending
        state = api_manager.get_state("cross-1")
        assert state.status == "pending"

        # "Worker" (same Redis, simulating different process)
        # Worker transitions to running
        api_manager.transition("cross-1", "running", worker_id="worker-process")

        # "API" can see the updated status
        state = api_manager.get_state("cross-1")
        assert state.status == "running"
        assert state.worker_id == "worker-process"

        # Worker completes
        api_manager.transition("cross-1", "completed")

        # API sees completed
        state = api_manager.get_state("cross-1")
        assert state.status == "completed"

        RunStateManager.reset_instance()

    def test_worker_heartbeat_visible_to_api(self, shared_redis):
        """Test that worker heartbeats are visible to API queries."""
        from server.run_state import RunStateManager

        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(shared_redis)

        manager.create_run("heartbeat-visible", "/p.yaml", "scenario")
        manager.transition("heartbeat-visible", "running", worker_id="w1")

        # Worker sends heartbeat
        manager.heartbeat("heartbeat-visible", "w1", step=42)

        # API can see heartbeat data
        state = manager.get_state("heartbeat-visible")
        assert state.current_step == 42
        assert state.last_heartbeat is not None

        # API can check if heartbeat is stale
        assert manager.is_heartbeat_stale("heartbeat-visible") is False

        RunStateManager.reset_instance()

    def test_stale_detection_works_across_simulated_processes(self, shared_redis):
        """Test stale detection when worker dies (no more heartbeats)."""
        from server.run_state import RunStateManager

        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(shared_redis)

        manager.create_run("stale-cross", "/p.yaml", "scenario")
        manager.transition("stale-cross", "running", worker_id="dying-worker")

        # Worker sends one heartbeat then "dies" (no more heartbeats)
        manager.heartbeat("stale-cross", "dying-worker", step=1)

        # At this point heartbeat exists, so not stale
        assert manager.is_heartbeat_stale("stale-cross") is False

        # Delete the heartbeat key to simulate TTL expiration
        # (FakeRedis doesn't auto-expire, so we manually delete)
        from server.run_state import KEY_PREFIX, HEARTBEAT_SUFFIX
        heartbeat_key = f"{KEY_PREFIX}stale-cross{HEARTBEAT_SUFFIX}"
        shared_redis.delete(heartbeat_key)

        # Now should be stale
        assert manager.is_heartbeat_stale("stale-cross") is True

        # check_stale_runs should find it
        stale_runs = manager.check_stale_runs()
        assert "stale-cross" in stale_runs

        RunStateManager.reset_instance()


class TestVersionIncrement:
    """Tests for optimistic locking version tracking."""

    @pytest.fixture
    def redis_conn(self):
        """Create FakeRedis."""
        return FakeRedis(decode_responses=True)

    @pytest.fixture
    def state_manager(self, redis_conn):
        """Create RunStateManager."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(redis_conn)
        yield manager
        RunStateManager.reset_instance()

    def test_version_increments_on_transition(self, state_manager):
        """Test version increments on each transition."""
        state = state_manager.create_run("version-test", "/p.yaml", "scenario")
        assert state.version == 1

        state = state_manager.transition("version-test", "running", worker_id="w1")
        assert state.version == 2

        state = state_manager.transition("version-test", "completed")
        assert state.version == 3

    def test_version_increments_on_progress_update(self, state_manager):
        """Test version increments on progress update."""
        state = state_manager.create_run("progress-version", "/p.yaml", "scenario")
        state_manager.transition("progress-version", "running", worker_id="w1")

        initial_version = state_manager.get_state("progress-version").version

        state_manager.update_progress("progress-version", current_step=5)
        new_version = state_manager.get_state("progress-version").version

        assert new_version == initial_version + 1


class TestApiDictFormat:
    """Tests for to_api_dict format used by API endpoints."""

    @pytest.fixture
    def redis_conn(self):
        """Create FakeRedis."""
        return FakeRedis(decode_responses=True)

    @pytest.fixture
    def state_manager(self, redis_conn):
        """Create RunStateManager."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()
        manager = RunStateManager.initialize(redis_conn)
        yield manager
        RunStateManager.reset_instance()

    def test_api_dict_has_correct_format(self, state_manager):
        """Test to_api_dict returns frontend-expected keys."""
        state_manager.create_run(
            run_id="api-format-test",
            scenario_path="/path/to/scenario.yaml",
            scenario_name="Test Scenario",
            priority="high",
            total_steps=100,
        )
        state_manager.transition("api-format-test", "running", worker_id="w1")
        state_manager.update_progress("api-format-test", current_step=50, danger_count=3)

        state = state_manager.get_state("api-format-test")
        api_dict = state.to_api_dict()

        # Check required keys (camelCase for frontend)
        assert "runId" in api_dict
        assert "scenario" in api_dict
        assert "status" in api_dict
        assert "currentStep" in api_dict
        assert "totalSteps" in api_dict
        assert "startedAt" in api_dict
        assert "dangerCount" in api_dict
        assert "workerId" in api_dict
        assert "priority" in api_dict

        # Check values
        assert api_dict["runId"] == "api-format-test"
        assert api_dict["scenario"] == "Test Scenario"
        assert api_dict["status"] == "running"
        assert api_dict["currentStep"] == 50
        assert api_dict["totalSteps"] == 100
        assert api_dict["dangerCount"] == 3
        assert api_dict["workerId"] == "w1"
        assert api_dict["priority"] == "high"
