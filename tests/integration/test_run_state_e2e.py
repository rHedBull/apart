"""
End-to-end tests for RunStateManager integration with API and worker.

Tests the full lifecycle:
- API creates run via POST /api/v1/runs
- RunStateManager tracks state
- Worker transitions (simulated)
- API returns correct status
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from fakeredis import FakeRedis
from fastapi.testclient import TestClient


@pytest.fixture
def fake_redis():
    """Create a shared FakeRedis instance."""
    return FakeRedis(decode_responses=True)


@pytest.fixture
def state_manager(fake_redis):
    """Initialize RunStateManager with FakeRedis."""
    from server.run_state import RunStateManager
    RunStateManager.reset_instance()
    manager = RunStateManager.initialize(fake_redis)
    yield manager
    RunStateManager.reset_instance()


@pytest.fixture
def test_client_with_state_manager(fake_redis, state_manager):
    """Create test client with state manager initialized."""
    import server.job_queue as job_queue_module
    from server.app import app
    from server.event_bus import EventBus

    def mock_init_job_queue(redis_url: str):
        from rq import Queue
        job_queue_module._redis_conn = fake_redis
        job_queue_module._queues = {
            "high": Queue("simulations-high", connection=fake_redis),
            "normal": Queue("simulations", connection=fake_redis),
            "low": Queue("simulations-low", connection=fake_redis),
        }

    # Mock EventBus Redis functions to avoid blocking subscriber
    def mock_init_event_bus_redis():
        event_bus = EventBus.get_instance()
        event_bus.set_redis_connection(fake_redis)

    async def mock_start_event_bus_subscriber():
        pass

    with patch.object(job_queue_module, 'init_job_queue', mock_init_job_queue):
        with patch('server.app._initialize_event_bus_redis', mock_init_event_bus_redis):
            with patch('server.app._start_event_bus_subscriber', mock_start_event_bus_subscriber):
                with TestClient(app) as client:
                    yield client

    job_queue_module._redis_conn = None
    job_queue_module._queues = {}


@pytest.fixture
def sample_scenario_path(tmp_path):
    """Create a sample scenario file."""
    import yaml
    scenario = {
        "name": "E2E Test Scenario",
        "max_steps": 3,
        "agents": [
            {"name": "TestAgent", "llm": {"provider": "mock", "model": "test"}}
        ],
    }
    path = tmp_path / "e2e_scenario.yaml"
    with open(path, "w") as f:
        yaml.dump(scenario, f)
    return path


class TestApiCreatesRunState:
    """Tests verifying API creates RunState entries."""

    def test_state_manager_creates_pending_state(self, state_manager):
        """Verify state manager creates pending state correctly."""
        # Create run directly in state manager (simulating what API does)
        state = state_manager.create_run(
            run_id="api-sim-1",
            scenario_path="/path/to/scenario.yaml",
            scenario_name="Test Scenario",
            priority="normal",
        )

        assert state.status == "pending"
        assert state.scenario_path == "/path/to/scenario.yaml"
        assert state.scenario_name == "Test Scenario"

        # Verify it can be retrieved
        retrieved = state_manager.get_state("api-sim-1")
        assert retrieved is not None
        assert retrieved.status == "pending"

    def test_get_run_returns_state_manager_status(
        self, test_client_with_state_manager, state_manager
    ):
        """GET /api/v1/runs/{id} should return status from RunStateManager."""
        # Create run directly in state manager
        state_manager.create_run(
            run_id="direct-create",
            scenario_path="/path/scenario.yaml",
            scenario_name="Direct Test",
        )

        response = test_client_with_state_manager.get("/api/v1/runs/direct-create")

        assert response.status_code == 200
        data = response.json()
        assert data["runId"] == "direct-create"
        assert data["status"] == "pending"
        assert data["scenario"] == "Direct Test"

    def test_list_runs_returns_state_manager_runs(
        self, test_client_with_state_manager, state_manager
    ):
        """GET /api/v1/runs should list runs from RunStateManager."""
        # Create multiple runs
        state_manager.create_run("list-1", "/p.yaml", "Scenario 1")
        state_manager.create_run("list-2", "/p.yaml", "Scenario 2")
        state_manager.transition("list-2", "running", worker_id="w1")

        response = test_client_with_state_manager.get("/api/v1/runs")

        assert response.status_code == 200
        data = response.json()
        runs = {r["runId"]: r for r in data["runs"]}

        assert "list-1" in runs
        assert "list-2" in runs
        assert runs["list-1"]["status"] == "pending"
        assert runs["list-2"]["status"] == "running"

    def test_get_nonexistent_run_returns_404(self, test_client_with_state_manager):
        """GET /api/v1/runs/{id} returns 404 for unknown run."""
        response = test_client_with_state_manager.get("/api/v1/runs/nonexistent")

        assert response.status_code == 404


class TestStatusTransitionsThroughApi:
    """Tests verifying API reflects state transitions."""

    def test_status_reflects_running_transition(
        self, test_client_with_state_manager, state_manager
    ):
        """API should show 'running' after transition."""
        state_manager.create_run("trans-1", "/p.yaml", "scenario")
        state_manager.transition("trans-1", "running", worker_id="w1")

        response = test_client_with_state_manager.get("/api/v1/runs/trans-1")

        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_status_reflects_completed_transition(
        self, test_client_with_state_manager, state_manager
    ):
        """API should show 'completed' after completion transition."""
        state_manager.create_run("trans-complete", "/p.yaml", "scenario")
        state_manager.transition("trans-complete", "running", worker_id="w1")
        state_manager.transition("trans-complete", "completed")

        response = test_client_with_state_manager.get("/api/v1/runs/trans-complete")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_status_reflects_failed_transition(
        self, test_client_with_state_manager, state_manager
    ):
        """API should show 'failed' status after failure transition."""
        state_manager.create_run("trans-fail", "/p.yaml", "scenario")
        state_manager.transition("trans-fail", "running", worker_id="w1")
        state_manager.transition("trans-fail", "failed", error="Simulation crashed")

        response = test_client_with_state_manager.get("/api/v1/runs/trans-fail")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"

        # Verify error in state manager (API detail view doesn't expose error field)
        state = state_manager.get_state("trans-fail")
        assert state.error == "Simulation crashed"

    def test_status_reflects_interrupted_transition(
        self, test_client_with_state_manager, state_manager
    ):
        """API should show 'interrupted' after worker crash detection."""
        state_manager.create_run("trans-interrupt", "/p.yaml", "scenario")
        state_manager.transition("trans-interrupt", "running", worker_id="w1")
        state_manager.mark_interrupted("trans-interrupt", reason="Heartbeat expired")

        response = test_client_with_state_manager.get("/api/v1/runs/trans-interrupt")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "interrupted"

        # Verify error in state manager
        state = state_manager.get_state("trans-interrupt")
        assert state.error == "Heartbeat expired"


class TestProgressTracking:
    """Tests for progress tracking through API."""

    def test_current_step_updates(
        self, test_client_with_state_manager, state_manager
    ):
        """API should reflect current step from state manager."""
        state_manager.create_run("progress-1", "/p.yaml", "scenario", total_steps=100)
        state_manager.transition("progress-1", "running", worker_id="w1")
        state_manager.update_progress("progress-1", current_step=42, danger_count=5)

        response = test_client_with_state_manager.get("/api/v1/runs/progress-1")

        assert response.status_code == 200
        data = response.json()
        assert data["currentStep"] == 42
        assert data["maxSteps"] == 100

        # Verify danger_count in state manager (API detail view exposes dangerSignals array instead)
        state = state_manager.get_state("progress-1")
        assert state.danger_count == 5


class TestDeleteRuns:
    """Tests for run deletion through API."""

    def test_delete_run_removes_from_state_manager(
        self, test_client_with_state_manager, state_manager
    ):
        """DELETE /api/v1/runs/{id} should remove from state manager."""
        state_manager.create_run("delete-api", "/p.yaml", "scenario")

        response = test_client_with_state_manager.delete("/api/v1/runs/delete-api")

        assert response.status_code == 200

        # Verify gone from state manager
        assert state_manager.get_state("delete-api") is None

        # Verify 404 on GET
        response = test_client_with_state_manager.get("/api/v1/runs/delete-api")
        assert response.status_code == 404


class TestWorkerTaskIntegration:
    """Tests simulating worker task execution."""

    @pytest.fixture
    def worker_state_manager(self, fake_redis):
        """Simulated worker's state manager (same Redis)."""
        from server.run_state import RunStateManager
        # Don't reset - use same instance as API
        return RunStateManager.get_instance()

    def test_worker_task_transitions_to_running(
        self, test_client_with_state_manager, state_manager, worker_state_manager
    ):
        """Simulated worker transitioning run to running."""
        # API creates pending run
        state_manager.create_run("worker-sim-1", "/p.yaml", "scenario")

        # Verify pending through API
        response = test_client_with_state_manager.get("/api/v1/runs/worker-sim-1")
        assert response.json()["status"] == "pending"

        # Worker picks up job and transitions
        worker_state_manager.transition("worker-sim-1", "running", worker_id="w1")

        # API now sees running
        response = test_client_with_state_manager.get("/api/v1/runs/worker-sim-1")
        assert response.json()["status"] == "running"

    def test_worker_heartbeat_visible_through_api(
        self, test_client_with_state_manager, state_manager, worker_state_manager
    ):
        """Worker heartbeats should update state visible through API."""
        state_manager.create_run("hb-api-1", "/p.yaml", "scenario", total_steps=10)
        state_manager.transition("hb-api-1", "running", worker_id="w1")

        # Worker sends heartbeat with step update
        worker_state_manager.heartbeat("hb-api-1", "w1", step=7)

        # API should see updated step
        response = test_client_with_state_manager.get("/api/v1/runs/hb-api-1")
        data = response.json()
        assert data["currentStep"] == 7

        # Verify worker_id in state manager
        state = state_manager.get_state("hb-api-1")
        assert state.worker_id == "w1"

    def test_worker_completion_visible_through_api(
        self, test_client_with_state_manager, state_manager, worker_state_manager
    ):
        """Worker completion should be visible through API."""
        state_manager.create_run("complete-api", "/p.yaml", "scenario")
        state_manager.transition("complete-api", "running", worker_id="w1")

        # Worker completes
        worker_state_manager.transition("complete-api", "completed")

        # API sees completed
        response = test_client_with_state_manager.get("/api/v1/runs/complete-api")
        data = response.json()
        assert data["status"] == "completed"

        # Verify completed_at in state manager
        state = state_manager.get_state("complete-api")
        assert state.completed_at is not None

    def test_worker_failure_visible_through_api(
        self, test_client_with_state_manager, state_manager, worker_state_manager
    ):
        """Worker failure should be visible through API."""
        state_manager.create_run("fail-api", "/p.yaml", "scenario")
        state_manager.transition("fail-api", "running", worker_id="w1")

        # Worker fails
        worker_state_manager.transition("fail-api", "failed", error="OOM killed")

        # API sees failure
        response = test_client_with_state_manager.get("/api/v1/runs/fail-api")
        data = response.json()
        assert data["status"] == "failed"

        # Verify error in state manager
        state = state_manager.get_state("fail-api")
        assert state.error == "OOM killed"


class TestHealthEndpoints:
    """Tests for health check endpoints with state manager."""

    def test_detailed_health_shows_run_count(
        self, test_client_with_state_manager, state_manager
    ):
        """Detailed health should show run count from state manager."""
        # Create some runs
        state_manager.create_run("health-1", "/p.yaml", "s1")
        state_manager.create_run("health-2", "/p.yaml", "s2")

        response = test_client_with_state_manager.get("/api/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["total_run_ids"] == 2


class TestListRunsFiltering:
    """Tests for /api/v1/runs filtering and sorting."""

    def test_runs_sorted_by_started_at_descending(
        self, test_client_with_state_manager, state_manager
    ):
        """Runs should be sorted by startedAt (most recent first)."""
        import time

        # Create runs with small delays to ensure different timestamps
        state_manager.create_run("sort-1", "/p.yaml", "First")
        state_manager.transition("sort-1", "running", worker_id="w1")
        time.sleep(0.01)

        state_manager.create_run("sort-2", "/p.yaml", "Second")
        state_manager.transition("sort-2", "running", worker_id="w1")
        time.sleep(0.01)

        state_manager.create_run("sort-3", "/p.yaml", "Third")
        state_manager.transition("sort-3", "running", worker_id="w1")

        response = test_client_with_state_manager.get("/api/v1/runs")

        assert response.status_code == 200
        runs = response.json()["runs"]

        # Most recent first
        run_ids = [r["runId"] for r in runs]
        assert run_ids.index("sort-3") < run_ids.index("sort-2")
        assert run_ids.index("sort-2") < run_ids.index("sort-1")


class TestRunStateManagerNotInitialized:
    """Tests for graceful handling when state manager is not initialized."""

    def test_get_state_manager_returns_none_before_init(self):
        """get_state_manager() returns None before initialization."""
        from server.run_state import RunStateManager, get_state_manager

        RunStateManager.reset_instance()

        assert get_state_manager() is None

        RunStateManager.reset_instance()

    def test_routes_raise_when_state_manager_not_initialized(self):
        """API routes should raise RuntimeError when state manager missing."""
        from server.run_state import RunStateManager
        from server.routes.v1 import _get_run_status

        RunStateManager.reset_instance()

        with pytest.raises(RuntimeError, match="not initialized"):
            _get_run_status("any-run")

        RunStateManager.reset_instance()
