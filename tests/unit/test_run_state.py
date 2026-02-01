"""Unit tests for RunStateManager.

Tests cover:
- RunState dataclass creation and serialization
- State machine transitions
- Optimistic locking
- Heartbeat mechanism
- Stale run detection
- Run lifecycle operations
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest


class TestRunState:
    """Tests for RunState dataclass."""

    def test_create_with_defaults(self):
        """Test creating RunState with minimum required fields."""
        from server.run_state import RunState

        state = RunState(
            run_id="test-run-1",
            status="pending",
            scenario_path="/path/to/scenario.yaml",
            scenario_name="test-scenario",
        )

        assert state.run_id == "test-run-1"
        assert state.status == "pending"
        assert state.scenario_path == "/path/to/scenario.yaml"
        assert state.scenario_name == "test-scenario"
        assert state.current_step == 0
        assert state.total_steps is None
        assert state.version == 1
        assert state.priority == "normal"
        assert state.created_at  # Should be set automatically

    def test_create_with_all_fields(self):
        """Test creating RunState with all fields."""
        from server.run_state import RunState

        state = RunState(
            run_id="full-run",
            status="running",
            scenario_path="/path/scenario.yaml",
            scenario_name="full-scenario",
            current_step=5,
            total_steps=10,
            created_at="2024-01-15T10:00:00",
            started_at="2024-01-15T10:01:00",
            paused_at=None,
            completed_at=None,
            worker_id="worker-1",
            last_heartbeat="2024-01-15T10:05:00",
            priority="high",
            error=None,
            danger_count=2,
            version=3,
        )

        assert state.current_step == 5
        assert state.total_steps == 10
        assert state.worker_id == "worker-1"
        assert state.danger_count == 2
        assert state.version == 3

    def test_to_dict(self):
        """Test to_dict() returns all fields."""
        from server.run_state import RunState

        state = RunState(
            run_id="dict-test",
            status="pending",
            scenario_path="/path.yaml",
            scenario_name="dict-scenario",
            created_at="2024-01-15T10:00:00",
        )

        d = state.to_dict()

        assert d["run_id"] == "dict-test"
        assert d["status"] == "pending"
        assert d["scenario_path"] == "/path.yaml"
        assert d["scenario_name"] == "dict-scenario"
        assert d["current_step"] == 0
        assert d["version"] == 1

    def test_to_api_dict(self):
        """Test to_api_dict() returns frontend-expected format."""
        from server.run_state import RunState

        state = RunState(
            run_id="api-test",
            status="running",
            scenario_path="/path.yaml",
            scenario_name="api-scenario",
            current_step=3,
            total_steps=10,
            started_at="2024-01-15T10:01:00",
            danger_count=1,
            created_at="2024-01-15T10:00:00",
        )

        d = state.to_api_dict()

        assert d["runId"] == "api-test"
        assert d["scenario"] == "api-scenario"
        assert d["status"] == "running"
        assert d["currentStep"] == 3
        assert d["totalSteps"] == 10
        assert d["dangerCount"] == 1
        assert d["startedAt"] == "2024-01-15T10:01:00"

    def test_json_serialization_roundtrip(self):
        """Test to_json() and from_json() are inverses."""
        from server.run_state import RunState

        original = RunState(
            run_id="json-test",
            status="paused",
            scenario_path="/scenarios/test.yaml",
            scenario_name="json-scenario",
            current_step=7,
            total_steps=20,
            created_at="2024-01-15T10:00:00",
            worker_id="worker-abc",
        )

        json_str = original.to_json()
        restored = RunState.from_json(json_str)

        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert restored.current_step == original.current_step
        assert restored.worker_id == original.worker_id

    def test_from_dict(self):
        """Test from_dict() creates RunState correctly."""
        from server.run_state import RunState

        data = {
            "run_id": "from-dict",
            "status": "completed",
            "scenario_path": "/path.yaml",
            "scenario_name": "dict-scenario",
            "current_step": 10,
            "total_steps": 10,
            "created_at": "2024-01-15T10:00:00",
            "started_at": "2024-01-15T10:01:00",
            "paused_at": None,
            "completed_at": "2024-01-15T10:10:00",
            "worker_id": None,
            "last_heartbeat": None,
            "priority": "normal",
            "error": None,
            "danger_count": 0,
            "version": 5,
        }

        state = RunState.from_dict(data)

        assert state.run_id == "from-dict"
        assert state.status == "completed"
        assert state.version == 5


class TestStateTransitions:
    """Tests for state machine transitions."""

    def test_valid_transitions(self):
        """Test all valid state transitions."""
        from server.run_state import VALID_TRANSITIONS

        # pending -> running
        assert "running" in VALID_TRANSITIONS["pending"]
        # pending -> cancelled
        assert "cancelled" in VALID_TRANSITIONS["pending"]

        # running -> paused
        assert "paused" in VALID_TRANSITIONS["running"]
        # running -> completed
        assert "completed" in VALID_TRANSITIONS["running"]
        # running -> failed
        assert "failed" in VALID_TRANSITIONS["running"]
        # running -> interrupted
        assert "interrupted" in VALID_TRANSITIONS["running"]

        # paused -> running
        assert "running" in VALID_TRANSITIONS["paused"]
        # paused -> cancelled
        assert "cancelled" in VALID_TRANSITIONS["paused"]
        # paused -> interrupted
        assert "interrupted" in VALID_TRANSITIONS["paused"]

        # interrupted -> running (resume)
        assert "running" in VALID_TRANSITIONS["interrupted"]

    def test_terminal_states_have_no_transitions(self):
        """Test that terminal states have no valid transitions."""
        from server.run_state import VALID_TRANSITIONS, TERMINAL_STATES

        for state in TERMINAL_STATES:
            assert VALID_TRANSITIONS[state] == [], f"{state} should have no transitions"


class TestRunStateManager:
    """Tests for RunStateManager class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis connection."""
        mock = MagicMock()
        mock.exists.return_value = False
        mock.get.return_value = None
        mock.pipeline.return_value.__enter__ = MagicMock(return_value=mock.pipeline.return_value)
        mock.pipeline.return_value.__exit__ = MagicMock(return_value=None)
        mock.pipeline.return_value.watch = MagicMock()
        mock.pipeline.return_value.unwatch = MagicMock()
        mock.pipeline.return_value.get = MagicMock(return_value=None)
        mock.pipeline.return_value.set = MagicMock()
        mock.pipeline.return_value.multi = MagicMock()
        mock.pipeline.return_value.execute = MagicMock(return_value=[True])
        mock.pipeline.return_value.reset = MagicMock()
        return mock

    def setup_method(self):
        """Reset singleton before each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def test_singleton_pattern(self, mock_redis):
        """Test get_instance() returns singleton."""
        from server.run_state import RunStateManager

        manager1 = RunStateManager.initialize(mock_redis)
        manager2 = RunStateManager.get_instance()

        assert manager1 is manager2

    def test_get_instance_before_initialize_returns_none(self):
        """Test get_instance() returns None before initialization."""
        from server.run_state import RunStateManager

        assert RunStateManager.get_instance() is None

    def test_create_run(self, mock_redis):
        """Test create_run() creates state in Redis."""
        from server.run_state import RunStateManager

        manager = RunStateManager.initialize(mock_redis)
        state = manager.create_run(
            run_id="new-run",
            scenario_path="/path/scenario.yaml",
            scenario_name="new-scenario",
            priority="high",
        )

        assert state.run_id == "new-run"
        assert state.status == "pending"
        assert state.priority == "high"
        assert mock_redis.set.called
        assert mock_redis.zadd.called

    def test_create_run_duplicate_raises(self, mock_redis):
        """Test create_run() raises if run already exists."""
        from server.run_state import RunStateManager

        mock_redis.exists.return_value = True  # Simulate existing run

        manager = RunStateManager.initialize(mock_redis)

        with pytest.raises(ValueError, match="already exists"):
            manager.create_run(
                run_id="existing-run",
                scenario_path="/path.yaml",
                scenario_name="scenario",
            )

    def test_get_state_returns_none_for_missing(self, mock_redis):
        """Test get_state() returns None when run doesn't exist."""
        from server.run_state import RunStateManager

        mock_redis.get.return_value = None

        manager = RunStateManager.initialize(mock_redis)
        state = manager.get_state("nonexistent-run")

        assert state is None

    def test_get_state_returns_run_state(self, mock_redis):
        """Test get_state() returns RunState when found."""
        from server.run_state import RunStateManager, RunState

        existing_state = RunState(
            run_id="existing-run",
            status="running",
            scenario_path="/path.yaml",
            scenario_name="scenario",
            created_at="2024-01-15T10:00:00",
        )
        mock_redis.get.return_value = existing_state.to_json()

        manager = RunStateManager.initialize(mock_redis)
        state = manager.get_state("existing-run")

        assert state is not None
        assert state.run_id == "existing-run"
        assert state.status == "running"

    def test_transition_valid(self, mock_redis):
        """Test transition() with valid state change."""
        from server.run_state import RunStateManager, RunState

        existing_state = RunState(
            run_id="trans-run",
            status="pending",
            scenario_path="/path.yaml",
            scenario_name="scenario",
            created_at="2024-01-15T10:00:00",
        )

        # Configure pipeline mock for transition
        pipe_mock = MagicMock()
        pipe_mock.watch = MagicMock()
        pipe_mock.unwatch = MagicMock()
        pipe_mock.get = MagicMock(return_value=existing_state.to_json())
        pipe_mock.multi = MagicMock()
        pipe_mock.set = MagicMock()
        pipe_mock.execute = MagicMock(return_value=[True])
        pipe_mock.reset = MagicMock()

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)
        new_state = manager.transition("trans-run", "running", worker_id="worker-1")

        assert new_state.status == "running"
        assert new_state.worker_id == "worker-1"
        assert new_state.version == 2  # Version should increment

    def test_transition_invalid_raises(self, mock_redis):
        """Test transition() with invalid state change raises."""
        from server.run_state import RunStateManager, RunState, InvalidTransitionError

        # State in terminal "completed" status
        existing_state = RunState(
            run_id="completed-run",
            status="completed",
            scenario_path="/path.yaml",
            scenario_name="scenario",
            created_at="2024-01-15T10:00:00",
        )

        pipe_mock = MagicMock()
        pipe_mock.watch = MagicMock()
        pipe_mock.unwatch = MagicMock()
        pipe_mock.get = MagicMock(return_value=existing_state.to_json())
        pipe_mock.reset = MagicMock()

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)

        with pytest.raises(InvalidTransitionError):
            manager.transition("completed-run", "running")

    def test_transition_nonexistent_raises(self, mock_redis):
        """Test transition() on missing run raises ValueError."""
        from server.run_state import RunStateManager

        pipe_mock = MagicMock()
        pipe_mock.watch = MagicMock()
        pipe_mock.unwatch = MagicMock()
        pipe_mock.get = MagicMock(return_value=None)
        pipe_mock.reset = MagicMock()

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)

        with pytest.raises(ValueError, match="not found"):
            manager.transition("ghost-run", "running")

    def test_transition_sets_timestamps(self, mock_redis):
        """Test transition() sets appropriate timestamps."""
        from server.run_state import RunStateManager, RunState

        pending_state = RunState(
            run_id="time-run",
            status="pending",
            scenario_path="/path.yaml",
            scenario_name="scenario",
            created_at="2024-01-15T10:00:00",
        )

        pipe_mock = MagicMock()
        pipe_mock.watch = MagicMock()
        pipe_mock.get = MagicMock(return_value=pending_state.to_json())
        pipe_mock.multi = MagicMock()
        pipe_mock.set = MagicMock()
        pipe_mock.execute = MagicMock(return_value=[True])
        pipe_mock.reset = MagicMock()

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)
        new_state = manager.transition("time-run", "running")

        # started_at should be set when transitioning to running
        assert new_state.started_at is not None

    def test_delete_run(self, mock_redis):
        """Test delete_run() removes state from Redis."""
        from server.run_state import RunStateManager

        pipe_mock = MagicMock()
        pipe_mock.delete = MagicMock()
        pipe_mock.zrem = MagicMock()
        pipe_mock.execute = MagicMock(return_value=[1, 0, 0])  # First delete succeeded

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)
        result = manager.delete_run("delete-me")

        assert result is True
        assert pipe_mock.delete.called
        assert pipe_mock.zrem.called


class TestHeartbeat:
    """Tests for heartbeat functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis connection with pipeline support."""
        mock = MagicMock()

        pipe_mock = MagicMock()
        pipe_mock.watch = MagicMock()
        pipe_mock.get = MagicMock(return_value=None)
        pipe_mock.multi = MagicMock()
        pipe_mock.set = MagicMock()
        pipe_mock.execute = MagicMock(return_value=[True])
        pipe_mock.reset = MagicMock()

        mock.pipeline.return_value = pipe_mock
        return mock

    def setup_method(self):
        """Reset singleton before each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def test_heartbeat_sets_ttl_key(self, mock_redis):
        """Test heartbeat() sets key with TTL."""
        from server.run_state import RunStateManager, RunState

        existing_state = RunState(
            run_id="hb-run",
            status="running",
            scenario_path="/path.yaml",
            scenario_name="scenario",
            created_at="2024-01-15T10:00:00",
        )

        pipe_mock = mock_redis.pipeline.return_value
        pipe_mock.get.return_value = existing_state.to_json()

        manager = RunStateManager.initialize(mock_redis)
        result = manager.heartbeat("hb-run", "worker-1", step=5)

        assert result is True
        assert mock_redis.setex.called

    def test_is_heartbeat_stale_true(self, mock_redis):
        """Test is_heartbeat_stale() returns True when key missing."""
        from server.run_state import RunStateManager

        mock_redis.exists.return_value = False

        manager = RunStateManager.initialize(mock_redis)
        result = manager.is_heartbeat_stale("stale-run")

        assert result is True

    def test_is_heartbeat_stale_false(self, mock_redis):
        """Test is_heartbeat_stale() returns False when key exists."""
        from server.run_state import RunStateManager

        mock_redis.exists.return_value = True

        manager = RunStateManager.initialize(mock_redis)
        result = manager.is_heartbeat_stale("alive-run")

        assert result is False


class TestListRuns:
    """Tests for list_runs functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis connection."""
        return MagicMock()

    def setup_method(self):
        """Reset singleton before each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def test_list_runs_empty(self, mock_redis):
        """Test list_runs() returns empty list when no runs."""
        from server.run_state import RunStateManager

        mock_redis.zrevrange.return_value = []

        manager = RunStateManager.initialize(mock_redis)
        runs = manager.list_runs()

        assert runs == []

    def test_list_runs_returns_states(self, mock_redis):
        """Test list_runs() returns RunState objects."""
        from server.run_state import RunStateManager, RunState

        state1 = RunState(
            run_id="run-1",
            status="running",
            scenario_path="/path.yaml",
            scenario_name="scenario-1",
            created_at="2024-01-15T10:00:00",
        )
        state2 = RunState(
            run_id="run-2",
            status="completed",
            scenario_path="/path.yaml",
            scenario_name="scenario-2",
            created_at="2024-01-15T11:00:00",
        )

        mock_redis.zrevrange.return_value = [b"run-1", b"run-2"]

        pipe_mock = MagicMock()
        pipe_mock.get = MagicMock()
        pipe_mock.execute = MagicMock(return_value=[state1.to_json(), state2.to_json()])

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)
        runs = manager.list_runs()

        assert len(runs) == 2
        assert runs[0].run_id == "run-1"
        assert runs[1].run_id == "run-2"

    def test_list_runs_filter_by_status(self, mock_redis):
        """Test list_runs() filters by status."""
        from server.run_state import RunStateManager, RunState

        state1 = RunState(
            run_id="run-1",
            status="running",
            scenario_path="/path.yaml",
            scenario_name="scenario-1",
            created_at="2024-01-15T10:00:00",
        )
        state2 = RunState(
            run_id="run-2",
            status="completed",
            scenario_path="/path.yaml",
            scenario_name="scenario-2",
            created_at="2024-01-15T11:00:00",
        )

        mock_redis.zrevrange.return_value = [b"run-1", b"run-2"]

        pipe_mock = MagicMock()
        pipe_mock.get = MagicMock()
        pipe_mock.execute = MagicMock(return_value=[state1.to_json(), state2.to_json()])

        mock_redis.pipeline.return_value = pipe_mock

        manager = RunStateManager.initialize(mock_redis)
        running_runs = manager.list_runs(status="running")

        assert len(running_runs) == 1
        assert running_runs[0].status == "running"


class TestModuleLevelAccessor:
    """Tests for module-level get_state_manager function."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def test_get_state_manager_returns_none_before_init(self):
        """Test get_state_manager() returns None when not initialized."""
        from server.run_state import get_state_manager

        assert get_state_manager() is None

    def test_get_state_manager_returns_instance(self):
        """Test get_state_manager() returns singleton after init."""
        from server.run_state import get_state_manager, RunStateManager

        mock_redis = MagicMock()
        manager = RunStateManager.initialize(mock_redis)

        assert get_state_manager() is manager
