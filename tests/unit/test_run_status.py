"""Unit tests for run status determination using RunStateManager.

The old EventBus/RQ-based status tracking has been replaced with
RunStateManager as the single source of truth.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestGetRunStatus:
    """Tests for _get_run_status using RunStateManager."""

    def setup_method(self):
        """Reset state manager before each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.run_state import RunStateManager
        RunStateManager.reset_instance()

    def test_raises_when_state_manager_not_initialized(self):
        """Test that RuntimeError is raised when state manager not initialized."""
        from server.routes.v1 import _get_run_status

        with pytest.raises(RuntimeError, match="RunStateManager not initialized"):
            _get_run_status("any_run")

    def test_returns_status_from_state_manager(self):
        """Test that status is retrieved from state manager."""
        from server.run_state import RunStateManager, RunState
        from server.routes.v1 import _get_run_status

        mock_redis = MagicMock()
        RunStateManager.initialize(mock_redis)

        # Mock a running state
        running_state = RunState(
            run_id="test_run",
            status="running",
            scenario_path="/path.yaml",
            scenario_name="test",
            created_at="2024-01-15T10:00:00",
        )
        mock_redis.get.return_value = running_state.to_json()

        status = _get_run_status("test_run")
        assert status == "running"

    def test_returns_none_for_nonexistent_run(self):
        """Test that None is returned when run doesn't exist."""
        from server.run_state import RunStateManager
        from server.routes.v1 import _get_run_status

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        RunStateManager.initialize(mock_redis)

        status = _get_run_status("nonexistent")
        assert status is None

    def test_returns_all_status_types(self):
        """Test that all status types are correctly returned."""
        from server.run_state import RunStateManager, RunState
        from server.routes.v1 import _get_run_status

        mock_redis = MagicMock()
        RunStateManager.initialize(mock_redis)

        for expected_status in ["pending", "running", "paused", "completed", "failed", "interrupted", "cancelled"]:
            state = RunState(
                run_id=f"test_{expected_status}",
                status=expected_status,
                scenario_path="/path.yaml",
                scenario_name="test",
                created_at="2024-01-15T10:00:00",
            )
            mock_redis.get.return_value = state.to_json()

            actual_status = _get_run_status(f"test_{expected_status}")
            assert actual_status == expected_status, f"Expected {expected_status}, got {actual_status}"
