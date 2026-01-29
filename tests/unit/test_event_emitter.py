"""Unit tests for event_emitter module."""

import pytest
from unittest.mock import patch, MagicMock
from core.event_emitter import (
    enable_event_emitter,
    disable_event_emitter,
    emit,
    EventTypes,
    _emitter_enabled,
    _run_id
)
import core.event_emitter as event_emitter_module


@pytest.fixture(autouse=True)
def reset_event_emitter():
    """Reset event emitter state before and after each test."""
    # Reset before test
    disable_event_emitter()
    yield
    # Reset after test
    disable_event_emitter()


class TestEnableDisableEventEmitter:
    """Tests for enable_event_emitter and disable_event_emitter."""

    def test_enable_sets_run_id(self):
        """Test that enable sets the run ID."""
        enable_event_emitter("test-run-123")

        assert event_emitter_module._emitter_enabled is True
        assert event_emitter_module._run_id == "test-run-123"

    def test_disable_clears_state(self):
        """Test that disable clears all state."""
        enable_event_emitter("test-run")
        disable_event_emitter()

        assert event_emitter_module._emitter_enabled is False
        assert event_emitter_module._run_id is None

    def test_enable_can_change_run_id(self):
        """Test that enable can change the run ID."""
        enable_event_emitter("run-1")
        enable_event_emitter("run-2")

        assert event_emitter_module._run_id == "run-2"

    def test_initial_state_is_disabled(self):
        """Test that initial state is disabled."""
        # After reset fixture runs, should be disabled
        assert event_emitter_module._emitter_enabled is False
        assert event_emitter_module._run_id is None


class TestEmit:
    """Tests for emit function."""

    def test_emit_does_nothing_when_disabled(self):
        """Test that emit is a no-op when disabled."""
        # Should not raise any errors
        emit("test_event", step=1, data="test")

        # Nothing to assert - just verifying no exception

    def test_emit_does_nothing_when_no_run_id(self):
        """Test that emit is a no-op when run_id is None."""
        event_emitter_module._emitter_enabled = True
        event_emitter_module._run_id = None

        # Should not raise any errors
        emit("test_event", step=1)

    def test_emit_calls_server_when_enabled(self):
        """Test that emit calls server.event_bus.emit_event when enabled."""
        enable_event_emitter("test-run")

        # The emit function imports server.event_bus inside the function
        # We need to patch at that level
        mock_emit_event = MagicMock()
        mock_module = MagicMock()
        mock_module.emit_event = mock_emit_event

        with patch.dict('sys.modules', {'server.event_bus': mock_module}):
            emit("test_event", step=5, custom_data="value")
            mock_emit_event.assert_called_once_with(
                "test_event",
                "test-run",
                5,
                custom_data="value"
            )

    def test_emit_with_step(self):
        """Test emit with step parameter."""
        enable_event_emitter("test-run")

        with patch('server.event_bus.emit_event') as mock_emit:
            emit("step_completed", step=3)

            mock_emit.assert_called_once_with(
                "step_completed",
                "test-run",
                3
            )

    def test_emit_with_data(self):
        """Test emit with additional data."""
        enable_event_emitter("test-run")

        with patch('server.event_bus.emit_event') as mock_emit:
            emit("agent_response", step=1, agent_name="Agent1", response="test")

            mock_emit.assert_called_once_with(
                "agent_response",
                "test-run",
                1,
                agent_name="Agent1",
                response="test"
            )

    def test_emit_without_step(self):
        """Test emit without step parameter."""
        enable_event_emitter("test-run")

        with patch('server.event_bus.emit_event') as mock_emit:
            emit("simulation_started")

            mock_emit.assert_called_once_with(
                "simulation_started",
                "test-run",
                None
            )

    def test_emit_handles_import_error(self):
        """Test that emit handles ImportError gracefully."""
        enable_event_emitter("test-run")

        # Patch to raise ImportError
        with patch.dict('sys.modules', {'server': None, 'server.event_bus': None}):
            # Should not raise
            emit("test_event")

    def test_emit_handles_general_exception(self):
        """Test that emit handles general exceptions gracefully."""
        enable_event_emitter("test-run")

        with patch('server.event_bus.emit_event', side_effect=RuntimeError("Server error")):
            # Should not raise - errors are logged and suppressed
            emit("test_event")


class TestEventTypes:
    """Tests for EventTypes constants."""

    def test_simulation_lifecycle_events(self):
        """Test simulation lifecycle event types exist."""
        assert EventTypes.SIMULATION_STARTED == "simulation_started"
        assert EventTypes.SIMULATION_COMPLETED == "simulation_completed"
        assert EventTypes.SIMULATION_FAILED == "simulation_failed"

    def test_step_events(self):
        """Test step event types exist."""
        assert EventTypes.STEP_STARTED == "step_started"
        assert EventTypes.STEP_COMPLETED == "step_completed"

    def test_agent_events(self):
        """Test agent event types exist."""
        assert EventTypes.AGENT_MESSAGE_SENT == "agent_message_sent"
        assert EventTypes.AGENT_RESPONSE_RECEIVED == "agent_response_received"

    def test_danger_events(self):
        """Test danger detection event types exist."""
        assert EventTypes.DANGER_SIGNAL == "danger_signal"

    def test_all_event_types_are_strings(self):
        """Test that all event types are strings."""
        event_types = [
            EventTypes.SIMULATION_STARTED,
            EventTypes.SIMULATION_COMPLETED,
            EventTypes.SIMULATION_FAILED,
            EventTypes.STEP_STARTED,
            EventTypes.STEP_COMPLETED,
            EventTypes.AGENT_MESSAGE_SENT,
            EventTypes.AGENT_RESPONSE_RECEIVED,
            EventTypes.DANGER_SIGNAL,
        ]

        for event_type in event_types:
            assert isinstance(event_type, str)

    def test_event_types_are_unique(self):
        """Test that all event types are unique."""
        event_types = [
            EventTypes.SIMULATION_STARTED,
            EventTypes.SIMULATION_COMPLETED,
            EventTypes.SIMULATION_FAILED,
            EventTypes.STEP_STARTED,
            EventTypes.STEP_COMPLETED,
            EventTypes.AGENT_MESSAGE_SENT,
            EventTypes.AGENT_RESPONSE_RECEIVED,
            EventTypes.DANGER_SIGNAL,
        ]

        assert len(event_types) == len(set(event_types))


class TestEmitIntegration:
    """Integration-style tests for emit with actual server module."""

    def test_emit_integration_when_server_available(self):
        """Test emit works when server module is available."""
        enable_event_emitter("integration-test")

        # This test will pass if server module is available
        # and emit doesn't raise an exception
        try:
            emit("test_event", step=1, test_key="test_value")
        except Exception as e:
            # Should not raise - even if server fails, emit catches it
            pytest.fail(f"emit raised unexpected exception: {e}")
