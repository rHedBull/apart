"""Unit tests for simulations routes, focusing on complete_simulation."""

import pytest
from unittest.mock import patch, MagicMock


class TestCompleteSimulation:
    """Tests for complete_simulation function."""

    def test_complete_simulation_emits_completed_event(self):
        """Test that complete_simulation emits simulation_completed to EventBus."""
        from server.routes.simulations import (
            complete_simulation,
            register_simulation,
            SimulationStatus,
            _simulations
        )

        run_id = "test-complete-event"

        # Register simulation first
        register_simulation(run_id, "test_scenario")

        with patch('server.event_bus.emit_event') as mock_emit:
            complete_simulation(run_id, SimulationStatus.COMPLETED)

            mock_emit.assert_called_once_with("simulation_completed", run_id)

        # Cleanup
        _simulations.pop(run_id, None)

    def test_complete_simulation_emits_failed_event_with_error(self):
        """Test that complete_simulation emits simulation_failed with error message."""
        from server.routes.simulations import (
            complete_simulation,
            register_simulation,
            SimulationStatus,
            _simulations
        )

        run_id = "test-failed-event"
        error_msg = "Engine crashed unexpectedly"

        register_simulation(run_id, "test_scenario")

        with patch('server.event_bus.emit_event') as mock_emit:
            complete_simulation(run_id, SimulationStatus.FAILED, error=error_msg)

            mock_emit.assert_called_once_with(
                "simulation_failed",
                run_id,
                error=error_msg
            )

        # Cleanup
        _simulations.pop(run_id, None)

    def test_complete_simulation_updates_in_memory_state(self):
        """Test that complete_simulation updates the in-memory simulation dict."""
        from server.routes.simulations import (
            complete_simulation,
            register_simulation,
            SimulationStatus,
            _simulations
        )

        run_id = "test-state-update"

        register_simulation(run_id, "test_scenario")

        with patch('server.event_bus.emit_event'):
            complete_simulation(run_id, SimulationStatus.COMPLETED)

        assert _simulations[run_id]["status"] == SimulationStatus.COMPLETED
        assert "completed_at" in _simulations[run_id]

        # Cleanup
        _simulations.pop(run_id, None)

    def test_complete_simulation_stores_error_message(self):
        """Test that error message is stored in simulation state."""
        from server.routes.simulations import (
            complete_simulation,
            register_simulation,
            SimulationStatus,
            _simulations
        )

        run_id = "test-error-storage"
        error_msg = "LLM timeout after 30s"

        register_simulation(run_id, "test_scenario")

        with patch('server.event_bus.emit_event'):
            complete_simulation(run_id, SimulationStatus.FAILED, error=error_msg)

        assert _simulations[run_id]["error_message"] == error_msg

        # Cleanup
        _simulations.pop(run_id, None)

    def test_complete_simulation_handles_unknown_run_id(self):
        """Test that complete_simulation doesn't crash on unknown run_id."""
        from server.routes.simulations import complete_simulation, SimulationStatus

        with patch('server.event_bus.emit_event') as mock_emit:
            # Should not raise, even for unknown run_id
            complete_simulation("nonexistent-run", SimulationStatus.FAILED, error="test")

            # Event should still be emitted
            mock_emit.assert_called_once()
