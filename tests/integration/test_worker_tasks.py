"""
Tests for worker_tasks.py - the actual simulation task execution.

Tests the run_simulation_task function with mocked dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestRunSimulationTask:
    """Tests for the run_simulation_task function."""

    def test_successful_simulation_returns_result(self, tmp_path):
        """run_simulation_task returns success result when simulation completes."""
        from server.worker_tasks import run_simulation_task

        # Create a dummy scenario file
        scenario_path = tmp_path / "test_scenario.yaml"
        scenario_path.write_text("name: test\nmax_steps: 1\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter") as mock_enable:

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = run_simulation_task("run-123", str(scenario_path))

            assert result["run_id"] == "run-123"
            assert result["status"] == "completed"
            assert result["scenario"] == "test_scenario"

    def test_enables_event_emitter_with_run_id(self, tmp_path):
        """run_simulation_task enables event emitter for the run."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "scenario.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter") as mock_enable:

            mock_orch_class.return_value = MagicMock()

            run_simulation_task("my-run-id", str(scenario_path))

            mock_enable.assert_called_once_with("my-run-id")

    def test_creates_orchestrator_with_correct_params(self, tmp_path):
        """run_simulation_task creates Orchestrator with correct parameters."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "my_scenario.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            run_simulation_task("run-456", str(scenario_path))

            mock_orch_class.assert_called_once_with(
                str(scenario_path),
                "my_scenario",
                save_frequency=1,
                run_id="run-456",
            )
            mock_orchestrator.run.assert_called_once()

    def test_failed_simulation_emits_event_and_raises(self, tmp_path):
        """run_simulation_task emits failure event and re-raises on error."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "failing_scenario.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"), \
             patch("server.event_bus.emit_event") as mock_emit:

            mock_orchestrator = MagicMock()
            mock_orchestrator.run.side_effect = RuntimeError("Simulation exploded")
            mock_orch_class.return_value = mock_orchestrator

            with pytest.raises(RuntimeError, match="Simulation exploded"):
                run_simulation_task("failing-run", str(scenario_path))

            mock_emit.assert_called_once_with(
                "simulation_failed",
                "failing-run",
                error="Simulation exploded"
            )

    def test_handles_path_object_conversion(self, tmp_path):
        """run_simulation_task handles string path conversion to Path."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "path_test.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orch_class.return_value = MagicMock()

            # Pass as string (as RQ would serialize it)
            result = run_simulation_task("run-789", str(scenario_path))

            assert result["scenario"] == "path_test"


class TestWorkerTaskErrorHandling:
    """Tests for error handling in worker tasks."""

    def test_orchestrator_init_error_propagates(self, tmp_path):
        """Errors during Orchestrator initialization are propagated."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "bad_scenario.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"), \
             patch("server.event_bus.emit_event") as mock_emit:

            mock_orch_class.side_effect = ValueError("Invalid scenario config")

            with pytest.raises(ValueError, match="Invalid scenario config"):
                run_simulation_task("init-fail", str(scenario_path))

            mock_emit.assert_called_once()

    def test_nonexistent_scenario_path(self):
        """run_simulation_task handles non-existent scenario path."""
        from server.worker_tasks import run_simulation_task

        with patch("core.event_emitter.enable_event_emitter"), \
             patch("server.event_bus.emit_event"):

            # This will fail when Orchestrator tries to load the file
            with patch("core.orchestrator.Orchestrator") as mock_orch_class:
                mock_orch_class.side_effect = FileNotFoundError("Scenario not found")

                with pytest.raises(FileNotFoundError):
                    run_simulation_task("missing-scenario", "/nonexistent/path.yaml")


class TestRunSimulationTaskWithResume:
    """Tests for resume functionality in run_simulation_task."""

    def test_run_simulation_task_with_resume(self, tmp_path):
        """Test running a simulation task that resumes from a saved state."""
        from server.worker_tasks import run_simulation_task

        # Create a dummy scenario file
        scenario_path = tmp_path / "resume_scenario.yaml"
        scenario_path.write_text("name: test\nmax_steps: 10\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter") as mock_enable:

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = run_simulation_task(
                "resume-run-123",
                str(scenario_path),
                resume_from_step=5
            )

            # Verify orchestrator.run was called with start_step
            mock_orchestrator.run.assert_called_once_with(start_step=5)

            # Verify result includes resume information
            assert result["run_id"] == "resume-run-123"
            assert result["status"] == "completed"
            assert result["scenario"] == "resume_scenario"
            assert result["resumed_from"] == 5

    def test_run_simulation_task_without_resume_no_start_step(self, tmp_path):
        """Test that run without resume_from_step calls run() without start_step."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "normal_scenario.yaml"
        scenario_path.write_text("name: test\nmax_steps: 5\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = run_simulation_task("normal-run", str(scenario_path))

            # Verify orchestrator.run was called without start_step
            mock_orchestrator.run.assert_called_once_with()

            # Result should not have resumed_from key
            assert "resumed_from" not in result

    def test_resume_from_step_zero_passes_start_step(self, tmp_path):
        """Test that resume_from_step=0 still passes start_step to orchestrator."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "step_zero_scenario.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            # resume_from_step=0 is explicitly set, so should pass start_step=0
            result = run_simulation_task(
                "step-zero-run",
                str(scenario_path),
                resume_from_step=0
            )

            # Even with 0, should call run(start_step=0) since it's not None
            mock_orchestrator.run.assert_called_once_with(start_step=0)
            assert result["resumed_from"] == 0

    def test_resume_from_step_one_passes_start_step(self, tmp_path):
        """Test that resume_from_step=1 passes start_step to orchestrator."""
        from server.worker_tasks import run_simulation_task

        scenario_path = tmp_path / "step_one_scenario.yaml"
        scenario_path.write_text("name: test\n")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = run_simulation_task(
                "step-one-run",
                str(scenario_path),
                resume_from_step=1
            )

            # resume_from_step=1 is truthy, should pass start_step
            mock_orchestrator.run.assert_called_once_with(start_step=1)
            assert result["resumed_from"] == 1
