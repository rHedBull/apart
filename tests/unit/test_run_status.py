"""Unit tests for run status determination with RQ job status sync."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from server.routes.v1 import _get_run_status, _get_job_status


class TestGetJobStatus:
    """Tests for _get_job_status helper."""

    def test_returns_job_status_when_exists(self):
        """Test that job status is returned when job exists."""
        with patch("server.job_queue.get_job_status") as mock_get:
            mock_get.return_value = {"status": "started", "id": "test123"}
            status = _get_job_status("test123")
            assert status == "started"

    def test_returns_none_when_job_not_found(self):
        """Test that None is returned when job doesn't exist."""
        with patch("server.job_queue.get_job_status") as mock_get:
            mock_get.side_effect = ValueError("Job not found")
            status = _get_job_status("nonexistent")
            assert status is None

    def test_returns_none_when_queue_not_initialized(self):
        """Test that None is returned when job queue isn't initialized."""
        with patch("server.job_queue.get_job_status") as mock_get:
            mock_get.side_effect = RuntimeError("Job queue not initialized")
            status = _get_job_status("test123")
            assert status is None


class TestGetRunStatus:
    """Tests for _get_run_status with RQ job status sync."""

    def test_returns_none_when_no_history_and_no_results(self):
        """Test that None is returned when run doesn't exist."""
        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = []
            with patch("server.routes.v1._get_job_status", return_value=None):
                with patch.object(Path, "exists", return_value=False):
                    status = _get_run_status("nonexistent")
                    assert status is None

    def test_returns_completed_when_results_dir_exists(self):
        """Test that completed is returned when results directory exists."""
        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = []
            with patch("server.routes.v1._get_job_status", return_value=None):
                with patch.object(Path, "exists", return_value=True):
                    status = _get_run_status("completed_run")
                    assert status == "completed"

    def test_returns_pending_when_job_queued(self):
        """Test that pending is returned when job is queued but no events yet."""
        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = []
            with patch("server.routes.v1._get_job_status", return_value="queued"):
                with patch.object(Path, "exists", return_value=False):
                    status = _get_run_status("queued_run")
                    assert status == "pending"

    def test_returns_running_from_events(self):
        """Test that running status is derived from simulation_started event."""
        mock_event = Mock()
        mock_event.event_type = "simulation_started"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [mock_event]
            with patch("server.routes.v1._get_job_status", return_value="started"):
                status = _get_run_status("running_run")
                assert status == "running"

    def test_returns_paused_from_events(self):
        """Test that paused status is derived from simulation_paused event."""
        started_event = Mock()
        started_event.event_type = "simulation_started"
        paused_event = Mock()
        paused_event.event_type = "simulation_paused"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [started_event, paused_event]
            with patch("server.routes.v1._get_job_status", return_value=None):
                status = _get_run_status("paused_run")
                assert status == "paused"

    def test_returns_completed_from_events(self):
        """Test that completed status is derived from simulation_completed event."""
        started_event = Mock()
        started_event.event_type = "simulation_started"
        completed_event = Mock()
        completed_event.event_type = "simulation_completed"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [started_event, completed_event]
            status = _get_run_status("completed_run")
            assert status == "completed"

    def test_returns_failed_from_events(self):
        """Test that failed status is derived from simulation_failed event."""
        started_event = Mock()
        started_event.event_type = "simulation_started"
        failed_event = Mock()
        failed_event.event_type = "simulation_failed"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [started_event, failed_event]
            status = _get_run_status("failed_run")
            assert status == "failed"

    def test_worker_crash_detected_via_rq_status(self):
        """Test that worker crash is detected when RQ job is failed but no failure event.

        This is the key robustness test: if worker dies without emitting a failure
        event, the RQ job status should still show the run as failed.
        """
        # Only started event, no failure event (simulates worker crash)
        started_event = Mock()
        started_event.event_type = "simulation_started"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [started_event]
            # RQ knows the job failed even though no event was emitted
            with patch("server.routes.v1._get_job_status", return_value="failed"):
                status = _get_run_status("crashed_run")
                assert status == "failed"

    def test_job_finished_without_completion_event(self):
        """Test that finished job with results dir shows as completed."""
        started_event = Mock()
        started_event.event_type = "simulation_started"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [started_event]
            with patch("server.routes.v1._get_job_status", return_value="finished"):
                # Results directory exists
                with patch.object(Path, "exists", return_value=True):
                    status = _get_run_status("finished_run")
                    assert status == "completed"

    def test_resume_after_pause_shows_running(self):
        """Test that resumed status correctly shows as running."""
        started_event = Mock()
        started_event.event_type = "simulation_started"
        paused_event = Mock()
        paused_event.event_type = "simulation_paused"
        resumed_event = Mock()
        resumed_event.event_type = "simulation_resumed"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [
                started_event, paused_event, resumed_event
            ]
            with patch("server.routes.v1._get_job_status", return_value="started"):
                status = _get_run_status("resumed_run")
                assert status == "running"


class TestStatusTransitions:
    """Tests for status transition edge cases."""

    def test_multiple_pause_resume_cycles(self):
        """Test that multiple pause/resume cycles resolve correctly."""
        events = []
        for event_type in ["simulation_started", "simulation_paused",
                          "simulation_resumed", "simulation_paused",
                          "simulation_resumed"]:
            event = Mock()
            event.event_type = event_type
            events.append(event)

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = events
            with patch("server.routes.v1._get_job_status", return_value="started"):
                status = _get_run_status("multi_cycle_run")
                assert status == "running"

    def test_final_status_wins(self):
        """Test that the final event determines status."""
        events = []
        for event_type in ["simulation_started", "simulation_paused",
                          "simulation_resumed", "simulation_completed"]:
            event = Mock()
            event.event_type = event_type
            events.append(event)

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = events
            status = _get_run_status("completed_after_resume")
            assert status == "completed"

    def test_worker_crash_shows_interrupted_when_rescheduled(self):
        """Test that job back in queue after crash shows as interrupted.

        When a worker dies mid-job, RQ reschedules it. If events show
        "running" but RQ shows "scheduled", the run is interrupted.
        """
        started_event = Mock()
        started_event.event_type = "simulation_started"

        with patch("server.routes.v1.get_event_bus") as mock_bus:
            mock_bus.return_value.get_history.return_value = [started_event]
            # RQ rescheduled the job after worker death
            with patch("server.routes.v1._get_job_status", return_value="scheduled"):
                status = _get_run_status("rescheduled_run")
                assert status == "interrupted"
