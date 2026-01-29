"""Unit tests for job queue module."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest


class TestJobQueueInitialization:
    """Tests for job queue initialization."""

    def teardown_method(self):
        """Reset module state after each test."""
        import server.job_queue as jq
        jq._redis_conn = None
        jq._queues = {}

    def test_init_job_queue_creates_connection(self):
        """Test that init_job_queue creates Redis connection."""
        import server.job_queue as jq

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        with patch("server.job_queue.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis

            jq.init_job_queue("redis://localhost:6379")

            MockRedis.from_url.assert_called_once_with("redis://localhost:6379")
            mock_redis.ping.assert_called_once()

    def test_init_job_queue_creates_priority_queues(self):
        """Test that init_job_queue creates all priority queues."""
        import server.job_queue as jq

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        with patch("server.job_queue.Redis") as MockRedis, \
             patch("server.job_queue.Queue") as MockQueue:
            MockRedis.from_url.return_value = mock_redis

            jq.init_job_queue()

            # Should create 3 queues
            assert MockQueue.call_count == 3

            # Check queue names
            call_args = [call[0] for call in MockQueue.call_args_list]
            queue_names = [args[0] for args in call_args]
            assert "simulations-high" in queue_names
            assert "simulations" in queue_names
            assert "simulations-low" in queue_names

    def test_get_redis_connection_raises_when_not_initialized(self):
        """Test that get_redis_connection raises when not initialized."""
        import server.job_queue as jq

        with pytest.raises(RuntimeError, match="not initialized"):
            jq.get_redis_connection()

    def test_get_redis_connection_returns_connection(self):
        """Test that get_redis_connection returns the connection."""
        import server.job_queue as jq

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        with patch("server.job_queue.Redis") as MockRedis, \
             patch("server.job_queue.Queue"):
            MockRedis.from_url.return_value = mock_redis

            jq.init_job_queue()
            conn = jq.get_redis_connection()

            assert conn is mock_redis


class TestEnqueueSimulation:
    """Tests for enqueue_simulation function."""

    def setup_method(self):
        """Initialize job queue with mocks before each test."""
        import server.job_queue as jq

        self.mock_redis = MagicMock()
        self.mock_redis.ping.return_value = True
        self.mock_queues = {}

        with patch("server.job_queue.Redis") as MockRedis:
            MockRedis.from_url.return_value = self.mock_redis

            with patch("server.job_queue.Queue") as MockQueue:
                def create_mock_queue(name, **kwargs):
                    q = MagicMock()
                    q.name = name
                    self.mock_queues[name] = q
                    return q
                MockQueue.side_effect = create_mock_queue

                jq.init_job_queue()

    def teardown_method(self):
        """Reset module state after each test."""
        import server.job_queue as jq
        jq._redis_conn = None
        jq._queues = {}

    def test_enqueue_simulation_raises_when_not_initialized(self):
        """Test enqueue raises when queue not initialized."""
        import server.job_queue as jq
        jq._redis_conn = None  # Simulate uninitialized state

        with pytest.raises(RuntimeError, match="not initialized"):
            jq.enqueue_simulation("run-1", "/path/scenario.yaml")

    def test_enqueue_simulation_invalid_priority(self):
        """Test enqueue raises for invalid priority."""
        import server.job_queue as jq

        with pytest.raises(ValueError, match="Invalid priority"):
            jq.enqueue_simulation("run-1", "/path.yaml", priority="ultra")

    def test_enqueue_simulation_uses_correct_queue(self):
        """Test that enqueue uses the correct priority queue."""
        import server.job_queue as jq

        with patch("server.worker_tasks.run_simulation_task"):
            # Test each priority
            for priority, queue_name in [("high", "simulations-high"),
                                          ("normal", "simulations"),
                                          ("low", "simulations-low")]:
                jq.enqueue_simulation(f"run-{priority}", "/path.yaml", priority=priority)
                self.mock_queues[queue_name].enqueue.assert_called()

    def test_enqueue_simulation_returns_job_id(self):
        """Test that enqueue returns the job ID."""
        import server.job_queue as jq

        mock_job = MagicMock()
        mock_job.id = "job-12345"
        self.mock_queues["simulations"].enqueue.return_value = mock_job

        with patch("server.worker_tasks.run_simulation_task"):
            job_id = jq.enqueue_simulation("run-x", "/path.yaml")

        assert job_id == "job-12345"

    def test_enqueue_simulation_sets_job_options(self):
        """Test that enqueue sets correct job options."""
        import server.job_queue as jq

        mock_job = MagicMock()
        mock_job.id = "job-1"
        self.mock_queues["simulations"].enqueue.return_value = mock_job

        with patch("server.worker_tasks.run_simulation_task"):
            jq.enqueue_simulation("run-opt", "/scenario.yaml")

        # Check that enqueue was called with expected kwargs
        call_kwargs = self.mock_queues["simulations"].enqueue.call_args[1]
        assert call_kwargs["job_id"] == "run-opt"
        assert call_kwargs["job_timeout"] == 3600
        assert "retry" in call_kwargs
        assert "meta" in call_kwargs


class TestGetJobStatus:
    """Tests for get_job_status function."""

    def setup_method(self):
        """Initialize job queue with mocks."""
        import server.job_queue as jq

        self.mock_redis = MagicMock()
        jq._redis_conn = self.mock_redis
        jq._queues = {"normal": MagicMock()}

    def teardown_method(self):
        """Reset module state."""
        import server.job_queue as jq
        jq._redis_conn = None
        jq._queues = {}

    def test_get_job_status_raises_when_not_initialized(self):
        """Test get_job_status raises when not initialized."""
        import server.job_queue as jq
        jq._redis_conn = None

        with pytest.raises(RuntimeError, match="not initialized"):
            jq.get_job_status("job-1")

    def test_get_job_status_raises_for_missing_job(self):
        """Test get_job_status raises ValueError for missing job."""
        import server.job_queue as jq

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.side_effect = Exception("Job not found")

            with pytest.raises(ValueError, match="not found"):
                jq.get_job_status("nonexistent")

    def test_get_job_status_returns_status_dict(self):
        """Test get_job_status returns proper status dictionary."""
        import server.job_queue as jq
        from datetime import datetime

        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.get_status.return_value = "finished"
        mock_job.enqueued_at = datetime(2024, 1, 1, 0, 0, 0)
        mock_job.started_at = datetime(2024, 1, 1, 0, 1, 0)
        mock_job.ended_at = datetime(2024, 1, 1, 0, 2, 0)
        mock_job.return_value = {"result": "success"}
        mock_job.meta = {"priority": "normal"}
        mock_job.latest_result.return_value = None

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.return_value = mock_job

            status = jq.get_job_status("job-123")

        assert status["id"] == "job-123"
        assert status["status"] == "finished"
        assert status["enqueued_at"] is not None
        assert status["started_at"] is not None
        assert status["ended_at"] is not None
        assert status["result"] == {"result": "success"}
        assert status["meta"]["priority"] == "normal"

    def test_get_job_status_includes_error_on_failure(self):
        """Test get_job_status includes error message for failed jobs."""
        import server.job_queue as jq

        mock_job = MagicMock()
        mock_job.id = "failed-job"
        mock_job.get_status.return_value = "failed"
        mock_job.enqueued_at = None
        mock_job.started_at = None
        mock_job.ended_at = None
        mock_job.return_value = None
        mock_job.meta = {}

        mock_result = MagicMock()
        mock_result.type.name = "FAILED"
        mock_result.exc_string = "ValueError: Something went wrong"
        mock_job.latest_result.return_value = mock_result

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.return_value = mock_job

            status = jq.get_job_status("failed-job")

        assert status["error"] is not None
        assert "ValueError" in status["error"]


class TestGetQueueStats:
    """Tests for get_queue_stats function."""

    def setup_method(self):
        """Initialize job queue with mocks."""
        import server.job_queue as jq

        self.mock_redis = MagicMock()
        jq._redis_conn = self.mock_redis

        # Create mock queues with stats
        self.mock_high = MagicMock()
        self.mock_high.__len__ = MagicMock(return_value=5)
        self.mock_high.failed_job_registry = [1, 2]
        self.mock_high.started_job_registry = [1]
        self.mock_high.finished_job_registry = [1, 2, 3]

        self.mock_normal = MagicMock()
        self.mock_normal.__len__ = MagicMock(return_value=10)
        self.mock_normal.failed_job_registry = [1]
        self.mock_normal.started_job_registry = [1, 2]
        self.mock_normal.finished_job_registry = [1, 2, 3, 4, 5]

        self.mock_low = MagicMock()
        self.mock_low.__len__ = MagicMock(return_value=2)
        self.mock_low.failed_job_registry = []
        self.mock_low.started_job_registry = []
        self.mock_low.finished_job_registry = [1]

        jq._queues = {
            "high": self.mock_high,
            "normal": self.mock_normal,
            "low": self.mock_low,
        }

    def teardown_method(self):
        """Reset module state."""
        import server.job_queue as jq
        jq._redis_conn = None
        jq._queues = {}

    def test_get_queue_stats_raises_when_not_initialized(self):
        """Test get_queue_stats raises when not initialized."""
        import server.job_queue as jq
        jq._redis_conn = None

        with pytest.raises(RuntimeError, match="not initialized"):
            jq.get_queue_stats()

    def test_get_queue_stats_returns_all_queues(self):
        """Test get_queue_stats returns stats for all queues."""
        import server.job_queue as jq

        stats = jq.get_queue_stats()

        assert "high" in stats
        assert "normal" in stats
        assert "low" in stats
        assert "total" in stats

    def test_get_queue_stats_counts_correctly(self):
        """Test get_queue_stats counts jobs correctly."""
        import server.job_queue as jq

        stats = jq.get_queue_stats()

        # Check high queue
        assert stats["high"]["queued"] == 5
        assert stats["high"]["failed"] == 2
        assert stats["high"]["started"] == 1
        assert stats["high"]["finished"] == 3

        # Check totals
        assert stats["total"]["queued"] == 17  # 5 + 10 + 2
        assert stats["total"]["failed"] == 3    # 2 + 1 + 0
        assert stats["total"]["started"] == 3   # 1 + 2 + 0
        assert stats["total"]["finished"] == 9  # 3 + 5 + 1


class TestCancelJob:
    """Tests for cancel_job function."""

    def setup_method(self):
        """Initialize job queue with mocks."""
        import server.job_queue as jq

        self.mock_redis = MagicMock()
        jq._redis_conn = self.mock_redis
        jq._queues = {"normal": MagicMock()}

    def teardown_method(self):
        """Reset module state."""
        import server.job_queue as jq
        jq._redis_conn = None
        jq._queues = {}

    def test_cancel_job_raises_when_not_initialized(self):
        """Test cancel_job raises when not initialized."""
        import server.job_queue as jq
        jq._redis_conn = None

        with pytest.raises(RuntimeError, match="not initialized"):
            jq.cancel_job("job-1")

    def test_cancel_queued_job_succeeds(self):
        """Test canceling a queued job returns True."""
        import server.job_queue as jq

        mock_job = MagicMock()
        mock_job.get_status.return_value = "queued"

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.return_value = mock_job

            result = jq.cancel_job("queued-job")

        assert result is True
        mock_job.cancel.assert_called_once()

    def test_cancel_running_job_returns_false(self):
        """Test canceling a running job returns False."""
        import server.job_queue as jq

        mock_job = MagicMock()
        mock_job.get_status.return_value = "started"

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.return_value = mock_job

            result = jq.cancel_job("running-job")

        assert result is False
        mock_job.cancel.assert_not_called()

    def test_cancel_completed_job_returns_false(self):
        """Test canceling a completed job returns False."""
        import server.job_queue as jq

        mock_job = MagicMock()
        mock_job.get_status.return_value = "finished"

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.return_value = mock_job

            result = jq.cancel_job("done-job")

        assert result is False

    def test_cancel_nonexistent_job_returns_false(self):
        """Test canceling nonexistent job returns False."""
        import server.job_queue as jq

        with patch("server.job_queue.Job") as MockJob:
            MockJob.fetch.side_effect = Exception("Job not found")

            result = jq.cancel_job("missing-job")

        assert result is False
