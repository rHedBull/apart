"""
Integration tests for the Redis job queue.

Uses fakeredis for CI testing without requiring a real Redis instance.
Note: RQ uses pickle for job serialization which is required for this functionality.
"""

import pytest
from unittest.mock import patch

# Skip tests if fakeredis not installed
fakeredis = pytest.importorskip("fakeredis")


# A simple test task function that can be serialized by RQ
def dummy_task(run_id: str, scenario_path: str) -> dict:
    """Dummy task for testing - does nothing."""
    return {"run_id": run_id, "scenario_path": scenario_path}


@pytest.fixture
def fake_redis():
    """Create a fake Redis instance for testing."""
    return fakeredis.FakeRedis()


@pytest.fixture
def job_queue_module(fake_redis):
    """
    Set up job_queue module with fake Redis connection.

    Patches the module-level connections before any imports.
    """
    from server import job_queue

    # Reset module state
    job_queue._redis_conn = None
    job_queue._queues = {}

    # Patch Redis.from_url to return our fake Redis
    with patch("server.job_queue.Redis") as mock_redis_class:
        mock_redis_class.from_url.return_value = fake_redis
        job_queue.init_job_queue("redis://fake:6379")

    yield job_queue

    # Clean up
    job_queue._redis_conn = None
    job_queue._queues = {}


class TestJobQueueInit:
    """Tests for job queue initialization."""

    def test_init_creates_queues(self, fake_redis):
        """init_job_queue creates priority queues."""
        from server import job_queue

        with patch("server.job_queue.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = fake_redis
            job_queue.init_job_queue("redis://test:6379")

        assert "high" in job_queue._queues
        assert "normal" in job_queue._queues
        assert "low" in job_queue._queues

        # Clean up
        job_queue._redis_conn = None
        job_queue._queues = {}

    def test_get_redis_connection_raises_if_not_initialized(self):
        """get_redis_connection raises if queue not initialized."""
        from server import job_queue

        # Reset state
        job_queue._redis_conn = None
        job_queue._queues = {}

        with pytest.raises(RuntimeError, match="not initialized"):
            job_queue.get_redis_connection()


class TestEnqueueSimulation:
    """Tests for enqueue_simulation function."""

    def test_enqueue_creates_job(self, job_queue_module, monkeypatch):
        """enqueue_simulation creates a job with correct parameters."""
        from server import job_queue

        # Replace the import in job_queue with our dummy task
        monkeypatch.setattr(
            "server.job_queue.enqueue_simulation",
            lambda run_id, scenario_path, priority="normal": _enqueue_with_dummy(
                job_queue, run_id, scenario_path, priority
            ),
        )

        job_id = _enqueue_with_dummy(
            job_queue,
            run_id="test-run-123",
            scenario_path="/path/to/scenario.yaml",
            priority="normal",
        )

        assert job_id == "test-run-123"

    def test_enqueue_respects_priority(self, job_queue_module):
        """enqueue_simulation puts jobs in correct priority queue."""
        from server import job_queue

        # Enqueue to each priority using the queue directly
        for priority, run_id in [
            ("high", "high-job"),
            ("normal", "normal-job"),
            ("low", "low-job"),
        ]:
            queue = job_queue._queues[priority]
            queue.enqueue(dummy_task, args=(run_id, "/path.yaml"), job_id=run_id)

        # Check queue lengths
        stats = job_queue.get_queue_stats()
        assert stats["total"]["queued"] == 3

    def test_enqueue_invalid_priority_raises(self, job_queue_module):
        """enqueue_simulation raises for invalid priority."""
        from server import job_queue

        with pytest.raises(ValueError, match="Invalid priority"):
            job_queue.enqueue_simulation("test-run", "/path.yaml", "ultra-high")


def _enqueue_with_dummy(job_queue, run_id, scenario_path, priority="normal"):
    """Helper to enqueue using the dummy task instead of the real one."""
    from rq import Retry

    queue = job_queue._queues[priority]
    job = queue.enqueue(
        dummy_task,
        args=(run_id, scenario_path),
        job_id=run_id,
        job_timeout=3600,
        retry=Retry(max=3, interval=[10, 30, 60]),
        meta={"scenario_path": scenario_path, "priority": priority},
    )
    return job.id


class TestGetJobStatus:
    """Tests for get_job_status function."""

    def test_get_status_returns_job_info(self, job_queue_module):
        """get_job_status returns correct job information."""
        from server import job_queue

        # Enqueue with dummy task
        _enqueue_with_dummy(job_queue, "status-test", "/path.yaml")

        status = job_queue.get_job_status("status-test")

        assert status["id"] == "status-test"
        assert status["status"] == "queued"
        assert status["meta"]["scenario_path"] == "/path.yaml"

    def test_get_status_nonexistent_raises(self, job_queue_module):
        """get_job_status raises for non-existent job."""
        from server import job_queue

        with pytest.raises(ValueError, match="not found"):
            job_queue.get_job_status("nonexistent-job")


class TestGetQueueStats:
    """Tests for get_queue_stats function."""

    def test_stats_returns_all_queues(self, job_queue_module):
        """get_queue_stats returns stats for all queues."""
        from server import job_queue

        stats = job_queue.get_queue_stats()

        assert "high" in stats
        assert "normal" in stats
        assert "low" in stats
        assert "total" in stats

        # Check structure
        assert "queued" in stats["total"]
        assert "failed" in stats["total"]
        assert "started" in stats["total"]
        assert "finished" in stats["total"]


class TestCancelJob:
    """Tests for cancel_job function."""

    def test_cancel_queued_job(self, job_queue_module):
        """cancel_job cancels a queued job."""
        from server import job_queue

        _enqueue_with_dummy(job_queue, "cancel-test", "/path.yaml")

        result = job_queue.cancel_job("cancel-test")
        assert result is True

    def test_cancel_nonexistent_returns_false(self, job_queue_module):
        """cancel_job returns False for non-existent job."""
        from server import job_queue

        result = job_queue.cancel_job("nonexistent")
        assert result is False


class TestJobQueueIntegration:
    """Integration tests for the full job queue workflow."""

    def test_full_workflow(self, job_queue_module):
        """Test complete workflow: enqueue -> check status -> cancel."""
        from server import job_queue

        # Enqueue
        job_id = _enqueue_with_dummy(
            job_queue,
            run_id="workflow-test",
            scenario_path="/scenarios/test.yaml",
            priority="high",
        )

        # Check status
        status = job_queue.get_job_status(job_id)
        assert status["status"] == "queued"

        # Check stats
        stats = job_queue.get_queue_stats()
        assert stats["high"]["queued"] >= 1

        # Cancel
        cancelled = job_queue.cancel_job(job_id)
        assert cancelled is True
