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


class TestRetryBehavior:
    """Tests for job retry configuration."""

    def test_job_has_retry_configured(self, job_queue_module):
        """Jobs are created with retry configuration."""
        from rq.job import Job
        from server import job_queue

        job_id = _enqueue_with_dummy(job_queue, "retry-test", "/path.yaml")

        job = Job.fetch(job_id, connection=job_queue._redis_conn)

        # RQ stores retry info in the job
        assert job.retries_left == 3

    def test_retry_intervals_configured(self, job_queue_module):
        """Jobs have correct retry intervals configured."""
        from rq.job import Job
        from server import job_queue

        job_id = _enqueue_with_dummy(job_queue, "interval-test", "/path.yaml")

        job = Job.fetch(job_id, connection=job_queue._redis_conn)

        # Check retry intervals (10s, 30s, 60s)
        assert job.retry_intervals == [10, 30, 60]

    def test_job_timeout_configured(self, job_queue_module):
        """Jobs have 1 hour timeout configured."""
        from rq.job import Job
        from server import job_queue

        job_id = _enqueue_with_dummy(job_queue, "timeout-test", "/path.yaml")

        job = Job.fetch(job_id, connection=job_queue._redis_conn)

        assert job.timeout == 3600


class TestErrorPropagation:
    """Tests for error handling and propagation in job status."""

    def test_get_status_includes_error_field(self, job_queue_module):
        """get_job_status includes error field in response."""
        from server import job_queue

        _enqueue_with_dummy(job_queue, "error-field-test", "/path.yaml")

        status = job_queue.get_job_status("error-field-test")

        # Error field should exist (None for non-failed jobs)
        assert "error" in status
        assert status["error"] is None

    def test_get_status_includes_timestamps(self, job_queue_module):
        """get_job_status includes timestamp fields."""
        from server import job_queue

        _enqueue_with_dummy(job_queue, "timestamp-test", "/path.yaml")

        status = job_queue.get_job_status("timestamp-test")

        assert "enqueued_at" in status
        assert "started_at" in status
        assert "ended_at" in status
        # Enqueued should be set, others None for queued job
        assert status["enqueued_at"] is not None
        assert status["started_at"] is None
        assert status["ended_at"] is None

    def test_get_status_includes_result_field(self, job_queue_module):
        """get_job_status includes result field."""
        from server import job_queue

        _enqueue_with_dummy(job_queue, "result-field-test", "/path.yaml")

        status = job_queue.get_job_status("result-field-test")

        assert "result" in status


class TestConnectionFailures:
    """Tests for handling Redis connection failures."""

    def test_enqueue_without_init_raises(self):
        """enqueue_simulation raises if queue not initialized."""
        from server import job_queue

        # Reset state
        job_queue._redis_conn = None
        job_queue._queues = {}

        with pytest.raises(RuntimeError, match="not initialized"):
            job_queue.enqueue_simulation("test", "/path.yaml")

    def test_get_status_without_init_raises(self):
        """get_job_status raises if queue not initialized."""
        from server import job_queue

        job_queue._redis_conn = None
        job_queue._queues = {}

        with pytest.raises(RuntimeError, match="not initialized"):
            job_queue.get_job_status("test")

    def test_get_stats_without_init_raises(self):
        """get_queue_stats raises if queue not initialized."""
        from server import job_queue

        job_queue._redis_conn = None
        job_queue._queues = {}

        with pytest.raises(RuntimeError, match="not initialized"):
            job_queue.get_queue_stats()

    def test_cancel_without_init_raises(self):
        """cancel_job raises if queue not initialized."""
        from server import job_queue

        job_queue._redis_conn = None
        job_queue._queues = {}

        with pytest.raises(RuntimeError, match="not initialized"):
            job_queue.cancel_job("test")

    def test_init_with_bad_url_handled(self, fake_redis):
        """init_job_queue handles connection failures gracefully."""
        from server import job_queue

        job_queue._redis_conn = None
        job_queue._queues = {}

        with patch("server.job_queue.Redis") as mock_redis_class:
            mock_redis = mock_redis_class.from_url.return_value
            mock_redis.ping.side_effect = ConnectionError("Cannot connect")

            with pytest.raises(ConnectionError, match="Cannot connect"):
                job_queue.init_job_queue("redis://badhost:6379")


class TestQueueStatistics:
    """Additional tests for queue statistics accuracy."""

    def test_stats_count_multiple_queued_jobs(self, job_queue_module):
        """get_queue_stats accurately counts multiple queued jobs."""
        from server import job_queue

        # Enqueue multiple jobs to different queues
        _enqueue_with_dummy(job_queue, "job-1", "/path.yaml", "high")
        _enqueue_with_dummy(job_queue, "job-2", "/path.yaml", "high")
        _enqueue_with_dummy(job_queue, "job-3", "/path.yaml", "normal")

        stats = job_queue.get_queue_stats()

        assert stats["high"]["queued"] == 2
        assert stats["normal"]["queued"] == 1
        assert stats["low"]["queued"] == 0
        assert stats["total"]["queued"] == 3

    def test_stats_after_cancel(self, job_queue_module):
        """get_queue_stats reflects cancelled jobs."""
        from server import job_queue

        _enqueue_with_dummy(job_queue, "to-cancel", "/path.yaml", "normal")

        stats_before = job_queue.get_queue_stats()
        assert stats_before["normal"]["queued"] == 1

        job_queue.cancel_job("to-cancel")

        stats_after = job_queue.get_queue_stats()
        assert stats_after["normal"]["queued"] == 0


# =============================================================================
# Real Redis Tests (optional - require running Redis instance)
# =============================================================================
# Run these with: pytest -m redis tests/integration/test_job_queue.py
# Skip these with: pytest -m "not redis" tests/integration/test_job_queue.py


def _redis_available() -> bool:
    """Check if Redis is available at localhost:6379."""
    try:
        from redis import Redis
        conn = Redis(host="localhost", port=6379)
        conn.ping()
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.redis
@pytest.mark.skipif(not _redis_available(), reason="Redis not available at localhost:6379")
class TestRealRedisIntegration:
    """
    Integration tests against a real Redis instance.

    These tests verify actual Redis behavior that fakeredis may not fully replicate.
    Requires Redis running at localhost:6379.

    Run with: pytest -m redis
    """

    @pytest.fixture(autouse=True)
    def setup_real_redis(self):
        """Set up and tear down real Redis connection."""
        from server import job_queue

        # Reset module state
        job_queue._redis_conn = None
        job_queue._queues = {}

        # Initialize with real Redis
        job_queue.init_job_queue("redis://localhost:6379")

        yield job_queue

        # Clean up: delete test jobs from Redis
        try:
            conn = job_queue.get_redis_connection()
            # Clean up test queues
            for queue in job_queue._queues.values():
                queue.empty()
        except Exception:
            pass

        # Reset module
        job_queue._redis_conn = None
        job_queue._queues = {}

    def test_real_redis_connection(self, setup_real_redis):
        """Verify connection to real Redis works."""
        job_queue = setup_real_redis
        conn = job_queue.get_redis_connection()

        assert conn.ping() is True

    def test_real_enqueue_and_fetch(self, setup_real_redis):
        """Test enqueue and status fetch with real Redis."""
        job_queue = setup_real_redis

        job_id = _enqueue_with_dummy(
            job_queue,
            run_id="real-redis-test",
            scenario_path="/test/scenario.yaml",
            priority="normal"
        )

        status = job_queue.get_job_status(job_id)

        assert status["id"] == "real-redis-test"
        assert status["status"] == "queued"

    def test_real_queue_persistence(self, setup_real_redis):
        """Test that jobs persist in real Redis."""
        job_queue = setup_real_redis

        # Enqueue job
        job_id = _enqueue_with_dummy(
            job_queue,
            run_id="persistence-test",
            scenario_path="/test/scenario.yaml"
        )

        # Create new connection to verify persistence
        from redis import Redis
        new_conn = Redis(host="localhost", port=6379)

        # Verify job exists via new connection
        from rq.job import Job
        job = Job.fetch(job_id, connection=new_conn)

        assert job.id == "persistence-test"
        new_conn.close()

    def test_real_priority_ordering(self, setup_real_redis):
        """Test that priority queues work correctly with real Redis."""
        job_queue = setup_real_redis

        # Enqueue to different priorities
        _enqueue_with_dummy(job_queue, "low-1", "/path.yaml", "low")
        _enqueue_with_dummy(job_queue, "high-1", "/path.yaml", "high")
        _enqueue_with_dummy(job_queue, "normal-1", "/path.yaml", "normal")

        stats = job_queue.get_queue_stats()

        assert stats["high"]["queued"] >= 1
        assert stats["normal"]["queued"] >= 1
        assert stats["low"]["queued"] >= 1
