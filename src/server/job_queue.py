"""
Redis Queue (RQ) integration for distributed job processing.

Enables horizontal scaling by offloading simulation jobs to Redis-backed workers.
Only active when APART_USE_JOB_QUEUE=1 is set.
"""

from rq import Queue, Retry
from rq.job import Job
from redis import Redis
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Module-level connections (initialized once)
_redis_conn: Optional[Redis] = None
_queues: dict[str, Queue] = {}


def init_job_queue(redis_url: str = "redis://localhost:6379") -> None:
    """
    Initialize Redis connection and job queues.

    Args:
        redis_url: Redis connection URL
    """
    global _redis_conn, _queues

    _redis_conn = Redis.from_url(redis_url)

    # Test connection
    _redis_conn.ping()

    # Create priority queues (workers process in order: high -> normal -> low)
    _queues = {
        "high": Queue("simulations-high", connection=_redis_conn),
        "normal": Queue("simulations", connection=_redis_conn),
        "low": Queue("simulations-low", connection=_redis_conn),
    }

    logger.info(f"Job queue initialized with Redis at {redis_url}")


def get_redis_connection() -> Redis:
    """Get the Redis connection instance."""
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")
    return _redis_conn


def enqueue_simulation(
    run_id: str,
    scenario_path: str,
    priority: str = "normal",
) -> str:
    """
    Enqueue a simulation job for worker processing.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML file
        priority: Queue priority - "high", "normal", or "low"

    Returns:
        Job ID for tracking

    Raises:
        RuntimeError: If job queue not initialized
        ValueError: If invalid priority
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    if priority not in _queues:
        raise ValueError(f"Invalid priority: {priority}. Must be one of: high, normal, low")

    queue = _queues[priority]

    # Import here to avoid circular imports
    from server.worker_tasks import run_simulation_task

    job = queue.enqueue(
        run_simulation_task,
        args=(run_id, scenario_path),
        job_id=run_id,
        job_timeout=3600,  # 1 hour max
        retry=Retry(max=3, interval=[10, 30, 60]),  # Retry with backoff
        meta={"scenario_path": scenario_path, "priority": priority},
    )

    logger.info(f"Enqueued simulation {run_id} to {priority} queue")
    return job.id


def get_job_status(job_id: str) -> dict:
    """
    Get status of a queued job.

    Args:
        job_id: The job ID to look up

    Returns:
        Dictionary with job status information

    Raises:
        RuntimeError: If job queue not initialized
        ValueError: If job not found
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    try:
        job = Job.fetch(job_id, connection=_redis_conn)
    except Exception as e:
        raise ValueError(f"Job {job_id} not found") from e

    # Get error from latest result if available
    error = None
    try:
        latest_result = job.latest_result()
        if latest_result and latest_result.type.name == "FAILED":
            error = str(latest_result.exc_string) if latest_result.exc_string else None
    except Exception:
        pass

    return {
        "id": job.id,
        "status": job.get_status(),
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "result": job.return_value,
        "error": error,
        "meta": job.meta,
    }


def get_queue_stats() -> dict:
    """
    Get statistics for all queues.

    Returns:
        Dictionary with queue statistics
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    stats = {}
    total_queued = 0
    total_failed = 0
    total_started = 0
    total_finished = 0

    for priority, queue in _queues.items():
        queued = len(queue)
        failed = len(queue.failed_job_registry)
        started = len(queue.started_job_registry)
        finished = len(queue.finished_job_registry)

        stats[priority] = {
            "queued": queued,
            "failed": failed,
            "started": started,
            "finished": finished,
        }

        total_queued += queued
        total_failed += failed
        total_started += started
        total_finished += finished

    stats["total"] = {
        "queued": total_queued,
        "failed": total_failed,
        "started": total_started,
        "finished": total_finished,
    }

    return stats


def cancel_job(job_id: str) -> bool:
    """
    Cancel a queued job.

    Args:
        job_id: The job ID to cancel

    Returns:
        True if job was cancelled, False if job was already running/completed
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    try:
        job = Job.fetch(job_id, connection=_redis_conn)
        status = job.get_status()

        if status == "queued":
            job.cancel()
            logger.info(f"Cancelled job {job_id}")
            return True
        else:
            logger.warning(f"Cannot cancel job {job_id} with status {status}")
            return False
    except Exception:
        return False
