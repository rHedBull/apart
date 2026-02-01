"""
Redis Queue (RQ) integration for distributed job processing.

Enables horizontal scaling by offloading simulation jobs to Redis-backed workers.
"""

import json
from pathlib import Path

from rq import Queue, Retry
from rq.job import Job
from redis import Redis
from typing import Optional

from utils.ops_logger import get_ops_logger

logger = get_ops_logger("queue")

# Module-level connections (initialized once)
_redis_conn: Optional[Redis] = None
_queues: dict[str, Queue] = {}

# Pause signal key prefix
PAUSE_SIGNAL_PREFIX = "apart:pause:"


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

    logger.info("Job queue initialized", extra={"redis_url": redis_url})


def get_redis_connection() -> Redis:
    """Get the Redis connection instance."""
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")
    return _redis_conn


def enqueue_simulation(
    run_id: str,
    scenario_path: str,
    priority: str = "normal",
    resume_from_step: int | None = None,
) -> str:
    """
    Enqueue a simulation job for worker processing.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML file
        priority: Queue priority - "high", "normal", or "low"
        resume_from_step: If set, resume simulation from this step

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

    # Create run state entry (if state manager is initialized)
    from server.run_state import get_state_manager
    state_manager = get_state_manager()
    if state_manager is not None and resume_from_step is None:
        # Only create new state for new runs, not resumes
        try:
            scenario_name = Path(scenario_path).stem
            state_manager.create_run(
                run_id=run_id,
                scenario_path=scenario_path,
                scenario_name=scenario_name,
                priority=priority,
            )
        except ValueError:
            # Run already exists (e.g., resume scenario) - that's fine
            pass
    elif state_manager is not None and resume_from_step is not None:
        # For resumes, transition back to running
        try:
            state_manager.transition(run_id, "running")
        except Exception as e:
            logger.warning(f"Could not transition {run_id} to running: {e}")

    # Import here to avoid circular imports
    from server.worker_tasks import run_simulation_task

    # Build job args - include resume_from_step if resuming
    job_args = (run_id, scenario_path)
    job_kwargs = {}
    if resume_from_step is not None:
        job_kwargs["resume_from_step"] = resume_from_step

    job = queue.enqueue(
        run_simulation_task,
        args=job_args,
        kwargs=job_kwargs,
        job_id=run_id,
        job_timeout=3600,  # 1 hour max
        retry=Retry(max=3, interval=[10, 30, 60]),  # Retry with backoff
        meta={
            "scenario_path": scenario_path,
            "priority": priority,
            "resume_from_step": resume_from_step,
        },
    )

    logger.info("Simulation enqueued", extra={
        "run_id": run_id,
        "job_id": job.id,
        "priority": priority,
        "scenario": scenario_path,
        "resume_from_step": resume_from_step,
    })
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
            logger.info("Job cancelled", extra={"job_id": job_id})
            return True
        else:
            logger.warning("Cannot cancel job", extra={"job_id": job_id, "status": status})
            return False
    except Exception:
        return False


def publish_pause_signal(run_id: str, force: bool = False) -> bool:
    """
    Publish a pause signal for a running simulation.

    Args:
        run_id: The simulation run ID to pause
        force: If True, pause immediately (drop current step)

    Returns:
        True if signal was published successfully
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    signal_data = json.dumps({"force": force})
    key = f"{PAUSE_SIGNAL_PREFIX}{run_id}"

    # Set with expiry (5 minutes - plenty of time for worker to see it)
    _redis_conn.setex(key, 300, signal_data)

    logger.info("Pause signal published", extra={"run_id": run_id, "force": force})
    return True


def check_pause_requested(run_id: str) -> dict | None:
    """
    Check if a pause has been requested for a simulation.

    Args:
        run_id: The simulation run ID to check

    Returns:
        Dict with pause info if requested, None otherwise
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    key = f"{PAUSE_SIGNAL_PREFIX}{run_id}"
    data = _redis_conn.get(key)

    if data:
        return json.loads(data)
    return None


def clear_pause_signal(run_id: str) -> bool:
    """
    Clear the pause signal for a simulation (after handling it).

    Args:
        run_id: The simulation run ID

    Returns:
        True if signal was cleared
    """
    if _redis_conn is None:
        raise RuntimeError("Job queue not initialized. Call init_job_queue() first.")

    key = f"{PAUSE_SIGNAL_PREFIX}{run_id}"
    _redis_conn.delete(key)
    return True
