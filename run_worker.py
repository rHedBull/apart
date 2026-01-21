#!/usr/bin/env python
"""
Run RQ worker for processing simulation jobs.

Usage:
    python run_worker.py                    # Use default Redis URL
    REDIS_URL=redis://host:6379 python run_worker.py

Workers process jobs from priority queues in order: high -> normal -> low.
Multiple workers can run concurrently for horizontal scaling.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from redis import Redis
from rq import Worker, Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rq.worker")


def main():
    """Start the RQ worker."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    logger.info(f"Connecting to Redis at {redis_url}")

    redis_conn = Redis.from_url(redis_url)

    # Test connection
    try:
        redis_conn.ping()
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Listen to multiple queues with priority order
    # Workers process queues in list order, so high-priority jobs are handled first
    queues = [
        Queue("simulations-high", connection=redis_conn),
        Queue("simulations", connection=redis_conn),
        Queue("simulations-low", connection=redis_conn),
    ]

    # Initialize database if enabled (workers need DB access for event persistence)
    if os.environ.get("APART_USE_DATABASE", "").lower() in ("1", "true", "yes"):
        from server.database import init_db
        from server.event_bus import EventBus
        EventBus._use_database = True
        init_db()
        logger.info("Database persistence enabled")

    logger.info(f"Worker starting, listening to queues: {[q.name for q in queues]}")

    worker = Worker(queues, connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
