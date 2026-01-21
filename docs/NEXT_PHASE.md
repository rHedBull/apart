# Next Phase: Distributed Job Queue

## Overview

This phase adds Redis-backed job queue for distributed deployments. Only needed when:
- Running multiple server instances behind a load balancer
- Simulations need to survive server restarts mid-execution
- Requiring job prioritization, retry logic, or rate limiting

## Current State

Fully implemented:
- ✅ Event persistence (JSONL + SQLite options)
- ✅ Redis job queue (RQ) for distributed processing
- ✅ Priority queues (high/normal/low)
- ✅ Retry policies with exponential backoff
- ✅ Horizontal scaling via worker processes

## What's Missing for Scale

| Gap | Impact | Solution |
|-----|--------|----------|
| Single-server execution | Can't scale horizontally | Redis job queue |
| No job retry | Failed sims lost | RQ retry policies |
| No prioritization | All jobs equal | Priority queues |
| No backpressure | Memory issues under load | Queue depth limits |

## Implementation Plan

### 1. Add Redis Queue (RQ) Dependency

```bash
uv add rq redis
```

### 2. Create Job Queue Module

**File:** `src/server/job_queue.py`

```python
from rq import Queue, Retry
from redis import Redis
from typing import Optional

# Connection to Redis
_redis_conn: Optional[Redis] = None
_sim_queue: Optional[Queue] = None

def init_job_queue(redis_url: str = "redis://localhost:6379"):
    """Initialize Redis connection and job queue."""
    global _redis_conn, _sim_queue
    _redis_conn = Redis.from_url(redis_url)
    _sim_queue = Queue('simulations', connection=_redis_conn)

def enqueue_simulation(
    run_id: str,
    scenario_path: str,
    priority: str = "normal"
) -> str:
    """
    Enqueue a simulation job.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML
        priority: "high", "normal", or "low"

    Returns:
        Job ID for tracking
    """
    from server.app import _run_simulation_sync

    queue = _sim_queue
    if priority == "high":
        queue = Queue('simulations-high', connection=_redis_conn)
    elif priority == "low":
        queue = Queue('simulations-low', connection=_redis_conn)

    job = queue.enqueue(
        _run_simulation_sync,
        args=(run_id, scenario_path),
        job_id=run_id,
        job_timeout=3600,  # 1 hour max
        retry=Retry(max=3, interval=[10, 30, 60]),
        meta={'scenario_path': scenario_path}
    )
    return job.id

def get_job_status(job_id: str) -> dict:
    """Get status of a queued job."""
    from rq.job import Job
    job = Job.fetch(job_id, connection=_redis_conn)
    return {
        "id": job.id,
        "status": job.get_status(),
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "result": job.result,
        "error": job.exc_info
    }

def get_queue_stats() -> dict:
    """Get queue statistics."""
    return {
        "queued": len(_sim_queue),
        "failed": len(_sim_queue.failed_job_registry),
        "started": len(_sim_queue.started_job_registry),
        "finished": len(_sim_queue.finished_job_registry),
    }
```

### 3. Update App to Use Job Queue

**File:** `src/server/app.py`

```python
import os
from server.job_queue import init_job_queue, enqueue_simulation

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Redis is required
    _initialize_database()
    init_job_queue(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    yield

@app.post("/api/simulations", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    # ... validation ...

    job_id = enqueue_simulation(run_id, str(scenario_path), priority)
    return StartSimulationResponse(
        run_id=run_id,
        status=SimulationStatus.PENDING,
        message=f"Simulation queued: {job_id}"
    )
```

### 4. Add Worker Script

**File:** `run_worker.py`

```python
#!/usr/bin/env python
"""Run RQ worker for processing simulation jobs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from redis import Redis
from rq import Worker, Queue

if __name__ == "__main__":
    redis_conn = Redis.from_url("redis://localhost:6379")

    # Listen to multiple queues with priority
    queues = [
        Queue('simulations-high', connection=redis_conn),
        Queue('simulations', connection=redis_conn),
        Queue('simulations-low', connection=redis_conn),
    ]

    worker = Worker(queues, connection=redis_conn)
    worker.work()
```

### 5. Add Job Status Endpoint

```python
@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a queued job."""
    if not USE_JOB_QUEUE:
        raise HTTPException(501, "Job queue not enabled")

    from server.job_queue import get_job_status
    try:
        return get_job_status(job_id)
    except Exception:
        raise HTTPException(404, f"Job {job_id} not found")

@app.get("/api/queue/stats")
async def queue_stats():
    """Get job queue statistics."""
    if not USE_JOB_QUEUE:
        raise HTTPException(501, "Job queue not enabled")

    from server.job_queue import get_queue_stats
    return get_queue_stats()
```

### 6. Docker Compose for Development

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APART_USE_DATABASE=1
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  worker:
    build: .
    command: python run_worker.py
    environment:
      - APART_USE_DATABASE=1
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 2  # Run 2 workers

volumes:
  redis_data:
```

## Testing Plan

1. **Unit tests** for job_queue.py
2. **Integration tests** with Redis (use `fakeredis` for CI)
3. **End-to-end test**: enqueue → worker picks up → completes

```python
# tests/integration/test_job_queue.py

@pytest.fixture
def mock_redis():
    import fakeredis
    return fakeredis.FakeRedis()

def test_enqueue_simulation(mock_redis, monkeypatch):
    from server import job_queue
    monkeypatch.setattr(job_queue, "_redis_conn", mock_redis)
    # ... test enqueue logic
```

## Deployment Checklist

- [ ] Redis instance provisioned
- [ ] `REDIS_URL` configured (defaults to `redis://localhost:6379`)
- [ ] Worker processes running (1 per CPU core recommended)
- [ ] Monitor queue depth (alert if > 100 pending)
- [ ] Set up Redis persistence (AOF recommended)

## Requirements

Redis is required for the server to start. For local development:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```
