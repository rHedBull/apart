"""
FastAPI application for the Apart Dashboard.

Provides:
- REST API for simulation management
- SSE event streaming for real-time updates
- Redis job queue for distributed simulation processing
- Background stale run detection
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from server.event_bus import get_event_bus
from server.routes.v1 import router as v1_router
from utils.ops_logger import get_ops_logger

logger = get_ops_logger("api")


def _initialize_database():
    """Initialize database if database mode is enabled."""
    if os.environ.get("APART_USE_DATABASE", "").lower() in ("1", "true", "yes"):
        from server.database import init_db
        from server.event_bus import EventBus
        EventBus._use_database = True
        init_db()


def _generate_mock_data_if_empty():
    """Generate mock data if results/ directory is empty.

    This provides sample data for development/demo purposes.
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Check if there are any run directories
    existing_runs = [
        d for d in results_dir.iterdir()
        if d.is_dir() and (d.name.startswith("run_") or d.name.startswith("mock_run_"))
    ]

    if existing_runs:
        return  # Already have data

    print("No simulation runs found. Generating mock data for development...")

    try:
        import subprocess
        import sys

        script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_mock_data.py"
        if script_path.exists():
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            if result.returncode == 0:
                print("Mock data generated successfully.")
            else:
                print(f"Failed to generate mock data: {result.stderr}")
        else:
            print(f"Mock data script not found at {script_path}")
    except Exception as e:
        print(f"Error generating mock data: {e}")


def _initialize_job_queue():
    """Initialize Redis job queue (optional in dev mode)."""
    if os.environ.get("SKIP_REDIS", "").lower() in ("1", "true", "yes"):
        logger.info("Skipping Redis job queue (SKIP_REDIS=1)")
        return
    from server.job_queue import init_job_queue
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    init_job_queue(redis_url)


def _initialize_state_manager():
    """Initialize the RunStateManager with Redis backend."""
    if os.environ.get("SKIP_REDIS", "").lower() in ("1", "true", "yes"):
        logger.info("Skipping RunStateManager (SKIP_REDIS=1)")
        return
    from server.job_queue import get_redis_connection
    from server.run_state import RunStateManager
    try:
        redis_conn = get_redis_connection()
        RunStateManager.initialize(redis_conn)
    except RuntimeError:
        logger.warning("Could not initialize RunStateManager (job queue not initialized)")


def _initialize_event_bus_redis():
    """Initialize EventBus with Redis connection for cross-process events."""
    if os.environ.get("SKIP_REDIS", "").lower() in ("1", "true", "yes"):
        logger.info("Skipping EventBus Redis (SKIP_REDIS=1)")
        return
    from server.job_queue import get_redis_connection
    try:
        redis_conn = get_redis_connection()
        event_bus = get_event_bus()
        event_bus.set_redis_connection(redis_conn)
        logger.info("EventBus Redis connection initialized")
    except RuntimeError:
        logger.warning("Could not initialize EventBus Redis (job queue not initialized)")


async def _start_event_bus_subscriber():
    """Start the EventBus Redis subscriber for real-time cross-process events."""
    if os.environ.get("SKIP_REDIS", "").lower() in ("1", "true", "yes"):
        return
    try:
        event_bus = get_event_bus()
        await event_bus.start_redis_subscriber()
        logger.info("EventBus Redis subscriber started")
    except Exception as e:
        logger.warning(f"Could not start EventBus Redis subscriber: {e}")


async def _stale_run_checker(interval_seconds: int = 30):
    """Background task that detects and marks stale runs as interrupted.

    A run is considered stale if:
    - Status is "running"
    - Worker heartbeat has expired (no heartbeat for 30+ seconds)

    This handles the case where a worker crashes without gracefully
    transitioning the run to a terminal state.
    """
    from server.run_state import get_state_manager

    logger.info("Stale run checker started", extra={"interval": interval_seconds})

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            state_manager = get_state_manager()
            if state_manager is None:
                continue

            stale_run_ids = state_manager.check_stale_runs()

            for run_id in stale_run_ids:
                state = state_manager.mark_interrupted(
                    run_id,
                    reason="Worker heartbeat expired"
                )
                if state:
                    logger.warning("Marked stale run as interrupted", extra={
                        "run_id": run_id,
                        "previous_worker": state.worker_id,
                    })

        except asyncio.CancelledError:
            logger.info("Stale run checker stopped")
            raise
        except Exception:
            logger.exception("Error in stale run checker")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Skip logging for health checks and SSE streams (too noisy)
        path = request.url.path
        if path not in ("/api/health", "/api/events/stream") and not path.startswith("/api/events/stream/"):
            logger.info("Request", extra={
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            })

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("API server starting")
    _initialize_database()
    _initialize_job_queue()
    _initialize_state_manager()
    _initialize_event_bus_redis()
    _generate_mock_data_if_empty()

    # Start background tasks
    stale_checker_task = None
    if os.environ.get("SKIP_REDIS", "").lower() not in ("1", "true", "yes"):
        stale_checker_task = asyncio.create_task(_stale_run_checker(interval_seconds=30))
        # Start EventBus Redis subscriber for real-time cross-process events
        await _start_event_bus_subscriber()

    logger.info("API server ready")
    yield

    # Shutdown - cancel background tasks
    if stale_checker_task:
        stale_checker_task.cancel()
        try:
            await stale_checker_task
        except asyncio.CancelledError:
            pass

    # Stop EventBus Redis subscriber
    event_bus = get_event_bus()
    await event_bus.stop_redis_subscriber()

    logger.info("API server shutting down")


app = FastAPI(
    title="Apart Dashboard API",
    description="API for monitoring AI safety simulations",
    version="0.1.0",
    docs_url="/api/docs",
    lifespan=lifespan,
)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount v1 API router
app.include_router(v1_router, prefix="/api/v1")


def _get_version() -> str:
    """Get package version from pyproject.toml."""
    try:
        from importlib.metadata import version
        return version("apart")
    except Exception:
        return "unknown"


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": _get_version()}


@app.get("/api/health/detailed")
async def detailed_health():
    """Detailed health check with system metrics."""
    from server.event_bus import EventBus
    from server.job_queue import get_queue_stats
    from server.run_state import get_state_manager

    event_bus = get_event_bus()
    state_manager = get_state_manager()

    result = {
        "status": "healthy",
        "version": _get_version(),
        "event_bus_subscribers": len(event_bus._subscribers),
        "total_run_ids": state_manager.count_runs() if state_manager else 0,
        "persistence_mode": "database" if EventBus._use_database else "jsonl",
        "queue_stats": get_queue_stats(),
    }

    # Add database stats if using database mode
    if EventBus._use_database:
        try:
            from server.database import get_db
            db = get_db()
            result["database_stats"] = db.get_stats()
        except Exception:
            result["database_stats"] = {"error": "unavailable"}

    return result


@app.get("/api/events/stream")
async def event_stream(run_id: Optional[str] = None, history: bool = False):
    """SSE event stream for real-time updates."""
    from server.run_state import get_state_manager

    event_bus = get_event_bus()
    state_manager = get_state_manager()

    # Get valid run IDs from RunStateManager to filter stale history events
    valid_run_ids: set[str] | None = None
    if history and state_manager:
        valid_run_ids = {s.run_id for s in state_manager.list_runs(limit=1000)}

    async def generate():
        # Send connection event
        yield 'data: {"event_type": "connected", "message": "Connected to event stream"}\n\n'

        # Use the async iterator subscribe method
        async for event in event_bus.subscribe(
            run_id=run_id,
            include_history=history,
            history_run_ids=valid_run_ids,
        ):
            yield event.to_sse()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/events/stream/{run_id}")
async def event_stream_for_run(run_id: str, history: bool = False):
    """SSE event stream for a specific run."""
    return await event_stream(run_id=run_id, history=history)


# ============================================================================
# Job Queue Endpoints
# ============================================================================


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a queued job."""
    from server.job_queue import get_job_status

    try:
        return get_job_status(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.get("/api/queue/stats")
async def queue_stats():
    """Get job queue statistics."""
    from server.job_queue import get_queue_stats

    return get_queue_stats()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job. Only works for jobs that haven't started yet."""
    from server.job_queue import cancel_job as do_cancel

    cancelled = do_cancel(job_id)
    if cancelled:
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} cannot be cancelled (may be running or completed)"
        )


# ============================================================================
# Dashboard Static Files
# ============================================================================

# Find dashboard dist directory (relative to project root)
_dashboard_dist = Path(__file__).parent.parent.parent / "dashboard" / "dist"

if _dashboard_dist.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=_dashboard_dist / "assets"), name="assets")

    @app.get("/")
    async def dashboard_root():
        """Serve dashboard index.html."""
        return FileResponse(_dashboard_dist / "index.html")

    @app.get("/{path:path}")
    async def dashboard_spa(path: str):
        """Serve dashboard for SPA routing (catch-all for non-API routes)."""
        # Don't intercept API routes
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        # Serve index.html for SPA routing
        return FileResponse(_dashboard_dist / "index.html")


def main():
    """Entry point for the apart-server command."""
    import uvicorn
    host = os.environ.get("APART_HOST", "127.0.0.1")
    port = int(os.environ.get("APART_PORT", "8000"))
    print(f"Starting APART server v{_get_version()} on http://{host}:{port}")
    if _dashboard_dist.exists():
        print(f"Dashboard available at http://{host}:{port}/")
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=os.environ.get("APART_RELOAD", "").lower() in ("1", "true", "yes"),
    )
