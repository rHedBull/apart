"""
FastAPI application for the Apart Dashboard.

Provides:
- REST API for simulation management
- SSE event streaming for real-time updates
- Redis job queue for distributed simulation processing
"""

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
    _generate_mock_data_if_empty()
    logger.info("API server ready")
    yield
    # Shutdown
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/health/detailed")
async def detailed_health():
    """Detailed health check with system metrics."""
    from server.event_bus import EventBus
    from server.job_queue import get_queue_stats

    event_bus = get_event_bus()

    result = {
        "status": "healthy",
        "event_bus_subscribers": len(event_bus._subscribers),
        "total_run_ids": len(event_bus.get_all_run_ids()),
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
    event_bus = get_event_bus()

    async def generate():
        # Send connection event
        yield f"data: {{\"event_type\": \"connected\", \"message\": \"Connected to event stream\"}}\n\n"

        # Use the async iterator subscribe method
        async for event in event_bus.subscribe(run_id=run_id, include_history=history):
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
    print(f"Starting APART server v0.1.3 on http://{host}:{port}")
    if _dashboard_dist.exists():
        print(f"Dashboard available at http://{host}:{port}/")
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=os.environ.get("APART_RELOAD", "").lower() in ("1", "true", "yes"),
    )
