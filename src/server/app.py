"""
FastAPI application for the Apart dashboard server.

Run with:
    uv run python -m server.app

Or from project root:
    uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from server.routes import simulations, events

# Create FastAPI app
app = FastAPI(
    title="Apart Dashboard API",
    description="Real-time API for Apart AI safety simulation monitoring",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Vite dev server
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(simulations.router)
app.include_router(events.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Apart Dashboard API",
        "version": "0.1.0",
        "docs": "/api/docs",
        "endpoints": {
            "simulations": "/api/simulations",
            "events": "/api/events/stream"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the server."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
