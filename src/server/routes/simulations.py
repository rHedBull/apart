"""
REST API routes for simulation management.

NOTE: This router is NOT currently mounted in the FastAPI app.
All simulation endpoints are defined directly in server/app.py.

This file is kept as a placeholder for potential future refactoring
where routes might be separated into their own modules.

Status tracking is handled entirely through EventBus events.
There is no separate in-memory registry - EventBus is the single
source of truth for simulation status.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/simulations", tags=["simulations"])

# Routes are defined in server/app.py
# This module exists for future refactoring purposes
