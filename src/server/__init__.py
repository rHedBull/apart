"""
Real-time dashboard server for Apart simulations.

This package provides a FastAPI server with:
- REST API for simulation management
- SSE (Server-Sent Events) for real-time updates
- Event bus for simulation event distribution
"""

from server.event_bus import EventBus, SimulationEvent

__all__ = ["EventBus", "SimulationEvent"]
