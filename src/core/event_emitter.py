"""
Event emitter for simulation events.

This module provides a way to emit events from the simulation core
to the dashboard server without tight coupling. Events are only emitted
if the server module is available.
"""

import logging

logger = logging.getLogger(__name__)

# Global state
_emitter_enabled = False
_run_id: str | None = None


def enable_event_emitter(run_id: str) -> None:
    """
    Enable event emission for a simulation run.

    Args:
        run_id: The unique ID for this simulation run
    """
    global _emitter_enabled, _run_id
    _emitter_enabled = True
    _run_id = run_id


def disable_event_emitter() -> None:
    """Disable event emission."""
    global _emitter_enabled, _run_id
    _emitter_enabled = False
    _run_id = None


def emit(
    event_type: str,
    step: int | None = None,
    **data
) -> None:
    """
    Emit a simulation event.

    This is a no-op if the event emitter is not enabled or
    the server module is not available.

    Args:
        event_type: Type of event (e.g., 'simulation_started', 'agent_response')
        step: Current simulation step (optional)
        **data: Event-specific data
    """
    if not _emitter_enabled or _run_id is None:
        return

    try:
        from server.event_bus import emit_event
        emit_event(event_type, _run_id, step, **data)
    except ImportError:
        # Server module not available, silently skip
        pass
    except Exception as e:
        # Don't let event emission errors affect simulation
        logger.debug(f"Event emission failed: {e}")


# Event type constants for consistency
class EventTypes:
    """Standard event types emitted by the simulation."""

    # Simulation lifecycle
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_COMPLETED = "simulation_completed"
    SIMULATION_FAILED = "simulation_failed"

    # Step events
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"

    # Agent events
    AGENT_MESSAGE_SENT = "agent_message_sent"
    AGENT_RESPONSE_RECEIVED = "agent_response_received"

    # Danger detection
    DANGER_SIGNAL = "danger_signal"
