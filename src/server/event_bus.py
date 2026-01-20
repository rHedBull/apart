"""
Event bus for distributing simulation events to SSE clients.

The EventBus is a singleton that:
- Receives events from running simulations
- Broadcasts events to all connected SSE clients
- Maintains event history for late-joining clients
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable
from collections import defaultdict
import json


@dataclass
class SimulationEvent:
    """
    A simulation event to be broadcast to clients.

    Attributes:
        event_type: Type of event (simulation_started, step_started, agent_response, etc.)
        timestamp: ISO format timestamp
        run_id: Unique identifier for the simulation run
        step: Current simulation step (optional)
        data: Event-specific data
    """
    event_type: str
    timestamp: str
    run_id: str
    step: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "step": self.step,
            "data": self.data
        }

    def to_sse(self) -> str:
        """Format as SSE message."""
        return f"data: {json.dumps(self.to_dict())}\n\n"

    @classmethod
    def create(
        cls,
        event_type: str,
        run_id: str,
        step: int | None = None,
        **data
    ) -> "SimulationEvent":
        """Factory method to create an event with current timestamp."""
        return cls(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            step=step,
            data=data
        )


class EventBus:
    """
    Singleton event bus for simulation event distribution.

    Usage:
        # Get the singleton instance
        bus = EventBus.get_instance()

        # Emit an event (from simulation code)
        bus.emit(SimulationEvent.create(
            "agent_response",
            run_id="abc123",
            step=5,
            agent_name="Agent A",
            response="I choose to cooperate."
        ))

        # Subscribe to events (in SSE endpoint)
        async for event in bus.subscribe():
            yield event.to_sse()
    """

    _instance: "EventBus | None" = None
    _lock: asyncio.Lock | None = None

    def __new__(cls) -> "EventBus":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the event bus (only once)."""
        if self._initialized:
            return

        self._initialized = True
        self._subscribers: list[asyncio.Queue[SimulationEvent]] = []
        self._event_history: dict[str, list[SimulationEvent]] = defaultdict(list)
        self._max_history_per_run = 1000
        self._callbacks: list[Callable[[SimulationEvent], None]] = []

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def emit(self, event: SimulationEvent) -> None:
        """
        Emit an event to all subscribers.

        This is synchronous and can be called from non-async code.
        Events are queued for async subscribers.

        Args:
            event: The simulation event to broadcast
        """
        # Store in history
        history = self._event_history[event.run_id]
        history.append(event)
        if len(history) > self._max_history_per_run:
            history.pop(0)

        # Queue for async subscribers
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event if queue is full
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass

        # Call sync callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors break event flow

    async def subscribe(
        self,
        run_id: str | None = None,
        include_history: bool = False
    ) -> AsyncIterator[SimulationEvent]:
        """
        Subscribe to events as an async iterator.

        Args:
            run_id: Optional filter to only receive events for specific run
            include_history: If True, yield historical events first

        Yields:
            SimulationEvent objects as they are emitted
        """
        queue: asyncio.Queue[SimulationEvent] = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)

        try:
            # Optionally yield history first
            if include_history:
                if run_id:
                    for event in self._event_history.get(run_id, []):
                        yield event
                else:
                    # Yield all history, sorted by timestamp
                    all_events = []
                    for events in self._event_history.values():
                        all_events.extend(events)
                    all_events.sort(key=lambda e: e.timestamp)
                    for event in all_events:
                        yield event

            # Then yield new events
            while True:
                event = await queue.get()
                if run_id is None or event.run_id == run_id:
                    yield event

        finally:
            self._subscribers.remove(queue)

    def add_callback(self, callback: Callable[[SimulationEvent], None]) -> None:
        """
        Add a synchronous callback for events.

        Args:
            callback: Function to call for each event
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SimulationEvent], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_history(self, run_id: str) -> list[SimulationEvent]:
        """
        Get event history for a specific run.

        Args:
            run_id: The run ID to get history for

        Returns:
            List of events for the run
        """
        return list(self._event_history.get(run_id, []))

    def get_all_run_ids(self) -> list[str]:
        """Get all run IDs with events."""
        return list(self._event_history.keys())

    def clear_history(self, run_id: str | None = None) -> None:
        """
        Clear event history.

        Args:
            run_id: Specific run to clear, or None to clear all
        """
        if run_id:
            self._event_history.pop(run_id, None)
        else:
            self._event_history.clear()


# Convenience function for emitting events from simulation code
def emit_event(
    event_type: str,
    run_id: str,
    step: int | None = None,
    **data
) -> None:
    """
    Emit a simulation event.

    This is a convenience wrapper around EventBus.emit().

    Args:
        event_type: Type of event
        run_id: Simulation run ID
        step: Current step (optional)
        **data: Event-specific data
    """
    event = SimulationEvent.create(event_type, run_id, step, **data)
    EventBus.get_instance().emit(event)
