"""Unit tests for the dashboard server components."""

import sys
from pathlib import Path
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest


class TestEventBus:
    """Tests for EventBus singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def test_singleton(self):
        """Test that EventBus is a singleton."""
        from server.event_bus import EventBus

        bus1 = EventBus.get_instance()
        bus2 = EventBus.get_instance()

        assert bus1 is bus2

    def test_emit_and_history(self):
        """Test emitting events and retrieving history."""
        from server.event_bus import EventBus, SimulationEvent

        bus = EventBus.get_instance()

        event = SimulationEvent.create(
            "test_event",
            run_id="test123",
            step=1,
            data="test_data"
        )
        bus.emit(event)

        history = bus.get_history("test123")
        assert len(history) == 1
        assert history[0].event_type == "test_event"
        assert history[0].run_id == "test123"
        assert history[0].step == 1

    def test_emit_multiple_runs(self):
        """Test events are stored per run."""
        from server.event_bus import EventBus, SimulationEvent

        bus = EventBus.get_instance()

        # Emit events for different runs
        bus.emit(SimulationEvent.create("event1", run_id="run_a", step=1))
        bus.emit(SimulationEvent.create("event2", run_id="run_b", step=1))
        bus.emit(SimulationEvent.create("event3", run_id="run_a", step=2))

        # Check histories
        history_a = bus.get_history("run_a")
        history_b = bus.get_history("run_b")

        assert len(history_a) == 2
        assert len(history_b) == 1
        assert history_a[0].event_type == "event1"
        assert history_a[1].event_type == "event3"

    def test_get_all_run_ids(self):
        """Test getting all run IDs."""
        from server.event_bus import EventBus, SimulationEvent

        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("event1", run_id="run_x"))
        bus.emit(SimulationEvent.create("event2", run_id="run_y"))

        run_ids = bus.get_all_run_ids()
        assert "run_x" in run_ids
        assert "run_y" in run_ids

    def test_clear_history(self):
        """Test clearing event history."""
        from server.event_bus import EventBus, SimulationEvent

        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("event1", run_id="run1"))
        bus.emit(SimulationEvent.create("event2", run_id="run2"))

        # Clear specific run
        bus.clear_history("run1")
        assert len(bus.get_history("run1")) == 0
        assert len(bus.get_history("run2")) == 1

        # Clear all
        bus.clear_history()
        assert len(bus.get_history("run2")) == 0

    def test_callback(self):
        """Test synchronous callbacks."""
        from server.event_bus import EventBus, SimulationEvent

        bus = EventBus.get_instance()
        received_events = []

        def callback(event):
            received_events.append(event)

        bus.add_callback(callback)

        bus.emit(SimulationEvent.create("test", run_id="run1"))
        bus.emit(SimulationEvent.create("test2", run_id="run1"))

        assert len(received_events) == 2
        assert received_events[0].event_type == "test"

        # Remove callback
        bus.remove_callback(callback)
        bus.emit(SimulationEvent.create("test3", run_id="run1"))
        assert len(received_events) == 2  # No new events


class TestSimulationEvent:
    """Tests for SimulationEvent dataclass."""

    def test_create_event(self):
        """Test event creation with factory method."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create(
            "agent_response",
            run_id="abc123",
            step=5,
            agent_name="Agent A",
            response="Hello"
        )

        assert event.event_type == "agent_response"
        assert event.run_id == "abc123"
        assert event.step == 5
        assert event.data["agent_name"] == "Agent A"
        assert event.data["response"] == "Hello"
        assert event.timestamp is not None

    def test_to_dict(self):
        """Test converting event to dictionary."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create("test", run_id="run1", step=1, key="value")
        d = event.to_dict()

        assert d["event_type"] == "test"
        assert d["run_id"] == "run1"
        assert d["step"] == 1
        assert d["data"]["key"] == "value"

    def test_to_sse(self):
        """Test SSE format output."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create("test", run_id="run1")
        sse = event.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        assert '"event_type": "test"' in sse


class TestEmitEvent:
    """Tests for the emit_event convenience function."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def test_emit_event_function(self):
        """Test the convenience emit_event function."""
        from server.event_bus import emit_event, EventBus

        emit_event("test_event", run_id="run123", step=1, custom_data="hello")

        bus = EventBus.get_instance()
        history = bus.get_history("run123")

        assert len(history) == 1
        assert history[0].event_type == "test_event"
        assert history[0].data["custom_data"] == "hello"


class TestEventEmitter:
    """Tests for the core event emitter module."""

    def test_emitter_disabled_by_default(self):
        """Test that emitter doesn't emit when disabled."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

        from core.event_emitter import emit, disable_event_emitter

        # Ensure disabled
        disable_event_emitter()

        # This should be a no-op
        emit("test_event", step=1, data="test")

        # No events should be recorded
        bus = EventBus.get_instance()
        assert len(bus.get_all_run_ids()) == 0

    def test_emitter_when_enabled(self):
        """Test that emitter works when enabled."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

        from core.event_emitter import emit, enable_event_emitter, disable_event_emitter

        enable_event_emitter("test_run_456")
        emit("test_event", step=1, data="test")

        bus = EventBus.get_instance()
        history = bus.get_history("test_run_456")

        assert len(history) == 1
        assert history[0].event_type == "test_event"

        # Cleanup
        disable_event_emitter()


class TestGetRunStatus:
    """Tests for the _get_run_status function."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def test_get_run_status_paused(self):
        """Test that _get_run_status returns paused for simulation_paused event."""
        from server.event_bus import EventBus, SimulationEvent
        from server.routes.v1 import _get_run_status

        bus = EventBus.get_instance()

        # Emit started then paused events
        bus.emit(SimulationEvent.create("simulation_started", run_id="test_paused"))
        bus.emit(SimulationEvent.create("simulation_paused", run_id="test_paused", step=5))

        status = _get_run_status("test_paused")
        assert status == "paused"

    def test_get_run_status_resumed(self):
        """Test that _get_run_status returns running after simulation_resumed event."""
        from server.event_bus import EventBus, SimulationEvent
        from server.routes.v1 import _get_run_status

        bus = EventBus.get_instance()

        # Emit started -> paused -> resumed events
        bus.emit(SimulationEvent.create("simulation_started", run_id="test_resumed"))
        bus.emit(SimulationEvent.create("simulation_paused", run_id="test_resumed", step=5))
        bus.emit(SimulationEvent.create("simulation_resumed", run_id="test_resumed", step=5))

        status = _get_run_status("test_resumed")
        assert status == "running"
