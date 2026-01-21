"""Integration tests for event persistence functionality."""

import pytest
import json
from pathlib import Path


class TestEventPersistence:
    """Tests for EventBus persistence to JSONL."""

    def test_events_written_to_file(self, persist_path):
        """Events should be written to the persistence file."""
        from server.event_bus import EventBus, SimulationEvent

        # Create bus with specific persist path
        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Emit an event
        event = SimulationEvent.create("test_event", run_id="persist-1", step=1)
        bus.emit(event)

        # Check file exists and has content
        assert persist_path.exists()

        with open(persist_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "test_event"
        assert data["run_id"] == "persist-1"

    def test_multiple_events_appended(self, persist_path):
        """Multiple events should be appended to the file."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Emit multiple events
        for i in range(5):
            event = SimulationEvent.create(f"event_{i}", run_id="multi-test", step=i)
            bus.emit(event)

        with open(persist_path) as f:
            lines = f.readlines()

        assert len(lines) == 5

        # Verify order
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["event_type"] == f"event_{i}"
            assert data["step"] == i

    def test_events_survive_reset(self, persist_path):
        """Events should be loaded after EventBus reset."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Emit events
        bus.emit(SimulationEvent.create("event_a", run_id="survive-test"))
        bus.emit(SimulationEvent.create("event_b", run_id="survive-test"))

        # Simulate "restart" by resetting
        EventBus.reset_instance()
        new_bus = EventBus.get_instance()

        # Events should be loaded from disk
        history = new_bus.get_history("survive-test")
        assert len(history) == 2
        assert history[0].event_type == "event_a"
        assert history[1].event_type == "event_b"

    def test_multiple_runs_persisted(self, persist_path):
        """Events for multiple runs should be persisted and restored."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Emit events for different runs
        bus.emit(SimulationEvent.create("start", run_id="run-a"))
        bus.emit(SimulationEvent.create("start", run_id="run-b"))
        bus.emit(SimulationEvent.create("step", run_id="run-a", step=1))
        bus.emit(SimulationEvent.create("step", run_id="run-b", step=1))

        # Reset and reload
        EventBus.reset_instance()
        new_bus = EventBus.get_instance()

        # Both runs should be restored
        run_ids = new_bus.get_all_run_ids()
        assert "run-a" in run_ids
        assert "run-b" in run_ids

        assert len(new_bus.get_history("run-a")) == 2
        assert len(new_bus.get_history("run-b")) == 2

    def test_clear_persistence(self, persist_path):
        """clear_history with clear_persistence should delete file."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("test", run_id="clear-test"))
        assert persist_path.exists()

        bus.clear_history(clear_persistence=True)
        assert not persist_path.exists()

    def test_malformed_lines_skipped(self, persist_path):
        """Malformed lines in persistence file should be skipped."""
        from server.event_bus import EventBus

        # Write some valid and invalid lines
        with open(persist_path, "w") as f:
            # Valid event
            f.write('{"event_type":"valid","timestamp":"2024-01-01T00:00:00","run_id":"test","step":1,"data":{}}\n')
            # Invalid JSON
            f.write('not valid json\n')
            # Missing required field
            f.write('{"event_type":"missing_run_id"}\n')
            # Another valid event
            f.write('{"event_type":"also_valid","timestamp":"2024-01-01T00:00:01","run_id":"test","step":2,"data":{}}\n')

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Should only load valid events
        history = bus.get_history("test")
        assert len(history) == 2
        assert history[0].event_type == "valid"
        assert history[1].event_type == "also_valid"


class TestEventSerialization:
    """Tests for SimulationEvent serialization."""

    def test_to_json_roundtrip(self):
        """Event should survive JSON roundtrip."""
        from server.event_bus import SimulationEvent

        original = SimulationEvent.create(
            "test_event",
            run_id="serial-test",
            step=5,
            custom_field="value",
            nested={"a": 1, "b": [1, 2, 3]}
        )

        json_str = original.to_json()
        restored = SimulationEvent.from_json(json_str)

        assert restored.event_type == original.event_type
        assert restored.run_id == original.run_id
        assert restored.step == original.step
        assert restored.timestamp == original.timestamp
        assert restored.data == original.data

    def test_from_json_missing_optional_fields(self):
        """from_json should handle missing optional fields."""
        from server.event_bus import SimulationEvent

        json_str = '{"event_type":"minimal","timestamp":"2024-01-01T00:00:00","run_id":"test"}'
        event = SimulationEvent.from_json(json_str)

        assert event.event_type == "minimal"
        assert event.run_id == "test"
        assert event.step is None
        assert event.data == {}


class TestPersistencePerformance:
    """Tests for persistence performance characteristics."""

    def test_many_events_persist_correctly(self, persist_path):
        """Large number of events should persist correctly."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Emit many events
        num_events = 100
        for i in range(num_events):
            bus.emit(SimulationEvent.create(f"event_{i}", run_id="perf-test", step=i))

        # Verify all persisted
        with open(persist_path) as f:
            lines = f.readlines()
        assert len(lines) == num_events

        # Verify all restored after reset
        EventBus.reset_instance()
        new_bus = EventBus.get_instance()
        history = new_bus.get_history("perf-test")
        assert len(history) == num_events

    def test_history_limit_respected(self, persist_path):
        """In-memory history limit should be respected on load."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = persist_path
        EventBus.reset_instance()
        bus = EventBus.get_instance()

        # Get the limit
        limit = bus._max_history_per_run

        # Emit more events than the limit
        for i in range(limit + 50):
            bus.emit(SimulationEvent.create(f"event_{i}", run_id="limit-test", step=i))

        # In-memory should be limited
        history = bus.get_history("limit-test")
        assert len(history) == limit

        # File should have all events
        with open(persist_path) as f:
            lines = f.readlines()
        assert len(lines) == limit + 50
