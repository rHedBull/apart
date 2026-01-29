"""Extended unit tests for EventBus covering persistence, reload, and edge cases."""

import json
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest


class TestEventBusPersistence:
    """Tests for EventBus JSONL file persistence."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_event_persisted_to_file(self, tmp_path):
        """Test that emitted events are persisted to JSONL file."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        event = SimulationEvent.create("test_event", run_id="persist-test", step=1)
        bus.emit(event)

        # Check file was created and contains event
        assert persist_file.exists()
        with open(persist_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["event_type"] == "test_event"
        assert data["run_id"] == "persist-test"

    def test_multiple_events_appended(self, tmp_path):
        """Test that multiple events are appended to the same file."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        for i in range(5):
            bus.emit(SimulationEvent.create(f"event_{i}", run_id="multi-test", step=i))

        with open(persist_file) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_persistence_creates_directory(self, tmp_path):
        """Test that persistence creates parent directories if needed."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "deep" / "nested" / "dir" / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("test", run_id="dir-test"))

        assert persist_file.exists()

    def test_load_history_from_existing_file(self, tmp_path):
        """Test that EventBus loads existing events from file on startup."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"

        # Pre-create file with events
        events_data = [
            {"event_type": "preexisting_1", "timestamp": "2024-01-01T00:00:00", "run_id": "load-test", "step": 1, "data": {}},
            {"event_type": "preexisting_2", "timestamp": "2024-01-01T00:00:01", "run_id": "load-test", "step": 2, "data": {}},
        ]
        with open(persist_file, "w") as f:
            for event in events_data:
                f.write(json.dumps(event) + "\n")

        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        history = bus.get_history("load-test")
        assert len(history) == 2
        assert history[0].event_type == "preexisting_1"
        assert history[1].event_type == "preexisting_2"

    def test_skip_malformed_lines_on_load(self, tmp_path):
        """Test that malformed lines in file are skipped during load."""
        from server.event_bus import EventBus

        persist_file = tmp_path / "events.jsonl"

        # Mix valid and invalid lines
        with open(persist_file, "w") as f:
            f.write('{"event_type": "valid", "timestamp": "2024-01-01T00:00:00", "run_id": "test", "step": 1, "data": {}}\n')
            f.write('not valid json\n')
            f.write('{"missing": "required fields"}\n')
            f.write('{"event_type": "also_valid", "timestamp": "2024-01-01T00:00:01", "run_id": "test", "step": 2, "data": {}}\n')
            f.write('\n')  # Empty line

        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        history = bus.get_history("test")
        assert len(history) == 2

    def test_history_max_limit_enforced(self, tmp_path):
        """Test that history is limited to max_history_per_run events."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()
        bus._max_history_per_run = 5  # Set low limit for testing

        # Emit more events than limit
        for i in range(10):
            bus.emit(SimulationEvent.create(f"event_{i}", run_id="limit-test", step=i))

        history = bus.get_history("limit-test")
        assert len(history) == 5
        # Should have the most recent events
        assert history[0].event_type == "event_5"
        assert history[-1].event_type == "event_9"


class TestEventBusReload:
    """Tests for EventBus cross-process reload mechanism."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_reload_detects_external_file_changes(self, tmp_path):
        """Test that EventBus reloads when file is modified externally."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        # Emit initial event
        bus.emit(SimulationEvent.create("initial", run_id="reload-test"))
        assert len(bus.get_history("reload-test")) == 1

        # Simulate external process writing to file
        # Need to ensure different mtime
        time.sleep(0.01)  # Small delay to ensure different mtime
        with open(persist_file, "a") as f:
            external_event = {"event_type": "external", "timestamp": "2024-01-01T00:00:00", "run_id": "reload-test", "step": 2, "data": {}}
            f.write(json.dumps(external_event) + "\n")

        # Force reload check
        history = bus.get_history("reload-test")
        assert len(history) == 2
        assert history[-1].event_type == "external"

    def test_no_reload_when_file_unchanged(self, tmp_path):
        """Test that EventBus doesn't reload when file hasn't changed."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("test", run_id="no-reload-test"))

        # Track mtime
        initial_mtime = bus._last_file_mtime

        # Call get_history multiple times
        for _ in range(5):
            bus.get_history("no-reload-test")

        # mtime should stay the same (no unnecessary reloads)
        assert bus._last_file_mtime == initial_mtime

    def test_reload_nonexistent_file_handles_gracefully(self, tmp_path):
        """Test that reload handles missing file gracefully."""
        from server.event_bus import EventBus

        persist_file = tmp_path / "nonexistent.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        # Should not raise
        history = bus.get_history("any-run")
        assert history == []


class TestEventBusSubscribers:
    """Tests for EventBus async subscriber functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_subscriber_queue_overflow_handling(self, tmp_path):
        """Test that queue overflow drops oldest event and adds new one."""
        import asyncio
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        # Create a small queue and subscribe
        queue = asyncio.Queue(maxsize=2)
        bus._subscribers.append(queue)

        try:
            # Emit more events than queue can hold
            for i in range(5):
                bus.emit(SimulationEvent.create(f"event_{i}", run_id="overflow-test"))

            # Queue should have latest events (due to overflow handling)
            assert queue.qsize() == 2
        finally:
            bus._subscribers.remove(queue)

    def test_callback_exception_doesnt_break_event_flow(self, tmp_path):
        """Test that callback exceptions don't prevent other callbacks from running."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        results = []

        def failing_callback(event):
            raise ValueError("Intentional failure")

        def success_callback(event):
            results.append(event.event_type)

        bus.add_callback(failing_callback)
        bus.add_callback(success_callback)

        # Should not raise, and success_callback should still run
        bus.emit(SimulationEvent.create("test", run_id="callback-test"))

        assert len(results) == 1
        assert results[0] == "test"


class TestSimulationEventSerialization:
    """Tests for SimulationEvent JSON serialization."""

    def test_from_json_roundtrip(self):
        """Test that from_json correctly deserializes JSON."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create(
            "test_event",
            run_id="serial-test",
            step=5,
            custom_data="value",
            nested={"key": "val"}
        )

        json_str = event.to_json()
        restored = SimulationEvent.from_json(json_str)

        assert restored.event_type == event.event_type
        assert restored.run_id == event.run_id
        assert restored.step == event.step
        assert restored.data == event.data
        assert restored.timestamp == event.timestamp

    def test_from_json_missing_optional_fields(self):
        """Test from_json handles missing optional fields."""
        from server.event_bus import SimulationEvent

        json_str = '{"event_type": "test", "timestamp": "2024-01-01T00:00:00", "run_id": "test"}'
        event = SimulationEvent.from_json(json_str)

        assert event.step is None
        assert event.data == {}

    def test_to_sse_format(self):
        """Test SSE format is correct."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create("sse_test", run_id="test")
        sse = event.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        # Should be valid JSON after "data: " prefix
        json_part = sse[6:-2]  # Remove "data: " and "\n\n"
        data = json.loads(json_part)
        assert data["event_type"] == "sse_test"


class TestEventBusClearHistory:
    """Tests for clear_history functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_clear_history_with_persistence(self, tmp_path):
        """Test that clear_history can also clear persistence file."""
        from server.event_bus import EventBus, SimulationEvent

        persist_file = tmp_path / "events.jsonl"
        EventBus._test_persist_path = persist_file
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("test", run_id="clear-test"))
        assert persist_file.exists()

        bus.clear_history(clear_persistence=True)

        assert not persist_file.exists()
        assert len(bus.get_all_run_ids()) == 0

    def test_clear_specific_run_only(self, tmp_path):
        """Test that clearing specific run doesn't affect others."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("e1", run_id="run1"))
        bus.emit(SimulationEvent.create("e2", run_id="run2"))

        bus.clear_history("run1")

        assert len(bus.get_history("run1")) == 0
        assert len(bus.get_history("run2")) == 1


class TestEventBusSetPersistPath:
    """Tests for dynamic persist path changes."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_set_persist_path_to_null_disables_persistence(self, tmp_path):
        """Test that setting persist path to None disables persistence."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        bus.set_persist_path(None)
        bus.emit(SimulationEvent.create("test", run_id="null-test"))

        # Event should be in memory but not persisted
        assert len(bus.get_history("null-test")) == 1
        assert not (tmp_path / "events.jsonl").exists()

    def test_set_persist_path_changes_output_file(self, tmp_path):
        """Test that changing persist path writes to new location."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "original.jsonl"
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("e1", run_id="path-test"))

        new_path = tmp_path / "new.jsonl"
        bus.set_persist_path(new_path)
        bus.emit(SimulationEvent.create("e2", run_id="path-test"))

        assert new_path.exists()
        with open(new_path) as f:
            content = f.read()
        assert "e2" in content


class TestEventBusThreadSafety:
    """Tests for EventBus thread safety."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_concurrent_emit_thread_safe(self, tmp_path):
        """Test that concurrent emits from multiple threads are safe."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        num_threads = 10
        events_per_thread = 50
        errors = []

        def emit_events(thread_id):
            try:
                for i in range(events_per_thread):
                    bus.emit(SimulationEvent.create(
                        f"thread_{thread_id}_event_{i}",
                        run_id=f"thread-test-{thread_id}"
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_events, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all events were recorded
        total_events = sum(len(bus.get_history(f"thread-test-{i}")) for i in range(num_threads))
        assert total_events == num_threads * events_per_thread
