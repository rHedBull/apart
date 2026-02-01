"""Unit tests for EventBus core functionality and convenience functions.

Complements test_event_bus_extended.py by covering:
- Singleton pattern
- Async subscribe() iterator with filtering and history
- Callback management (add/remove)
- Convenience functions (emit_event, get_event_bus)
- SimulationEvent creation and serialization
- Database persistence mode
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest


class TestSimulationEvent:
    """Tests for SimulationEvent dataclass."""

    def test_create_with_timestamp(self):
        """Test create() factory generates current timestamp."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create("test_type", run_id="run-123", step=5)

        assert event.event_type == "test_type"
        assert event.run_id == "run-123"
        assert event.step == 5
        assert event.timestamp is not None
        # Timestamp should be ISO format
        assert "T" in event.timestamp

    def test_create_with_data_kwargs(self):
        """Test create() passes extra kwargs to data dict."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create(
            "agent_action",
            run_id="run-456",
            agent_name="Agent A",
            action="cooperate",
            score=100
        )

        assert event.data["agent_name"] == "Agent A"
        assert event.data["action"] == "cooperate"
        assert event.data["score"] == 100

    def test_create_without_step(self):
        """Test create() with no step parameter."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create("simple_event", run_id="run-789")

        assert event.step is None

    def test_to_dict_all_fields(self):
        """Test to_dict() returns all fields."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent(
            event_type="test",
            timestamp="2024-01-15T10:30:00",
            run_id="dict-test",
            step=3,
            data={"key": "value"}
        )

        d = event.to_dict()

        assert d == {
            "event_type": "test",
            "timestamp": "2024-01-15T10:30:00",
            "run_id": "dict-test",
            "step": 3,
            "data": {"key": "value"}
        }

    def test_to_dict_with_none_step(self):
        """Test to_dict() handles None step."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent(
            event_type="test",
            timestamp="2024-01-15T10:30:00",
            run_id="none-step",
            step=None,
            data={}
        )

        d = event.to_dict()
        assert d["step"] is None

    def test_to_json_produces_valid_json(self):
        """Test to_json() produces parseable JSON."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create("json_test", run_id="json-run", step=1, nested={"a": [1, 2, 3]})

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == "json_test"
        assert parsed["data"]["nested"] == {"a": [1, 2, 3]}


class TestEventBusSingleton:
    """Tests for EventBus singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_get_instance_returns_same_object(self, tmp_path):
        """Test get_instance() always returns the same singleton."""
        from server.event_bus import EventBus

        EventBus._test_persist_path = tmp_path / "events.jsonl"

        instance1 = EventBus.get_instance()
        instance2 = EventBus.get_instance()
        instance3 = EventBus()

        assert instance1 is instance2
        assert instance1 is instance3

    def test_reset_instance_clears_singleton(self, tmp_path):
        """Test reset_instance() allows new instance creation."""
        from server.event_bus import EventBus

        EventBus._test_persist_path = tmp_path / "events.jsonl"

        instance1 = EventBus.get_instance()
        EventBus.reset_instance()

        EventBus._test_persist_path = tmp_path / "events2.jsonl"
        instance2 = EventBus.get_instance()

        assert instance1 is not instance2

    def test_init_only_runs_once(self, tmp_path):
        """Test __init__ body only runs once per singleton."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"

        bus1 = EventBus.get_instance()
        bus1.emit(SimulationEvent.create("test", run_id="init-test"))

        # Re-calling EventBus() should not reinitialize
        bus2 = EventBus()

        # History should still be present
        assert len(bus2.get_history("init-test")) == 1


class TestEventBusCallbacks:
    """Tests for callback management."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_add_callback_receives_events(self, tmp_path):
        """Test add_callback() registers callback that receives events."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        received = []
        def callback(event):
            received.append(event)

        bus.add_callback(callback)
        bus.emit(SimulationEvent.create("cb_test", run_id="callback-run"))

        assert len(received) == 1
        assert received[0].event_type == "cb_test"

    def test_add_multiple_callbacks(self, tmp_path):
        """Test multiple callbacks all receive events."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        results1 = []
        results2 = []
        results3 = []

        bus.add_callback(lambda e: results1.append(e.event_type))
        bus.add_callback(lambda e: results2.append(e.run_id))
        bus.add_callback(lambda e: results3.append(e.step))

        bus.emit(SimulationEvent.create("multi_cb", run_id="multi-run", step=7))

        assert results1 == ["multi_cb"]
        assert results2 == ["multi-run"]
        assert results3 == [7]

    def test_remove_callback_stops_receiving(self, tmp_path):
        """Test remove_callback() stops callback from receiving events."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        received = []
        def callback(event):
            received.append(event.event_type)

        bus.add_callback(callback)
        bus.emit(SimulationEvent.create("before_remove", run_id="remove-test"))

        bus.remove_callback(callback)
        bus.emit(SimulationEvent.create("after_remove", run_id="remove-test"))

        assert received == ["before_remove"]

    def test_remove_nonexistent_callback_no_error(self, tmp_path):
        """Test remove_callback() on non-registered callback doesn't error."""
        from server.event_bus import EventBus

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        def not_registered(event):
            pass

        # Should not raise
        bus.remove_callback(not_registered)


class TestEventBusSubscribe:
    """Tests for async subscribe() functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    @pytest.mark.asyncio
    async def test_subscribe_receives_emitted_events(self, tmp_path):
        """Test subscribe() yields events as they are emitted."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        received = []

        async def collect_events():
            async for event in bus.subscribe():
                received.append(event)
                if len(received) >= 3:
                    break

        # Start subscriber in background
        task = asyncio.create_task(collect_events())

        # Give subscriber time to register
        await asyncio.sleep(0.01)

        # Emit events
        for i in range(3):
            bus.emit(SimulationEvent.create(f"event_{i}", run_id="sub-test"))

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 3
        assert [e.event_type for e in received] == ["event_0", "event_1", "event_2"]

    @pytest.mark.asyncio
    async def test_subscribe_filter_by_run_id(self, tmp_path):
        """Test subscribe() filters events by run_id."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        received = []

        async def collect_events():
            async for event in bus.subscribe(run_id="target-run"):
                received.append(event)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.01)

        # Emit mix of run_ids
        bus.emit(SimulationEvent.create("other_1", run_id="other-run"))
        bus.emit(SimulationEvent.create("target_1", run_id="target-run"))
        bus.emit(SimulationEvent.create("other_2", run_id="other-run"))
        bus.emit(SimulationEvent.create("target_2", run_id="target-run"))

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 2
        assert all(e.run_id == "target-run" for e in received)

    @pytest.mark.asyncio
    async def test_subscribe_include_history(self, tmp_path):
        """Test subscribe() yields history first when include_history=True."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        # Pre-emit historical events
        bus.emit(SimulationEvent.create("historical_1", run_id="history-run", step=1))
        bus.emit(SimulationEvent.create("historical_2", run_id="history-run", step=2))

        received = []

        async def collect_events():
            async for event in bus.subscribe(run_id="history-run", include_history=True):
                received.append(event)
                if len(received) >= 3:
                    break

        task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.01)

        # Emit new event after subscription started
        bus.emit(SimulationEvent.create("live_event", run_id="history-run", step=3))

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 3
        assert received[0].event_type == "historical_1"
        assert received[1].event_type == "historical_2"
        assert received[2].event_type == "live_event"

    @pytest.mark.asyncio
    async def test_subscribe_include_all_history_sorted(self, tmp_path):
        """Test subscribe() with include_history=True and no run_id sorts all events."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        # Emit events for multiple runs
        bus.emit(SimulationEvent.create("run1_e1", run_id="run1"))
        await asyncio.sleep(0.001)  # Ensure different timestamps
        bus.emit(SimulationEvent.create("run2_e1", run_id="run2"))
        await asyncio.sleep(0.001)
        bus.emit(SimulationEvent.create("run1_e2", run_id="run1"))

        received = []

        async def collect_events():
            async for event in bus.subscribe(include_history=True):
                received.append(event)
                if len(received) >= 3:
                    break

        # Need a live event to finish collecting since we break at 3
        task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.01)

        # Cancel task after history is yielded
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have received all historical events in timestamp order
        assert len(received) == 3
        assert received[0].event_type == "run1_e1"
        assert received[1].event_type == "run2_e1"
        assert received[2].event_type == "run1_e2"

    @pytest.mark.asyncio
    async def test_subscribe_cleanup_on_cancellation(self, tmp_path):
        """Test that subscriber is removed when iterator is cancelled."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        initial_subscribers = len(bus._subscribers)

        async def long_subscription():
            async for event in bus.subscribe():
                pass  # Would run forever

        # Start subscription
        task = asyncio.create_task(long_subscription())
        await asyncio.sleep(0.01)

        # Subscriber should be registered
        assert len(bus._subscribers) == initial_subscribers + 1

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Give cleanup time to run
        await asyncio.sleep(0.01)

        # Subscriber should be cleaned up via finally block
        assert len(bus._subscribers) == initial_subscribers


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_emit_event_creates_and_emits(self, tmp_path):
        """Test emit_event() convenience function."""
        from server.event_bus import EventBus, emit_event

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        emit_event("convenience_test", run_id="emit-func-test", step=10, extra="data")

        history = bus.get_history("emit-func-test")
        assert len(history) == 1
        assert history[0].event_type == "convenience_test"
        assert history[0].step == 10
        assert history[0].data["extra"] == "data"

    def test_get_event_bus_returns_singleton(self, tmp_path):
        """Test get_event_bus() returns the singleton instance."""
        from server.event_bus import EventBus, get_event_bus

        EventBus._test_persist_path = tmp_path / "events.jsonl"

        bus1 = get_event_bus()
        bus2 = EventBus.get_instance()

        assert bus1 is bus2


class TestEventBusDatabaseMode:
    """Tests for database persistence mode."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()
        EventBus._use_database = False

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus._use_database = False
        EventBus.reset_instance()

    def test_database_mode_persist_calls_db(self, tmp_path):
        """Test that database mode calls database for persistence."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        EventBus._use_database = True

        mock_db = MagicMock()
        mock_db.get_all_run_ids.return_value = []

        with patch.dict("sys.modules", {"server.database": MagicMock(get_db=lambda: mock_db)}):
            bus = EventBus.get_instance()
            event = SimulationEvent.create("db_test", run_id="db-run")
            bus.emit(event)

            mock_db.insert_event.assert_called_once_with(event)

    def test_database_mode_load_history_from_db(self, tmp_path):
        """Test that database mode loads history from database."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        EventBus._use_database = True

        mock_event = SimulationEvent(
            event_type="loaded",
            timestamp="2024-01-01T00:00:00",
            run_id="db-load-test",
            step=1,
            data={}
        )
        mock_db = MagicMock()
        mock_db.get_all_run_ids.return_value = ["db-load-test"]
        mock_db.get_events.return_value = [mock_event]

        with patch.dict("sys.modules", {"server.database": MagicMock(get_db=lambda: mock_db)}):
            bus = EventBus.get_instance()

            history = bus.get_history("db-load-test")
            assert len(history) == 1
            assert history[0].event_type == "loaded"

    def test_database_mode_handles_db_errors(self, tmp_path):
        """Test that database errors are handled gracefully."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        EventBus._use_database = True

        mock_module = MagicMock()
        mock_module.get_db.side_effect = Exception("Database error")

        with patch.dict("sys.modules", {"server.database": mock_module}):
            # Should not raise during initialization
            bus = EventBus.get_instance()

            # Should not raise during emit
            bus.emit(SimulationEvent.create("error_test", run_id="error-run"))

            # Event should still be in memory
            assert len(bus.get_history("error-run")) == 1


class TestEventBusGetAllRunIds:
    """Tests for get_all_run_ids functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        from server.event_bus import EventBus
        EventBus.reset_instance()

    def teardown_method(self):
        """Cleanup after each test."""
        from server.event_bus import EventBus
        EventBus._test_persist_path = None
        EventBus.reset_instance()

    def test_get_all_run_ids_empty(self, tmp_path):
        """Test get_all_run_ids() returns empty list when no events."""
        from server.event_bus import EventBus

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        assert bus.get_all_run_ids() == []

    def test_get_all_run_ids_multiple_runs(self, tmp_path):
        """Test get_all_run_ids() returns all unique run IDs."""
        from server.event_bus import EventBus, SimulationEvent

        EventBus._test_persist_path = tmp_path / "events.jsonl"
        bus = EventBus.get_instance()

        bus.emit(SimulationEvent.create("e1", run_id="run-a"))
        bus.emit(SimulationEvent.create("e2", run_id="run-b"))
        bus.emit(SimulationEvent.create("e3", run_id="run-a"))  # Duplicate
        bus.emit(SimulationEvent.create("e4", run_id="run-c"))

        run_ids = bus.get_all_run_ids()
        assert set(run_ids) == {"run-a", "run-b", "run-c"}
