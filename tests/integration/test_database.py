"""Integration tests for SQLite database functionality."""

import pytest
from pathlib import Path


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def test_db(db_path):
    """Create a test database instance."""
    from server.database import set_db_path, init_db, reset_db, get_db, close_connection

    set_db_path(db_path)
    init_db()

    yield get_db()

    # Cleanup
    close_connection()
    set_db_path(None)  # Reset to default


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_init_creates_tables(self, db_path):
        """init_db should create required tables."""
        from server.database import set_db_path, init_db, get_connection, close_connection

        set_db_path(db_path)
        init_db()

        with get_connection() as conn:
            # Check tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t["name"] for t in tables]

            assert "simulations" in table_names
            assert "events" in table_names

        close_connection()
        set_db_path(None)

    def test_init_is_idempotent(self, db_path):
        """init_db should be safe to call multiple times."""
        from server.database import set_db_path, init_db, close_connection

        set_db_path(db_path)

        # Call multiple times - should not raise
        init_db()
        init_db()
        init_db()

        close_connection()
        set_db_path(None)


class TestSimulationOperations:
    """Tests for simulation CRUD operations."""

    def test_insert_simulation(self, test_db):
        """Should insert a new simulation record."""
        test_db.insert_simulation(
            run_id="test-run-1",
            scenario_path="/path/to/scenario.yaml",
            scenario_name="test_scenario",
            max_steps=10
        )

        sim = test_db.get_simulation("test-run-1")
        assert sim is not None
        assert sim["run_id"] == "test-run-1"
        assert sim["status"] == "running"
        assert sim["scenario_path"] == "/path/to/scenario.yaml"
        assert sim["max_steps"] == 10

    def test_update_simulation_status(self, test_db):
        """Should update simulation status."""
        test_db.insert_simulation(run_id="status-test")

        test_db.update_simulation_status("status-test", "completed")

        sim = test_db.get_simulation("status-test")
        assert sim["status"] == "completed"
        assert sim["completed_at"] is not None

    def test_update_simulation_status_with_error(self, test_db):
        """Should store error message on failed status."""
        test_db.insert_simulation(run_id="error-test")

        test_db.update_simulation_status("error-test", "failed", "Something went wrong")

        sim = test_db.get_simulation("error-test")
        assert sim["status"] == "failed"
        assert sim["error_message"] == "Something went wrong"

    def test_update_simulation_step(self, test_db):
        """Should update current step."""
        test_db.insert_simulation(run_id="step-test")

        test_db.update_simulation_step("step-test", 5)

        sim = test_db.get_simulation("step-test")
        assert sim["current_step"] == 5

    def test_get_all_simulations(self, test_db):
        """Should return all simulations."""
        test_db.insert_simulation(run_id="sim-1")
        test_db.insert_simulation(run_id="sim-2")
        test_db.insert_simulation(run_id="sim-3")

        sims = test_db.get_all_simulations()
        assert len(sims) == 3

    def test_get_simulations_by_status(self, test_db):
        """Should filter simulations by status."""
        test_db.insert_simulation(run_id="running-1")
        test_db.insert_simulation(run_id="running-2")
        test_db.insert_simulation(run_id="completed-1")
        test_db.update_simulation_status("completed-1", "completed")

        running = test_db.get_all_simulations(status="running")
        completed = test_db.get_all_simulations(status="completed")

        assert len(running) == 2
        assert len(completed) == 1

    def test_delete_simulation(self, test_db):
        """Should delete simulation and its events."""
        from server.event_bus import SimulationEvent

        test_db.insert_simulation(run_id="delete-test")
        test_db.insert_event(SimulationEvent.create("test", run_id="delete-test"))

        test_db.delete_simulation("delete-test")

        assert test_db.get_simulation("delete-test") is None
        assert test_db.get_events("delete-test") == []


class TestEventOperations:
    """Tests for event CRUD operations."""

    def test_insert_event(self, test_db):
        """Should insert an event."""
        from server.event_bus import SimulationEvent

        event = SimulationEvent.create(
            "test_event",
            run_id="event-test",
            step=1,
            custom_data="hello"
        )

        event_id = test_db.insert_event(event)
        assert event_id > 0

    def test_get_events(self, test_db):
        """Should retrieve events for a run."""
        from server.event_bus import SimulationEvent

        test_db.insert_event(SimulationEvent.create("event1", run_id="get-test", step=1))
        test_db.insert_event(SimulationEvent.create("event2", run_id="get-test", step=2))
        test_db.insert_event(SimulationEvent.create("event3", run_id="other-run", step=1))

        events = test_db.get_events("get-test")
        assert len(events) == 2
        assert events[0].event_type == "event1"
        assert events[1].event_type == "event2"

    def test_get_events_by_type(self, test_db):
        """Should filter events by type."""
        from server.event_bus import SimulationEvent

        test_db.insert_event(SimulationEvent.create("type_a", run_id="type-test"))
        test_db.insert_event(SimulationEvent.create("type_b", run_id="type-test"))
        test_db.insert_event(SimulationEvent.create("type_a", run_id="type-test"))

        type_a_events = test_db.get_events("type-test", event_type="type_a")
        assert len(type_a_events) == 2

    def test_get_events_with_limit(self, test_db):
        """Should respect limit parameter."""
        from server.event_bus import SimulationEvent

        for i in range(10):
            test_db.insert_event(SimulationEvent.create(f"event_{i}", run_id="limit-test"))

        events = test_db.get_events("limit-test", limit=5)
        assert len(events) == 5

    def test_get_events_with_offset(self, test_db):
        """Should respect offset parameter."""
        from server.event_bus import SimulationEvent

        for i in range(10):
            test_db.insert_event(SimulationEvent.create(f"event_{i}", run_id="offset-test"))

        events = test_db.get_events("offset-test", limit=3, offset=5)
        assert len(events) == 3
        assert events[0].event_type == "event_5"

    def test_get_event_count(self, test_db):
        """Should return correct event count."""
        from server.event_bus import SimulationEvent

        for i in range(7):
            test_db.insert_event(SimulationEvent.create(f"event_{i}", run_id="count-test"))

        count = test_db.get_event_count("count-test")
        assert count == 7

    def test_get_all_run_ids(self, test_db):
        """Should return all unique run IDs."""
        from server.event_bus import SimulationEvent

        test_db.insert_event(SimulationEvent.create("e1", run_id="run-a"))
        test_db.insert_event(SimulationEvent.create("e2", run_id="run-b"))
        test_db.insert_event(SimulationEvent.create("e3", run_id="run-a"))

        run_ids = test_db.get_all_run_ids()
        assert sorted(run_ids) == ["run-a", "run-b"]

    def test_delete_events(self, test_db):
        """Should delete all events for a run."""
        from server.event_bus import SimulationEvent

        for i in range(5):
            test_db.insert_event(SimulationEvent.create(f"event_{i}", run_id="delete-events-test"))

        deleted = test_db.delete_events("delete-events-test")
        assert deleted == 5
        assert test_db.get_events("delete-events-test") == []


class TestDatabaseStats:
    """Tests for database statistics."""

    def test_get_stats(self, test_db):
        """Should return database statistics."""
        from server.event_bus import SimulationEvent

        test_db.insert_simulation(run_id="stats-1")
        test_db.insert_simulation(run_id="stats-2")
        test_db.update_simulation_status("stats-2", "completed")

        test_db.insert_event(SimulationEvent.create("e1", run_id="stats-1"))
        test_db.insert_event(SimulationEvent.create("e2", run_id="stats-1"))

        stats = test_db.get_stats()

        assert stats["total_simulations"] == 2
        assert stats["total_events"] == 2
        assert stats["simulations_by_status"]["running"] == 1
        assert stats["simulations_by_status"]["completed"] == 1


class TestEventBusDatabaseIntegration:
    """Tests for EventBus with database backend."""

    def test_eventbus_database_mode(self, db_path):
        """EventBus should use database when _use_database is True."""
        from server.database import set_db_path, init_db, get_db, close_connection
        from server.event_bus import EventBus, SimulationEvent

        # Setup database
        set_db_path(db_path)
        init_db()

        # Enable database mode
        EventBus._use_database = True
        EventBus._test_persist_path = None
        EventBus.reset_instance()

        bus = EventBus.get_instance()

        # Emit event
        event = SimulationEvent.create("db_test", run_id="db-integration")
        bus.emit(event)

        # Verify event is in database
        db = get_db()
        events = db.get_events("db-integration")
        assert len(events) == 1
        assert events[0].event_type == "db_test"

        # Cleanup
        EventBus._use_database = False
        EventBus.reset_instance()
        close_connection()
        set_db_path(None)

    def test_eventbus_loads_from_database(self, db_path):
        """EventBus should load history from database on init."""
        from server.database import set_db_path, init_db, get_db, close_connection
        from server.event_bus import EventBus, SimulationEvent

        # Setup database and insert events directly
        set_db_path(db_path)
        init_db()
        db = get_db()
        db.insert_event(SimulationEvent.create("preexisting", run_id="load-test"))

        # Enable database mode and create EventBus
        EventBus._use_database = True
        EventBus._test_persist_path = None
        EventBus.reset_instance()

        bus = EventBus.get_instance()

        # Should have loaded the preexisting event
        history = bus.get_history("load-test")
        assert len(history) == 1
        assert history[0].event_type == "preexisting"

        # Cleanup
        EventBus._use_database = False
        EventBus.reset_instance()
        close_connection()
        set_db_path(None)
