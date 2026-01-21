"""
SQLite database for persistent storage of simulations and events.

Provides:
- ACID transactions for data integrity
- Efficient querying without loading all data to memory
- Survives server restarts
- Easy backup/restore

Usage:
    from server.database import get_db, init_db

    # Initialize on startup
    init_db()

    # Use in code
    db = get_db()
    db.insert_simulation(run_id, scenario_path)
    db.insert_event(event)
    events = db.get_events(run_id)
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from server.event_bus import SimulationEvent


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "apart.db"

# Thread-local storage for connections
_local = threading.local()

# Global database path (can be overridden for testing)
_db_path: Path = DEFAULT_DB_PATH


def set_db_path(path: Path | str | None) -> None:
    """Set the database path. Use None to reset to default."""
    global _db_path
    _db_path = Path(path) if path else DEFAULT_DB_PATH


def get_db_path() -> Path:
    """Get the current database path."""
    return _db_path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """
    Get a database connection for the current thread.

    Uses thread-local storage to provide one connection per thread.
    Connections are reused within a thread for efficiency.
    """
    if not hasattr(_local, "connection") or _local.connection is None:
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        _local.connection = sqlite3.connect(
            str(_db_path),
            check_same_thread=False,
            timeout=30.0
        )
        _local.connection.row_factory = sqlite3.Row
        # Enable foreign keys
        _local.connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield _local.connection
    except Exception:
        _local.connection.rollback()
        raise


def close_connection() -> None:
    """Close the connection for the current thread."""
    if hasattr(_local, "connection") and _local.connection:
        _local.connection.close()
        _local.connection = None


def init_db() -> None:
    """
    Initialize the database schema.

    Creates tables if they don't exist. Safe to call multiple times.
    """
    with get_connection() as conn:
        conn.executescript("""
            -- Simulation runs table
            CREATE TABLE IF NOT EXISTS simulations (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                scenario_path TEXT,
                scenario_name TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                max_steps INTEGER,
                current_step INTEGER DEFAULT 0,
                config_json TEXT
            );

            -- Events table (no foreign key - events can exist independently)
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                step INTEGER,
                data_json TEXT
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);
        """)
        conn.commit()


def reset_db() -> None:
    """
    Reset the database by dropping all tables.

    WARNING: This deletes all data! Use only for testing.
    """
    with get_connection() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS events;
            DROP TABLE IF EXISTS simulations;
        """)
        conn.commit()
    init_db()


class Database:
    """
    Database interface for simulation and event storage.

    Thread-safe wrapper around SQLite operations.
    """

    def __init__(self):
        self._lock = threading.Lock()

    # =========================================================================
    # Simulation operations
    # =========================================================================

    def insert_simulation(
        self,
        run_id: str,
        scenario_path: str | None = None,
        scenario_name: str | None = None,
        max_steps: int | None = None,
        config: dict | None = None
    ) -> None:
        """Insert a new simulation record."""
        with self._lock:
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO simulations (run_id, status, scenario_path, scenario_name,
                                            started_at, max_steps, config_json)
                    VALUES (?, 'running', ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        scenario_path,
                        scenario_name,
                        datetime.now().isoformat(),
                        max_steps,
                        json.dumps(config) if config else None
                    )
                )
                conn.commit()

    def update_simulation_status(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None
    ) -> None:
        """Update simulation status."""
        with self._lock:
            with get_connection() as conn:
                if status in ("completed", "failed", "stopped"):
                    conn.execute(
                        """
                        UPDATE simulations
                        SET status = ?, completed_at = ?, error_message = ?
                        WHERE run_id = ?
                        """,
                        (status, datetime.now().isoformat(), error_message, run_id)
                    )
                else:
                    conn.execute(
                        "UPDATE simulations SET status = ? WHERE run_id = ?",
                        (status, run_id)
                    )
                conn.commit()

    def update_simulation_step(self, run_id: str, step: int) -> None:
        """Update current step for a simulation."""
        with self._lock:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE simulations SET current_step = ? WHERE run_id = ?",
                    (step, run_id)
                )
                conn.commit()

    def get_simulation(self, run_id: str) -> dict | None:
        """Get a simulation by run_id."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM simulations WHERE run_id = ?",
                (run_id,)
            ).fetchone()

            if row is None:
                return None

            return dict(row)

    def get_all_simulations(self, status: str | None = None) -> list[dict]:
        """Get all simulations, optionally filtered by status."""
        with get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM simulations WHERE status = ? ORDER BY started_at DESC",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM simulations ORDER BY started_at DESC"
                ).fetchall()

            return [dict(row) for row in rows]

    def delete_simulation(self, run_id: str) -> None:
        """Delete a simulation and its events."""
        with self._lock:
            with get_connection() as conn:
                conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM simulations WHERE run_id = ?", (run_id,))
                conn.commit()

    # =========================================================================
    # Event operations
    # =========================================================================

    def insert_event(self, event: SimulationEvent) -> int:
        """
        Insert an event and return its ID.

        Args:
            event: The SimulationEvent to insert

        Returns:
            The auto-generated event ID
        """
        with self._lock:
            with get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO events (run_id, event_type, timestamp, step, data_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.run_id,
                        event.event_type,
                        event.timestamp,
                        event.step,
                        json.dumps(event.data)
                    )
                )
                conn.commit()
                return cursor.lastrowid

    def get_events(
        self,
        run_id: str,
        event_type: str | None = None,
        limit: int | None = None,
        offset: int = 0
    ) -> list[SimulationEvent]:
        """
        Get events for a simulation.

        Args:
            run_id: The simulation run ID
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of SimulationEvent objects
        """
        with get_connection() as conn:
            query = "SELECT * FROM events WHERE run_id = ?"
            params: list = [run_id]

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            query += " ORDER BY id ASC"

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()

            return [
                SimulationEvent(
                    event_type=row["event_type"],
                    timestamp=row["timestamp"],
                    run_id=row["run_id"],
                    step=row["step"],
                    data=json.loads(row["data_json"]) if row["data_json"] else {}
                )
                for row in rows
            ]

    def get_event_count(self, run_id: str) -> int:
        """Get the total number of events for a simulation."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            return row["count"]

    def get_all_run_ids(self) -> list[str]:
        """Get all unique run IDs that have events."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM events ORDER BY run_id"
            ).fetchall()
            return [row["run_id"] for row in rows]

    def delete_events(self, run_id: str) -> int:
        """Delete all events for a simulation. Returns count deleted."""
        with self._lock:
            with get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM events WHERE run_id = ?",
                    (run_id,)
                )
                conn.commit()
                return cursor.rowcount

    # =========================================================================
    # Utility operations
    # =========================================================================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with get_connection() as conn:
            sim_count = conn.execute(
                "SELECT COUNT(*) as count FROM simulations"
            ).fetchone()["count"]

            event_count = conn.execute(
                "SELECT COUNT(*) as count FROM events"
            ).fetchone()["count"]

            status_counts = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) as count FROM simulations GROUP BY status"
            ).fetchall():
                status_counts[row["status"]] = row["count"]

            return {
                "total_simulations": sim_count,
                "total_events": event_count,
                "simulations_by_status": status_counts
            }


# Global database instance
_db: Database | None = None


def get_db() -> Database:
    """Get the global Database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def reset_db_instance() -> None:
    """Reset the global Database instance (for testing)."""
    global _db
    _db = None
