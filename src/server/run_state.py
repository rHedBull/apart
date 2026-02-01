"""
Centralized run state management with Redis backend.

Provides:
- Single source of truth for simulation run state
- Defined state machine with valid transitions
- Atomic state transitions with optimistic locking
- Worker heartbeat tracking for crash detection
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any

from redis import Redis

from utils.ops_logger import get_ops_logger

logger = get_ops_logger("run_state")

# Redis key prefixes
KEY_PREFIX = "apart:run:"
HEARTBEAT_SUFFIX = ":heartbeat"
STATE_SUFFIX = ":state"
INDEX_KEY = "apart:runs:index"

# Heartbeat TTL in seconds (worker must send heartbeat within this time)
HEARTBEAT_TTL = 30

# Valid state transitions
VALID_TRANSITIONS = {
    "pending": ["running", "cancelled"],
    "running": ["paused", "completed", "failed", "interrupted"],
    "paused": ["running", "cancelled", "interrupted"],
    "completed": [],  # terminal
    "failed": [],  # terminal
    "interrupted": ["running"],  # can resume interrupted runs
    "cancelled": [],  # terminal
}

# Terminal states (no further transitions allowed)
TERMINAL_STATES = {"completed", "failed", "cancelled"}


@dataclass
class RunState:
    """State of a simulation run stored in Redis."""

    run_id: str
    status: str  # pending, running, paused, completed, failed, interrupted, cancelled
    scenario_path: str
    scenario_name: str

    # Progress
    current_step: int = 0
    total_steps: Optional[int] = None

    # Timestamps
    created_at: str = ""
    started_at: Optional[str] = None
    paused_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Worker tracking
    worker_id: Optional[str] = None
    last_heartbeat: Optional[str] = None

    # Metadata
    priority: str = "normal"
    error: Optional[str] = None
    danger_count: int = 0

    # Version for optimistic locking
    version: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to frontend-expected format."""
        return {
            "runId": self.run_id,
            "scenario": self.scenario_name,
            "status": self.status,
            "currentStep": self.current_step,
            "totalSteps": self.total_steps,
            "startedAt": self.started_at or self.created_at,
            "completedAt": self.completed_at,
            "dangerCount": self.danger_count,
            "workerId": self.worker_id,
            "priority": self.priority,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "RunState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class InvalidTransitionError(Exception):
    """Raised when attempting an invalid state transition."""
    pass


class OptimisticLockError(Exception):
    """Raised when optimistic locking detects a conflict."""
    pass


class RunStateManager:
    """
    Centralized run state management with Redis backend.

    Provides:
    - State creation and retrieval
    - State machine enforcement for valid transitions
    - Atomic transitions with optimistic locking
    - Worker heartbeat tracking for crash detection
    """

    _instance: Optional["RunStateManager"] = None
    _lock = threading.Lock()

    def __init__(self, redis_conn: Redis):
        """
        Initialize the state manager.

        Args:
            redis_conn: Redis connection instance
        """
        self._redis = redis_conn

    @classmethod
    def get_instance(cls) -> Optional["RunStateManager"]:
        """Get the singleton instance (may be None if not initialized)."""
        return cls._instance

    @classmethod
    def initialize(cls, redis_conn: Redis) -> "RunStateManager":
        """Initialize the singleton instance."""
        with cls._lock:
            cls._instance = cls(redis_conn)
            logger.info("RunStateManager initialized")
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _state_key(self, run_id: str) -> str:
        """Get Redis key for run state."""
        return f"{KEY_PREFIX}{run_id}{STATE_SUFFIX}"

    def _heartbeat_key(self, run_id: str) -> str:
        """Get Redis key for heartbeat."""
        return f"{KEY_PREFIX}{run_id}{HEARTBEAT_SUFFIX}"

    # =========================================================================
    # State Operations
    # =========================================================================

    def create_run(
        self,
        run_id: str,
        scenario_path: str,
        scenario_name: str,
        priority: str = "normal",
        total_steps: Optional[int] = None,
    ) -> RunState:
        """
        Create a new run state entry.

        Args:
            run_id: Unique simulation ID
            scenario_path: Path to scenario YAML file
            scenario_name: Human-readable scenario name
            priority: Queue priority (high, normal, low)
            total_steps: Total number of steps (if known)

        Returns:
            The created RunState

        Raises:
            ValueError: If run already exists
        """
        state_key = self._state_key(run_id)

        # Check if run already exists
        if self._redis.exists(state_key):
            raise ValueError(f"Run {run_id} already exists")

        state = RunState(
            run_id=run_id,
            status="pending",
            scenario_path=scenario_path,
            scenario_name=scenario_name,
            priority=priority,
            total_steps=total_steps,
        )

        # Store state
        self._redis.set(state_key, state.to_json())

        # Add to index (sorted set by created_at timestamp)
        self._redis.zadd(INDEX_KEY, {run_id: datetime.now().timestamp()})

        logger.info("Run state created", extra={
            "run_id": run_id,
            "scenario": scenario_name,
            "priority": priority,
        })

        return state

    def get_state(self, run_id: str) -> Optional[RunState]:
        """
        Get the current state of a run.

        Args:
            run_id: The run ID to look up

        Returns:
            RunState if found, None otherwise
        """
        state_key = self._state_key(run_id)
        data = self._redis.get(state_key)

        if data is None:
            return None

        return RunState.from_json(data)

    def transition(
        self,
        run_id: str,
        new_status: str,
        **kwargs,
    ) -> RunState:
        """
        Transition a run to a new status.

        This method:
        1. Validates the transition is allowed
        2. Uses optimistic locking to prevent conflicts
        3. Updates timestamps appropriately

        Args:
            run_id: The run ID to transition
            new_status: The new status to transition to
            **kwargs: Additional fields to update (worker_id, error, current_step, etc.)

        Returns:
            The updated RunState

        Raises:
            ValueError: If run doesn't exist
            InvalidTransitionError: If transition is not allowed
            OptimisticLockError: If concurrent modification detected
        """
        state_key = self._state_key(run_id)

        # Retry loop for optimistic locking
        max_retries = 3
        for attempt in range(max_retries):
            # Start watching the key
            pipe = self._redis.pipeline(True)
            try:
                pipe.watch(state_key)

                # Get current state
                data = pipe.get(state_key)
                if data is None:
                    pipe.unwatch()
                    raise ValueError(f"Run {run_id} not found")

                state = RunState.from_json(data)

                # Validate transition
                if new_status not in VALID_TRANSITIONS.get(state.status, []):
                    pipe.unwatch()
                    raise InvalidTransitionError(
                        f"Cannot transition from '{state.status}' to '{new_status}'"
                    )

                # Update state
                old_status = state.status
                state.status = new_status
                state.version += 1

                # Update timestamps based on transition
                now = datetime.now().isoformat()
                if new_status == "running" and old_status in ("pending", "paused", "interrupted"):
                    if old_status == "pending":
                        state.started_at = now
                    state.paused_at = None
                elif new_status == "paused":
                    state.paused_at = now
                elif new_status in TERMINAL_STATES:
                    state.completed_at = now

                # Apply additional updates
                for key, value in kwargs.items():
                    if hasattr(state, key):
                        setattr(state, key, value)

                # Execute atomic update
                pipe.multi()
                pipe.set(state_key, state.to_json())
                pipe.execute()

                logger.info("Run state transitioned", extra={
                    "run_id": run_id,
                    "from_status": old_status,
                    "to_status": new_status,
                    "version": state.version,
                })

                return state

            except Exception as e:
                # WatchError means another client modified the key
                if "WatchError" in type(e).__name__ or "WATCH" in str(e):
                    if attempt < max_retries - 1:
                        continue  # Retry
                    raise OptimisticLockError(
                        f"Concurrent modification of run {run_id}"
                    ) from e
                raise
            finally:
                pipe.reset()

        # Should not reach here
        raise OptimisticLockError(f"Failed to update run {run_id} after {max_retries} attempts")

    def update_progress(
        self,
        run_id: str,
        current_step: int,
        danger_count: Optional[int] = None,
    ) -> Optional[RunState]:
        """
        Update run progress (step count, danger signals).

        This is a non-transitional update used during normal execution.

        Args:
            run_id: The run ID to update
            current_step: Current step number
            danger_count: Updated danger signal count (optional)

        Returns:
            Updated RunState, or None if run not found
        """
        state_key = self._state_key(run_id)

        pipe = self._redis.pipeline(True)
        try:
            pipe.watch(state_key)

            data = pipe.get(state_key)
            if data is None:
                pipe.unwatch()
                return None

            state = RunState.from_json(data)
            state.current_step = current_step
            state.version += 1

            if danger_count is not None:
                state.danger_count = danger_count

            pipe.multi()
            pipe.set(state_key, state.to_json())
            pipe.execute()

            return state

        finally:
            pipe.reset()

    def list_runs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunState]:
        """
        List all runs, optionally filtered by status.

        Args:
            status: Filter by status (optional)
            limit: Maximum number of runs to return
            offset: Number of runs to skip

        Returns:
            List of RunState objects sorted by creation time (newest first)
        """
        # Get run IDs from index (sorted by timestamp, descending)
        run_ids = self._redis.zrevrange(INDEX_KEY, offset, offset + limit - 1)

        if not run_ids:
            return []

        # Fetch all states in batch
        pipe = self._redis.pipeline(False)
        for run_id in run_ids:
            pipe.get(self._state_key(run_id.decode() if isinstance(run_id, bytes) else run_id))
        results = pipe.execute()

        states = []
        for data in results:
            if data is not None:
                state = RunState.from_json(data)
                if status is None or state.status == status:
                    states.append(state)

        return states

    def count_runs(self) -> int:
        """Return total number of tracked runs."""
        return self._redis.zcard(INDEX_KEY)

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run's state.

        Args:
            run_id: The run ID to delete

        Returns:
            True if deleted, False if not found
        """
        state_key = self._state_key(run_id)
        heartbeat_key = self._heartbeat_key(run_id)

        pipe = self._redis.pipeline(False)
        pipe.delete(state_key)
        pipe.delete(heartbeat_key)
        pipe.zrem(INDEX_KEY, run_id)
        results = pipe.execute()

        deleted = results[0] > 0
        if deleted:
            logger.info("Run state deleted", extra={"run_id": run_id})

        return deleted

    # =========================================================================
    # Worker Heartbeat
    # =========================================================================

    def heartbeat(
        self,
        run_id: str,
        worker_id: str,
        step: Optional[int] = None,
    ) -> bool:
        """
        Send a heartbeat for a running simulation.

        Args:
            run_id: The run ID
            worker_id: The worker's unique ID
            step: Current step (optional, updates progress if provided)

        Returns:
            True if heartbeat was recorded
        """
        heartbeat_key = self._heartbeat_key(run_id)
        state_key = self._state_key(run_id)
        now = datetime.now().isoformat()

        # Set heartbeat with TTL
        self._redis.setex(heartbeat_key, HEARTBEAT_TTL, now)

        # Update state with heartbeat timestamp and optionally step
        pipe = self._redis.pipeline(True)
        try:
            pipe.watch(state_key)

            data = pipe.get(state_key)
            if data is None:
                pipe.unwatch()
                return False

            state = RunState.from_json(data)
            state.last_heartbeat = now
            state.worker_id = worker_id
            if step is not None:
                state.current_step = step

            pipe.multi()
            pipe.set(state_key, state.to_json())
            pipe.execute()

            return True

        except Exception:
            return False
        finally:
            pipe.reset()

    def is_heartbeat_stale(self, run_id: str) -> bool:
        """
        Check if a run's heartbeat has expired.

        Args:
            run_id: The run ID to check

        Returns:
            True if heartbeat is missing or expired
        """
        heartbeat_key = self._heartbeat_key(run_id)
        return not self._redis.exists(heartbeat_key)

    def check_stale_runs(self, timeout_seconds: int = 60) -> list[str]:
        """
        Find runs that appear to be stale (worker died).

        A run is considered stale if:
        - Status is "running"
        - Heartbeat has expired (not refreshed within timeout)

        Args:
            timeout_seconds: How long without heartbeat before considered stale

        Returns:
            List of stale run IDs
        """
        stale_runs = []

        # Get all running runs
        running_runs = self.list_runs(status="running", limit=1000)

        for state in running_runs:
            if self.is_heartbeat_stale(state.run_id):
                stale_runs.append(state.run_id)

        if stale_runs:
            logger.warning("Found stale runs", extra={
                "stale_run_ids": stale_runs,
                "count": len(stale_runs),
            })

        return stale_runs

    def mark_interrupted(self, run_id: str, reason: str = "Worker heartbeat expired") -> Optional[RunState]:
        """
        Mark a run as interrupted (typically due to worker crash).

        Args:
            run_id: The run ID to mark as interrupted
            reason: Reason for interruption

        Returns:
            Updated RunState, or None if transition failed
        """
        try:
            return self.transition(
                run_id,
                "interrupted",
                error=reason,
            )
        except (ValueError, InvalidTransitionError) as e:
            logger.warning(f"Could not mark {run_id} as interrupted: {e}")
            return None


# =========================================================================
# Module-level accessor
# =========================================================================

def get_state_manager() -> Optional[RunStateManager]:
    """Get the RunStateManager singleton instance."""
    return RunStateManager.get_instance()
