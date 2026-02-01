"""
Worker tasks for RQ job processing.

These functions are designed to run in separate worker processes,
handling simulation execution independently of the main API server.
"""

import os
import threading
import time
from pathlib import Path

from utils.ops_logger import get_ops_logger

logger = get_ops_logger("worker")


def _get_worker_id() -> str:
    """Get a unique worker ID based on hostname and PID."""
    import socket
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"


class HeartbeatThread:
    """Background thread that sends periodic heartbeats to the state manager."""

    def __init__(self, run_id: str, worker_id: str, interval: float = 10.0):
        self.run_id = run_id
        self.worker_id = worker_id
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_step = 0

    def start(self):
        """Start the heartbeat thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def update_step(self, step: int):
        """Update the current step for heartbeat reports."""
        self._current_step = step

    def _run(self):
        """Heartbeat loop."""
        from server.run_state import get_state_manager

        while not self._stop_event.is_set():
            try:
                state_manager = get_state_manager()
                if state_manager:
                    state_manager.heartbeat(
                        self.run_id,
                        self.worker_id,
                        step=self._current_step,
                    )
            except Exception as e:
                logger.warning(f"Heartbeat failed for {self.run_id}: {e}")

            self._stop_event.wait(self.interval)


def run_simulation_task(
    run_id: str,
    scenario_path: str,
    resume_from_step: int | None = None
) -> dict:
    """
    Execute a simulation as an RQ task.

    This is the main entry point for worker-based simulation execution.
    It handles the full simulation lifecycle including event emission.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML file
        resume_from_step: If set, resume simulation from this step

    Returns:
        Dictionary with simulation result summary
    """
    from core.orchestrator import Orchestrator
    from core.event_emitter import enable_event_emitter
    from server.event_bus import emit_event
    from server.run_state import get_state_manager

    scenario_path = Path(scenario_path)
    worker_id = _get_worker_id()
    heartbeat: HeartbeatThread | None = None

    logger.info("Starting simulation", extra={
        "run_id": run_id,
        "scenario": scenario_path.stem,
        "scenario_path": str(scenario_path),
        "resume_from_step": resume_from_step,
        "worker_id": worker_id,
    })

    # Transition to running state
    state_manager = get_state_manager()
    if state_manager:
        try:
            state_manager.transition(run_id, "running", worker_id=worker_id)
        except Exception as e:
            # May already be in running state (e.g., resume)
            logger.debug(f"Could not transition {run_id} to running: {e}")

        # Start heartbeat thread
        heartbeat = HeartbeatThread(run_id, worker_id)
        heartbeat.start()

    try:
        # Enable event emission for this run
        enable_event_emitter(run_id)

        # Create step callback for state updates
        def on_step(step: int, danger_count: int = 0):
            """Callback from orchestrator to update state."""
            if heartbeat:
                heartbeat.update_step(step)
            if state_manager:
                state_manager.update_progress(run_id, step, danger_count)

        # Run simulation
        orchestrator = Orchestrator(
            str(scenario_path),
            scenario_path.stem,
            save_frequency=1,
            run_id=run_id,
        )

        # Set the state callback if orchestrator supports it
        if hasattr(orchestrator, 'set_step_callback'):
            orchestrator.set_step_callback(on_step)

        if resume_from_step is not None:
            orchestrator.run(start_step=resume_from_step)
        else:
            orchestrator.run()

        # Transition to completed
        if state_manager:
            try:
                state_manager.transition(run_id, "completed")
            except Exception as e:
                logger.warning(f"Could not transition {run_id} to completed: {e}")

        logger.info("Simulation completed", extra={
            "run_id": run_id,
            "scenario": scenario_path.stem,
            "resumed_from": resume_from_step,
        })

        result = {
            "run_id": run_id,
            "status": "completed",
            "scenario": scenario_path.stem,
        }
        if resume_from_step is not None:
            result["resumed_from"] = resume_from_step

        return result

    except Exception as e:
        logger.error("Simulation failed", extra={
            "run_id": run_id,
            "error": str(e),
        })

        # Transition to failed state
        if state_manager:
            try:
                state_manager.transition(run_id, "failed", error=str(e))
            except Exception as te:
                logger.warning(f"Could not transition {run_id} to failed: {te}")

        emit_event("simulation_failed", run_id, error=str(e))
        raise  # Re-raise so RQ marks job as failed

    finally:
        # Stop heartbeat thread
        if heartbeat:
            heartbeat.stop()
