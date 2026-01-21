"""
Worker tasks for RQ job processing.

These functions are designed to run in separate worker processes,
handling simulation execution independently of the main API server.
"""

from pathlib import Path

from utils.ops_logger import get_ops_logger

logger = get_ops_logger("worker")


def run_simulation_task(run_id: str, scenario_path: str) -> dict:
    """
    Execute a simulation as an RQ task.

    This is the main entry point for worker-based simulation execution.
    It handles the full simulation lifecycle including event emission.

    Args:
        run_id: Unique simulation ID
        scenario_path: Path to scenario YAML file

    Returns:
        Dictionary with simulation result summary
    """
    from core.orchestrator import Orchestrator
    from core.event_emitter import enable_event_emitter
    from server.event_bus import emit_event

    scenario_path = Path(scenario_path)

    logger.info("Starting simulation", extra={
        "run_id": run_id,
        "scenario": scenario_path.stem,
        "scenario_path": str(scenario_path),
    })

    try:
        # Enable event emission for this run
        enable_event_emitter(run_id)

        # Run simulation
        orchestrator = Orchestrator(
            str(scenario_path),
            scenario_path.stem,
            save_frequency=1,
            run_id=run_id,
        )
        orchestrator.run()

        logger.info("Simulation completed", extra={
            "run_id": run_id,
            "scenario": scenario_path.stem,
        })
        return {
            "run_id": run_id,
            "status": "completed",
            "scenario": scenario_path.stem,
        }

    except Exception as e:
        logger.error("Simulation failed", extra={
            "run_id": run_id,
            "error": str(e),
        })
        emit_event("simulation_failed", run_id, error=str(e))
        raise  # Re-raise so RQ marks job as failed
