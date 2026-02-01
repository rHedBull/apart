"""
Worker tasks for RQ job processing.

These functions are designed to run in separate worker processes,
handling simulation execution independently of the main API server.
"""

from pathlib import Path

from utils.ops_logger import get_ops_logger

logger = get_ops_logger("worker")


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

    scenario_path = Path(scenario_path)

    logger.info("Starting simulation", extra={
        "run_id": run_id,
        "scenario": scenario_path.stem,
        "scenario_path": str(scenario_path),
        "resume_from_step": resume_from_step,
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

        if resume_from_step is not None:
            orchestrator.run(start_step=resume_from_step)
        else:
            orchestrator.run()

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
        emit_event("simulation_failed", run_id, error=str(e))
        raise  # Re-raise so RQ marks job as failed
