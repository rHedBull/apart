"""
Experiment results persistence and analysis.

Save, load, and analyze experiment results.
"""

import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment.models import (
    ConditionResults,
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    RunResult,
)


def save_experiment(
    result: ExperimentResult,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Save experiment results to disk.

    Args:
        result: ExperimentResult to save
        output_dir: Directory to save to (default: data/experiments/)

    Returns:
        Path to saved experiment directory
    """
    base_dir = Path(output_dir or "data/experiments")
    experiment_dir = base_dir / result.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save main result file
    result_data = _result_to_dict(result)
    result_path = experiment_dir / "experiment.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)

    # Save summary
    summary = generate_summary(result)
    summary_path = experiment_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    return experiment_dir


def load_experiment(experiment_id: str, base_dir: Optional[str] = None) -> ExperimentResult:
    """
    Load experiment results from disk.

    Args:
        experiment_id: ID of experiment to load
        base_dir: Base directory (default: data/experiments/)

    Returns:
        ExperimentResult
    """
    base_path = Path(base_dir or "data/experiments")
    result_path = base_path / experiment_id / "experiment.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")

    with open(result_path) as f:
        data = json.load(f)

    return _dict_to_result(data)


def list_experiments(base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all saved experiments.

    Args:
        base_dir: Base directory (default: data/experiments/)

    Returns:
        List of experiment summaries (id, name, date, conditions, runs)
    """
    base_path = Path(base_dir or "data/experiments")
    experiments = []

    if not base_path.exists():
        return experiments

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        result_path = exp_dir / "experiment.json"
        if not result_path.exists():
            continue

        try:
            with open(result_path) as f:
                data = json.load(f)

            experiments.append({
                "id": data.get("experiment_id", exp_dir.name),
                "name": data.get("config", {}).get("name", "unknown"),
                "started_at": data.get("started_at"),
                "conditions": len(data.get("condition_results", {})),
                "total_runs": sum(
                    len(cr.get("runs", []))
                    for cr in data.get("condition_results", {}).values()
                ),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(experiments, key=lambda x: x.get("started_at", ""), reverse=True)


def compare_conditions(
    result: ExperimentResult,
    var_path: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare a variable across conditions.

    Args:
        result: ExperimentResult to analyze
        var_path: Dot-path to variable, e.g. "global_vars.tension"

    Returns:
        Dict mapping condition name to statistics:
        {
            "condition_name": {
                "values": [v1, v2, ...],
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
                "n": int
            }
        }
    """
    comparison = {}

    for cond_name, cond_results in result.condition_results.items():
        values = cond_results.get_final_var_values(var_path)

        if not values:
            comparison[cond_name] = {
                "values": [],
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n": 0,
            }
            continue

        # Filter to numeric values only
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if not numeric_values:
            comparison[cond_name] = {
                "values": values,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n": len(values),
            }
            continue

        comparison[cond_name] = {
            "values": numeric_values,
            "mean": statistics.mean(numeric_values),
            "std": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
            "min": min(numeric_values),
            "max": max(numeric_values),
            "n": len(numeric_values),
        }

    return comparison


def generate_summary(result: ExperimentResult) -> str:
    """
    Generate human-readable summary of experiment results.

    Args:
        result: ExperimentResult to summarize

    Returns:
        Formatted summary string
    """
    lines = [
        f"Experiment: {result.config.name}",
        f"ID: {result.experiment_id}",
        f"Description: {result.config.description}",
        "",
        f"Scenario: {result.config.scenario_path}",
        f"Runs per condition: {result.config.runs_per_condition}",
        "",
        "=" * 60,
        "CONDITIONS",
        "=" * 60,
    ]

    for cond_name, cond_results in result.condition_results.items():
        cond = cond_results.condition
        lines.append(f"\n{cond_name}:")
        if cond.description:
            lines.append(f"  Description: {cond.description}")
        if cond.modifications:
            lines.append("  Modifications:")
            for path, value in cond.modifications.items():
                lines.append(f"    {path}: {value}")

        lines.append(f"  Runs: {len(cond_results.runs)}")
        lines.append(
            f"  Success rate: {cond_results.success_rate:.1%} "
            f"({len(cond_results.successful_runs)}/{len(cond_results.runs)})"
        )

        if cond_results.successful_runs:
            durations = [r.duration_seconds for r in cond_results.successful_runs]
            avg_duration = statistics.mean(durations)
            lines.append(f"  Avg duration: {avg_duration:.1f}s")

    lines.extend([
        "",
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"Total runs: {result.total_runs}",
        f"Successful: {result.successful_runs}",
        f"Started: {result.started_at}",
        f"Completed: {result.completed_at}",
    ])

    return "\n".join(lines)


def _result_to_dict(result: ExperimentResult) -> Dict[str, Any]:
    """Convert ExperimentResult to serializable dict."""
    return {
        "experiment_id": result.experiment_id,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "config": {
            "name": result.config.name,
            "description": result.config.description,
            "scenario_path": result.config.scenario_path,
            "runs_per_condition": result.config.runs_per_condition,
            "output_dir": result.config.output_dir,
            "conditions": [
                {
                    "name": c.name,
                    "description": c.description,
                    "modifications": c.modifications,
                }
                for c in result.config.conditions
            ],
        },
        "condition_results": {
            name: {
                "condition": {
                    "name": cr.condition.name,
                    "description": cr.condition.description,
                    "modifications": cr.condition.modifications,
                },
                "runs": [
                    {
                        "condition_name": r.condition_name,
                        "run_index": r.run_index,
                        "final_state": r.final_state,
                        "run_id": r.run_id,
                        "run_dir": r.run_dir,
                        "duration_seconds": r.duration_seconds,
                        "error": r.error,
                    }
                    for r in cr.runs
                ],
            }
            for name, cr in result.condition_results.items()
        },
    }


def _dict_to_result(data: Dict[str, Any]) -> ExperimentResult:
    """Convert dict back to ExperimentResult."""
    config_data = data["config"]
    conditions = [
        ExperimentCondition(
            name=c["name"],
            description=c.get("description", ""),
            modifications=c.get("modifications", {}),
        )
        for c in config_data["conditions"]
    ]

    config = ExperimentConfig(
        name=config_data["name"],
        description=config_data.get("description", ""),
        scenario_path=config_data["scenario_path"],
        conditions=conditions,
        runs_per_condition=config_data.get("runs_per_condition", 1),
        output_dir=config_data.get("output_dir"),
    )

    result = ExperimentResult(
        config=config,
        experiment_id=data.get("experiment_id", ""),
        started_at=_parse_datetime(data.get("started_at")),
        completed_at=_parse_datetime(data.get("completed_at")),
    )

    # Restore condition results
    for name, cr_data in data.get("condition_results", {}).items():
        cond_data = cr_data["condition"]
        condition = ExperimentCondition(
            name=cond_data["name"],
            description=cond_data.get("description", ""),
            modifications=cond_data.get("modifications", {}),
        )

        runs = [
            RunResult(
                condition_name=r["condition_name"],
                run_index=r["run_index"],
                final_state=r.get("final_state", {}),
                run_id=r.get("run_id", ""),
                run_dir=r.get("run_dir", ""),
                duration_seconds=r.get("duration_seconds", 0.0),
                error=r.get("error"),
            )
            for r in cr_data.get("runs", [])
        ]

        result.condition_results[name] = ConditionResults(
            condition=condition,
            runs=runs,
        )

    return result


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
