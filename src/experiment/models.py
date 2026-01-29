"""
Data models for experiment runner.

Experiments run multiple conditions of a scenario and compare results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentCondition:
    """
    A single experimental condition.

    Conditions modify the base scenario configuration to test
    different parameter values or settings.
    """

    name: str
    description: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)
    """
    Dot-path modifications to apply to the base config.

    Examples:
        {"agent_vars.trust_level.default": 80}
        {"agents.0.variables.resource": 100}
        {"global_vars.cooperation_bonus.default": 1.5}
    """


@dataclass
class ExperimentConfig:
    """
    Configuration for running an experiment.

    An experiment consists of a base scenario run under multiple
    conditions, with multiple repetitions per condition.
    """

    name: str
    description: str
    scenario_path: str
    conditions: List[ExperimentCondition]
    runs_per_condition: int = 1
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.runs_per_condition < 1:
            raise ValueError("runs_per_condition must be at least 1")
        if not self.conditions:
            raise ValueError("At least one condition is required")

        # Validate condition names are non-empty and unique
        seen_names = set()
        for condition in self.conditions:
            if not condition.name or not condition.name.strip():
                raise ValueError("Condition names must be non-empty")
            if condition.name in seen_names:
                raise ValueError(f"Duplicate condition name: '{condition.name}'")
            seen_names.add(condition.name)


@dataclass
class RunResult:
    """
    Result of a single simulation run.
    """

    condition_name: str
    run_index: int
    final_state: Dict[str, Any]
    """Final global_vars and agent_vars snapshot."""

    history: List[Dict[str, Any]] = field(default_factory=list)
    """Step-by-step state history if captured."""

    run_id: str = ""
    """Unique identifier for this run."""

    run_dir: str = ""
    """Directory where run artifacts are stored."""

    duration_seconds: float = 0.0
    """Wall-clock time for the run."""

    error: Optional[str] = None
    """Error message if run failed."""

    @property
    def succeeded(self) -> bool:
        """Whether the run completed without error."""
        return self.error is None


@dataclass
class ConditionResults:
    """
    Aggregated results for a single condition.
    """

    condition: ExperimentCondition
    runs: List[RunResult] = field(default_factory=list)

    @property
    def successful_runs(self) -> List[RunResult]:
        """Runs that completed without error."""
        return [r for r in self.runs if r.succeeded]

    @property
    def failed_runs(self) -> List[RunResult]:
        """Runs that failed with an error."""
        return [r for r in self.runs if not r.succeeded]

    @property
    def success_rate(self) -> float:
        """Fraction of runs that succeeded."""
        if not self.runs:
            return 0.0
        return len(self.successful_runs) / len(self.runs)

    def get_final_var_values(self, var_path: str) -> List[Any]:
        """
        Get final values of a variable across all successful runs.

        Args:
            var_path: Dot-path to variable, e.g. "global_vars.tension"
                     or "agent_vars.USA.trust_level"

        Returns:
            List of values, one per successful run.
        """
        values = []
        for run in self.successful_runs:
            value = _get_nested_value(run.final_state, var_path)
            if value is not None:
                values.append(value)
        return values


@dataclass
class ExperimentResult:
    """
    Complete results of an experiment.
    """

    config: ExperimentConfig
    condition_results: Dict[str, ConditionResults] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    experiment_id: str = ""

    def __post_init__(self):
        if not self.experiment_id:
            timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"{self.config.name}_{timestamp}"

    @property
    def total_runs(self) -> int:
        """Total number of runs across all conditions."""
        return sum(len(cr.runs) for cr in self.condition_results.values())

    @property
    def successful_runs(self) -> int:
        """Total successful runs across all conditions."""
        return sum(len(cr.successful_runs) for cr in self.condition_results.values())

    @property
    def is_complete(self) -> bool:
        """Whether all expected runs have been executed."""
        expected = len(self.config.conditions) * self.config.runs_per_condition
        return self.total_runs >= expected

    def get_condition_results(self, condition_name: str) -> Optional[ConditionResults]:
        """Get results for a specific condition."""
        return self.condition_results.get(condition_name)


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """
    Get a value from nested dict using dot-path.

    Args:
        data: Nested dictionary
        path: Dot-separated path, e.g. "global_vars.tension"

    Returns:
        Value at path, or None if not found.
    """
    parts = path.split(".")
    current = data

    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None

    return current
