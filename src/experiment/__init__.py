"""
Experiment runner module.

Run multi-condition experiments and analyze results.
"""

from experiment.models import (
    ConditionResults,
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    RunResult,
)
from experiment.condition import apply_modifications, validate_modifications
from experiment.runner import ExperimentRunner
from experiment.results import (
    compare_conditions,
    generate_summary,
    list_experiments,
    load_experiment,
    save_experiment,
)

__all__ = [
    # Models
    "ConditionResults",
    "ExperimentCondition",
    "ExperimentConfig",
    "ExperimentResult",
    "RunResult",
    # Condition utilities
    "apply_modifications",
    "validate_modifications",
    # Runner
    "ExperimentRunner",
    # Results
    "compare_conditions",
    "generate_summary",
    "list_experiments",
    "load_experiment",
    "save_experiment",
]
