"""
Experiment runner for multi-condition simulations.

Runs the same scenario under different conditions and collects results.
"""

import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from experiment.models import (
    ConditionResults,
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    RunResult,
)
from experiment.condition import apply_modifications, validate_modifications


class ExperimentRunner:
    """
    Runs experiments with multiple conditions and repetitions.

    Example:
        config = ExperimentConfig(
            name="trust_sensitivity",
            description="Test how trust level affects outcomes",
            scenario_path="scenarios/cooperation.yaml",
            conditions=[
                ExperimentCondition("low_trust", modifications={"agent_vars.trust.default": 20}),
                ExperimentCondition("high_trust", modifications={"agent_vars.trust.default": 80}),
            ],
            runs_per_condition=3
        )
        runner = ExperimentRunner(config)
        result = runner.run_all()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm_provider: Optional[Any] = None,
        verbose: bool = True,
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            llm_provider: Optional LLM provider to inject (for testing)
            verbose: Whether to print progress
        """
        self.config = config
        self.llm_provider = llm_provider
        self.verbose = verbose
        self.result = ExperimentResult(config=config)

        # Load and validate base scenario
        self._base_config = self._load_scenario(config.scenario_path)
        self._validate_conditions()

    def _load_scenario(self, path: str) -> Dict[str, Any]:
        """Load scenario configuration from YAML file."""
        scenario_path = Path(path)
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(scenario_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid or empty scenario YAML: expected mapping, got {type(data).__name__}")

        return data

    def _validate_conditions(self) -> None:
        """Validate all conditions can be applied to base config."""
        all_errors = []

        for condition in self.config.conditions:
            errors = validate_modifications(self._base_config, condition.modifications)
            if errors:
                all_errors.extend(
                    [f"Condition '{condition.name}': {e}" for e in errors]
                )

        if all_errors:
            raise ValueError(
                "Invalid experiment conditions:\n  " + "\n  ".join(all_errors)
            )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in file paths to prevent directory traversal."""
        import re
        # Replace path separators and other unsafe characters with underscores
        sanitized = re.sub(r'[/\\:\x00<>"|?*]', '_', name)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        # Ensure non-empty
        return sanitized or 'unnamed'

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def run_single(
        self,
        condition: ExperimentCondition,
        run_index: int,
    ) -> RunResult:
        """
        Run a single simulation with the given condition.

        Args:
            condition: Condition to apply
            run_index: Index of this run (0-based)

        Returns:
            RunResult with final state and metadata
        """
        # Avoid circular import
        from core.orchestrator import Orchestrator

        # Apply condition modifications
        modified_config = apply_modifications(self._base_config, condition.modifications)

        # Create unique scenario name for this run (sanitized to prevent path traversal)
        safe_experiment_name = self._sanitize_name(self.config.name)
        safe_condition_name = self._sanitize_name(condition.name)
        scenario_name = f"{safe_experiment_name}_{safe_condition_name}_run{run_index}"

        # Write modified config to temp file
        temp_dir = Path(self.config.output_dir or "data/experiments") / self.result.experiment_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_config_path = temp_dir / f"{scenario_name}_config.yaml"

        with open(temp_config_path, "w") as f:
            yaml.safe_dump(modified_config, f)

        start_time = time.time()
        error = None
        final_state = {}
        run_dir = ""
        run_id = ""

        try:
            # Create and run orchestrator
            orchestrator = Orchestrator(
                config_path=str(temp_config_path),
                scenario_name=scenario_name,
                save_frequency=1,  # Save every step for analysis
                engine_llm_provider=self.llm_provider,
            )

            run_dir = str(orchestrator.persistence.run_dir)
            run_id = orchestrator.persistence.run_id

            orchestrator.run()

            # Get final state
            final_state = orchestrator.game_engine.get_state_snapshot()

        except Exception as e:
            error = str(e)
            self._log(f"  Run {run_index} failed: {error}")

        duration = time.time() - start_time

        return RunResult(
            condition_name=condition.name,
            run_index=run_index,
            final_state=final_state,
            run_id=run_id,
            run_dir=run_dir,
            duration_seconds=duration,
            error=error,
        )

    def run_condition(self, condition_name: str) -> ConditionResults:
        """
        Run all repetitions for a specific condition.

        Args:
            condition_name: Name of condition to run

        Returns:
            ConditionResults with all runs
        """
        # Find the condition
        condition = None
        for c in self.config.conditions:
            if c.name == condition_name:
                condition = c
                break

        if condition is None:
            raise ValueError(f"Unknown condition: {condition_name}")

        self._log(f"\nRunning condition: {condition_name}")
        if condition.description:
            self._log(f"  Description: {condition.description}")

        results = ConditionResults(condition=condition)

        for run_idx in range(self.config.runs_per_condition):
            self._log(f"  Run {run_idx + 1}/{self.config.runs_per_condition}...")
            run_result = self.run_single(condition, run_idx)
            results.runs.append(run_result)

            if run_result.succeeded:
                self._log(f"    Completed in {run_result.duration_seconds:.1f}s")
            else:
                self._log(f"    Failed: {run_result.error}")

        self._log(
            f"  Condition complete: {len(results.successful_runs)}/{len(results.runs)} succeeded"
        )

        # Store in experiment result
        self.result.condition_results[condition_name] = results

        return results

    def run_all(self) -> ExperimentResult:
        """
        Run all conditions and return complete results.

        Returns:
            ExperimentResult with all condition results
        """
        self._log(f"Starting experiment: {self.config.name}")
        self._log(f"  Conditions: {len(self.config.conditions)}")
        self._log(f"  Runs per condition: {self.config.runs_per_condition}")
        self._log(f"  Total runs: {len(self.config.conditions) * self.config.runs_per_condition}")

        for condition in self.config.conditions:
            self.run_condition(condition.name)

        self.result.completed_at = datetime.now()

        self._log(f"\nExperiment complete!")
        self._log(f"  Total runs: {self.result.total_runs}")
        self._log(f"  Successful: {self.result.successful_runs}")
        self._log(f"  Experiment ID: {self.result.experiment_id}")

        return self.result
