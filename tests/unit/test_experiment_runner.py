"""
Tests for experiment runner module.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

from experiment.models import (
    ConditionResults,
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    RunResult,
    _get_nested_value,
)
from experiment.condition import (
    apply_modifications,
    validate_modifications,
    _set_nested_value,
)
from experiment.results import (
    compare_conditions,
    generate_summary,
    save_experiment,
    load_experiment,
    _result_to_dict,
    _dict_to_result,
)


class TestExperimentCondition:
    """Tests for ExperimentCondition dataclass."""

    def test_create_simple_condition(self):
        cond = ExperimentCondition(name="baseline")
        assert cond.name == "baseline"
        assert cond.modifications == {}
        assert cond.description == ""

    def test_create_condition_with_modifications(self):
        cond = ExperimentCondition(
            name="high_trust",
            description="Test with high initial trust",
            modifications={"agent_vars.trust.default": 80},
        )
        assert cond.name == "high_trust"
        assert cond.modifications == {"agent_vars.trust.default": 80}


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_create_config(self):
        conditions = [
            ExperimentCondition(name="baseline"),
            ExperimentCondition(name="variant", modifications={"x": 1}),
        ]
        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
            scenario_path="scenarios/test.yaml",
            conditions=conditions,
            runs_per_condition=3,
        )
        assert config.name == "test_exp"
        assert len(config.conditions) == 2
        assert config.runs_per_condition == 3

    def test_config_requires_conditions(self):
        with pytest.raises(ValueError, match="At least one condition"):
            ExperimentConfig(
                name="test",
                description="",
                scenario_path="test.yaml",
                conditions=[],
            )

    def test_config_requires_positive_runs(self):
        with pytest.raises(ValueError, match="runs_per_condition must be at least 1"):
            ExperimentConfig(
                name="test",
                description="",
                scenario_path="test.yaml",
                conditions=[ExperimentCondition(name="test")],
                runs_per_condition=0,
            )


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_succeeded_property(self):
        success = RunResult(
            condition_name="test",
            run_index=0,
            final_state={"global_vars": {"x": 1}},
        )
        assert success.succeeded is True

        failed = RunResult(
            condition_name="test",
            run_index=0,
            final_state={},
            error="Something went wrong",
        )
        assert failed.succeeded is False


class TestConditionResults:
    """Tests for ConditionResults dataclass."""

    def test_success_rate(self):
        cond = ExperimentCondition(name="test")
        results = ConditionResults(
            condition=cond,
            runs=[
                RunResult("test", 0, {"x": 1}),
                RunResult("test", 1, {}, error="failed"),
                RunResult("test", 2, {"x": 3}),
            ],
        )
        assert results.success_rate == pytest.approx(2 / 3)
        assert len(results.successful_runs) == 2
        assert len(results.failed_runs) == 1

    def test_get_final_var_values(self):
        cond = ExperimentCondition(name="test")
        results = ConditionResults(
            condition=cond,
            runs=[
                RunResult("test", 0, {"global_vars": {"tension": 50}}),
                RunResult("test", 1, {"global_vars": {"tension": 60}}),
                RunResult("test", 2, {}, error="failed"),
            ],
        )
        values = results.get_final_var_values("global_vars.tension")
        assert values == [50, 60]


class TestApplyModifications:
    """Tests for apply_modifications function."""

    def test_simple_modification(self):
        config = {"agent_vars": {"trust": {"default": 50}}}
        mods = {"agent_vars.trust.default": 80}
        result = apply_modifications(config, mods)

        assert result["agent_vars"]["trust"]["default"] == 80
        # Original unchanged
        assert config["agent_vars"]["trust"]["default"] == 50

    def test_list_index_modification(self):
        config = {"agents": [{"name": "A", "x": 1}, {"name": "B", "x": 2}]}
        mods = {"agents.0.x": 10}
        result = apply_modifications(config, mods)

        assert result["agents"][0]["x"] == 10
        assert result["agents"][1]["x"] == 2

    def test_bracket_notation(self):
        config = {"agents": [{"name": "A", "x": 1}]}
        mods = {"agents[0].x": 10}
        result = apply_modifications(config, mods)

        assert result["agents"][0]["x"] == 10

    def test_multiple_modifications(self):
        config = {"a": {"b": 1}, "c": {"d": 2}}
        mods = {"a.b": 10, "c.d": 20}
        result = apply_modifications(config, mods)

        assert result["a"]["b"] == 10
        assert result["c"]["d"] == 20


class TestSetNestedValue:
    """Tests for _set_nested_value function."""

    def test_missing_key_raises(self):
        data = {"a": {"b": 1}}
        with pytest.raises(KeyError, match="not found"):
            _set_nested_value(data, "a.c", 2)

    def test_invalid_list_index_raises(self):
        data = {"items": [1, 2, 3]}
        with pytest.raises(IndexError, match="out of range"):
            _set_nested_value(data, "items.5", 10)


class TestValidateModifications:
    """Tests for validate_modifications function."""

    def test_valid_modifications(self):
        config = {"a": {"b": 1}, "c": 2}
        mods = {"a.b": 10, "c": 20}
        errors = validate_modifications(config, mods)
        assert errors == []

    def test_invalid_path_returns_error(self):
        config = {"a": {"b": 1}}
        mods = {"a.x": 10}
        errors = validate_modifications(config, mods)
        assert len(errors) == 1
        assert "a.x" in errors[0]


class TestGetNestedValue:
    """Tests for _get_nested_value function."""

    def test_simple_path(self):
        data = {"a": {"b": {"c": 42}}}
        assert _get_nested_value(data, "a.b.c") == 42

    def test_list_index(self):
        data = {"items": [10, 20, 30]}
        assert _get_nested_value(data, "items.1") == 20

    def test_missing_key_returns_none(self):
        data = {"a": {"b": 1}}
        assert _get_nested_value(data, "a.c") is None

    def test_invalid_index_returns_none(self):
        data = {"items": [1, 2]}
        assert _get_nested_value(data, "items.5") is None


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_auto_generates_experiment_id(self):
        config = ExperimentConfig(
            name="test",
            description="",
            scenario_path="test.yaml",
            conditions=[ExperimentCondition(name="baseline")],
        )
        result = ExperimentResult(config=config)
        assert result.experiment_id.startswith("test_")

    def test_total_runs(self):
        config = ExperimentConfig(
            name="test",
            description="",
            scenario_path="test.yaml",
            conditions=[ExperimentCondition(name="a"), ExperimentCondition(name="b")],
        )
        result = ExperimentResult(config=config)
        result.condition_results["a"] = ConditionResults(
            condition=config.conditions[0],
            runs=[RunResult("a", 0, {}), RunResult("a", 1, {})],
        )
        result.condition_results["b"] = ConditionResults(
            condition=config.conditions[1],
            runs=[RunResult("b", 0, {})],
        )

        assert result.total_runs == 3


class TestCompareConditions:
    """Tests for compare_conditions function."""

    def test_compare_numeric_values(self):
        config = ExperimentConfig(
            name="test",
            description="",
            scenario_path="test.yaml",
            conditions=[
                ExperimentCondition(name="low"),
                ExperimentCondition(name="high"),
            ],
        )
        result = ExperimentResult(config=config)

        result.condition_results["low"] = ConditionResults(
            condition=config.conditions[0],
            runs=[
                RunResult("low", 0, {"global_vars": {"tension": 20}}),
                RunResult("low", 1, {"global_vars": {"tension": 30}}),
            ],
        )
        result.condition_results["high"] = ConditionResults(
            condition=config.conditions[1],
            runs=[
                RunResult("high", 0, {"global_vars": {"tension": 70}}),
                RunResult("high", 1, {"global_vars": {"tension": 80}}),
            ],
        )

        comparison = compare_conditions(result, "global_vars.tension")

        assert comparison["low"]["mean"] == 25.0
        assert comparison["high"]["mean"] == 75.0
        assert comparison["low"]["n"] == 2
        assert comparison["high"]["n"] == 2


class TestResultSerialization:
    """Tests for result save/load functions."""

    def test_result_roundtrip(self):
        config = ExperimentConfig(
            name="test",
            description="Test description",
            scenario_path="test.yaml",
            conditions=[
                ExperimentCondition("baseline", "Base case", {}),
                ExperimentCondition("variant", "Modified", {"x.y": 10}),
            ],
            runs_per_condition=2,
        )
        result = ExperimentResult(
            config=config,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        result.condition_results["baseline"] = ConditionResults(
            condition=config.conditions[0],
            runs=[
                RunResult("baseline", 0, {"global_vars": {"x": 1}}, run_id="r1"),
                RunResult("baseline", 1, {"global_vars": {"x": 2}}, run_id="r2"),
            ],
        )

        # Convert to dict and back
        data = _result_to_dict(result)
        restored = _dict_to_result(data)

        assert restored.config.name == "test"
        assert len(restored.config.conditions) == 2
        assert "baseline" in restored.condition_results
        assert len(restored.condition_results["baseline"].runs) == 2

    def test_save_and_load(self):
        config = ExperimentConfig(
            name="save_test",
            description="",
            scenario_path="test.yaml",
            conditions=[ExperimentCondition("test")],
        )
        result = ExperimentResult(config=config, experiment_id="test_save_123")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_experiment(result, tmpdir)
            loaded = load_experiment("test_save_123", tmpdir)

            assert loaded.experiment_id == "test_save_123"
            assert loaded.config.name == "save_test"


class TestGenerateSummary:
    """Tests for generate_summary function."""

    def test_summary_includes_key_info(self):
        config = ExperimentConfig(
            name="summary_test",
            description="Test for summary",
            scenario_path="test.yaml",
            conditions=[
                ExperimentCondition("baseline"),
                ExperimentCondition("variant", modifications={"x": 1}),
            ],
            runs_per_condition=2,
        )
        result = ExperimentResult(config=config)
        result.condition_results["baseline"] = ConditionResults(
            condition=config.conditions[0],
            runs=[
                RunResult("baseline", 0, {}, duration_seconds=10.0),
                RunResult("baseline", 1, {}, error="failed"),
            ],
        )
        result.condition_results["variant"] = ConditionResults(
            condition=config.conditions[1],
            runs=[
                RunResult("variant", 0, {}, duration_seconds=5.0),
            ],
        )

        summary = generate_summary(result)

        assert "summary_test" in summary
        assert "baseline" in summary
        assert "variant" in summary
        assert "50.0%" in summary  # 1/2 success rate for baseline
