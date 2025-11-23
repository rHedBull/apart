"""Unit tests for MetricsCollector."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from core.metrics_collector import MetricsCollector, RunMetrics


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector("test-model", "test-provider")

        assert collector.metrics.model_name == "test-model"
        assert collector.metrics.provider == "test-provider"
        assert collector.metrics.completed is False
        assert collector.metrics.error_count == 0
        assert len(collector.metrics.step_times) == 0

    def test_step_timing(self):
        """Test step timing collection."""
        collector = MetricsCollector("test-model", "test-provider")

        # Start and end a step
        collector.start_step(1)
        import time
        time.sleep(0.01)  # Small delay
        collector.end_step(1)

        assert len(collector.metrics.step_times) == 1
        assert collector.metrics.step_times[0] > 0

    def test_error_tracking(self):
        """Test error tracking."""
        collector = MetricsCollector("test-model", "test-provider")

        collector.start_step(1)
        errors = ["Error 1", "Error 2"]
        collector.end_step(1, errors=errors)

        assert collector.metrics.error_count == 2
        assert collector.metrics.error_messages == errors
        assert 1 in collector.metrics.step_failures

    def test_variable_changes_tracking(self):
        """Test variable changes tracking."""
        collector = MetricsCollector("test-model", "test-provider",
                                     metrics_config={"quality": {"enabled": True}})

        state_snapshot = {
            "global_vars": {"round": 1},
            "agent_vars": {
                "Agent A": {"score": 100}
            }
        }

        collector.start_step(1)
        collector.end_step(1, state_snapshot=state_snapshot)

        assert len(collector.metrics.variable_changes) == 1
        assert collector.metrics.variable_changes[0]["step"] == 1
        assert collector.metrics.variable_changes[0]["global_vars"]["round"] == 1

    def test_conversation_tracking(self):
        """Test conversation turn recording."""
        collector = MetricsCollector("test-model", "test-provider")

        agent_messages = {
            "Agent A": "What should I do?",
            "Agent B": "How should I respond?"
        }
        agent_responses = {
            "Agent A": "I will take action X",
            "Agent B": "I will do Y"
        }

        collector.record_conversation_turn(1, agent_messages, agent_responses)

        assert len(collector.metrics.conversation) == 1
        turn = collector.metrics.conversation[0]
        assert turn["step"] == 1
        assert len(turn["exchanges"]) == 2
        assert turn["exchanges"][0]["agent"] == "Agent A"
        assert turn["exchanges"][0]["message_to_agent"] == "What should I do?"
        assert turn["exchanges"][0]["response_from_agent"] == "I will take action X"

    def test_completion_recording(self):
        """Test completion recording."""
        collector = MetricsCollector("test-model", "test-provider")

        collector.start_step(1)
        import time
        time.sleep(0.01)
        collector.end_step(1)

        collector.record_completion(completed=True)

        assert collector.metrics.completed is True
        assert collector.metrics.end_time is not None
        assert collector.metrics.total_time > 0
        assert collector.metrics.avg_step_time > 0

    def test_final_state_recording(self):
        """Test final state recording."""
        collector = MetricsCollector("test-model", "test-provider",
                                     metrics_config={"custom": {"enabled": True,
                                                                "track_variables": ["*.score"]}})

        final_snapshot = {
            "game_state": {"round": 5},
            "global_vars": {"market": 1.0},
            "agent_vars": {
                "Agent A": {"score": 150},
                "Agent B": {"score": 120}
            }
        }

        collector.record_final_state(final_snapshot)

        assert collector.metrics.final_state == final_snapshot
        assert "*.score" in collector.metrics.custom_metrics
        assert collector.metrics.custom_metrics["*.score"]["Agent A"] == 150

    def test_constraint_violations(self):
        """Test constraint violation detection."""
        collector = MetricsCollector("test-model", "test-provider",
                                     metrics_config={"quality": {"enabled": True}})

        # State with constraint violation (negative capital)
        bad_state = {
            "global_vars": {},
            "agent_vars": {
                "Agent A": {"capital": -100}  # Violation!
            }
        }

        collector.start_step(1)
        collector.end_step(1, state_snapshot=bad_state)

        assert collector.metrics.constraint_violations > 0

    def test_to_dict_serialization(self):
        """Test metrics serialization to dict."""
        collector = MetricsCollector("test-model", "test-provider")
        collector.start_step(1)
        collector.end_step(1)
        collector.record_completion(completed=True)

        metrics_dict = collector.get_metrics().to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["model_name"] == "test-model"
        assert metrics_dict["provider"] == "test-provider"
        assert metrics_dict["completed"] is True
        assert "start_time" in metrics_dict
        assert "end_time" in metrics_dict


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_default_initialization(self):
        """Test RunMetrics default values."""
        metrics = RunMetrics(
            model_name="test",
            provider="test-provider",
            start_time="2025-01-01T00:00:00"
        )

        assert metrics.total_time == 0.0
        assert metrics.step_times == []
        assert metrics.completed is False
        assert metrics.conversation == []
        assert metrics.custom_metrics == {}

    def test_to_dict_includes_conversation(self):
        """Test that to_dict includes conversation data."""
        metrics = RunMetrics(
            model_name="test",
            provider="test-provider",
            start_time="2025-01-01T00:00:00"
        )
        metrics.conversation = [{"step": 1, "exchanges": []}]

        data = metrics.to_dict()

        assert "conversation" in data
        assert len(data["conversation"]) == 1

    def test_run_metrics_with_danger_scores(self):
        """Test RunMetrics includes danger_scores field."""
        from core.danger_detector import DangerScores

        danger_scores = {
            "Agent A": DangerScores(
                agent_name="Agent A",
                run_name="test",
                power_seeking=7,
                deception=3,
                rule_exploitation=2,
                reasoning="Test"
            )
        }

        metrics = RunMetrics(
            model_name="test_model",
            provider="test_provider",
            start_time="2025-11-23T10:00:00",
            danger_scores=danger_scores
        )

        assert metrics.danger_scores == danger_scores
        assert "Agent A" in metrics.danger_scores

        # Test to_dict includes danger scores
        data = metrics.to_dict()
        assert "danger_scores" in data
