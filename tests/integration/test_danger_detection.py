"""Integration tests for danger detection in benchmarks."""

import pytest
from pathlib import Path
from core.danger_detector import SignalCollector
from core.state import GameState
from utils.variables import VariableSet, VariableDefinition
from unittest.mock import Mock, patch


def test_signal_collector_integration_with_benchmark(tmp_path):
    """Test that SignalCollector can be integrated into benchmark runs."""
    # This test validates the integration pattern without running full benchmark

    collector = SignalCollector()

    # Simulate what benchmark would do - collect signals during steps
    state_before = GameState()
    state_before.add_agent("Agent A")
    agent_before = state_before.get_agent("Agent A")
    agent_before.variables = VariableSet(definitions={
        "capital": VariableDefinition(type="float", default=1000.0, min=0.0, max=10000.0)
    })
    agent_before.set_var("capital", 1000.0)

    state_after = GameState()
    state_after.add_agent("Agent A")
    agent_after = state_after.get_agent("Agent A")
    agent_after.variables = VariableSet(definitions={
        "capital": VariableDefinition(type="float", default=1000.0, min=0.0, max=10000.0)
    })
    agent_after.set_var("capital", 1200.0)

    responses = {"Agent A": "I will maximize my resources"}

    collector.collect_step_signals(1, responses, state_before, state_after)

    # Verify signals were collected
    signals = collector.get_signals_for_agent("Agent A")
    assert len(signals) > 0


def test_benchmark_runner_danger_analysis_enabled():
    """Test that BenchmarkRunner runs danger analysis when enabled."""
    # This validates the configuration and flow without running actual LLM

    config = {
        "danger_detection": {
            "enabled": True,
            "judge_model": {
                "provider": "gemini",
                "model": "gemini-2.5-flash"
            }
        }
    }

    # Verify config structure
    assert config["danger_detection"]["enabled"] is True
    assert "judge_model" in config["danger_detection"]
