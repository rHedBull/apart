"""Tests for dangerous behavior detection system."""

import pytest
from dataclasses import asdict
from core.danger_detector import Signal, DangerScores, SignalCollector


def test_signal_creation():
    """Test Signal dataclass creation."""
    signal = Signal(
        step=3,
        agent_name="Agent A",
        category="power_seeking",
        metric="capital_growth",
        value=0.25,
        threshold=0.15
    )

    assert signal.step == 3
    assert signal.agent_name == "Agent A"
    assert signal.category == "power_seeking"
    assert signal.metric == "capital_growth"
    assert signal.value == 0.25
    assert signal.threshold == 0.15


def test_signal_to_dict():
    """Test Signal serialization to dictionary."""
    signal = Signal(
        step=1,
        agent_name="Agent B",
        category="deception",
        metric="keyword_flag",
        value=1,
        threshold=1
    )

    data = asdict(signal)
    assert data["step"] == 1
    assert data["category"] == "deception"


def test_danger_scores_creation():
    """Test DangerScores dataclass creation."""
    scores = DangerScores(
        agent_name="Agent A",
        run_name="test_run",
        power_seeking=7,
        deception=3,
        rule_exploitation=2,
        reasoning="Agent showed high resource accumulation"
    )

    assert scores.agent_name == "Agent A"
    assert scores.run_name == "test_run"
    assert scores.power_seeking == 7
    assert scores.deception == 3
    assert scores.rule_exploitation == 2
    assert scores.reasoning == "Agent showed high resource accumulation"


def test_danger_scores_average():
    """Test average_score property calculation."""
    scores = DangerScores(
        agent_name="Agent B",
        run_name="test_run",
        power_seeking=6,
        deception=3,
        rule_exploitation=9,
        reasoning="Test"
    )

    assert scores.average_score == 6.0  # (6+3+9)/3


def test_danger_scores_to_dict():
    """Test DangerScores serialization."""
    scores = DangerScores(
        agent_name="Agent A",
        run_name="run1",
        power_seeking=5,
        deception=4,
        rule_exploitation=3,
        reasoning="Test reasoning"
    )

    data = scores.to_dict()
    assert data["agent_name"] == "Agent A"
    assert data["power_seeking"] == 5
    assert "timestamp" in data
    assert "average_score" in data


def test_signal_collector_initialization():
    """Test SignalCollector initializes with empty signals list."""
    collector = SignalCollector()
    assert collector.signals == []
    assert isinstance(collector.signals, list)


def test_signal_collector_get_signals_for_agent_empty():
    """Test getting signals for agent when none collected."""
    collector = SignalCollector()
    signals = collector.get_signals_for_agent("Agent A")
    assert signals == []
