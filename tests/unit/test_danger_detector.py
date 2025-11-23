"""Tests for dangerous behavior detection system."""

import pytest
from dataclasses import asdict
from core.danger_detector import Signal, DangerScores


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
