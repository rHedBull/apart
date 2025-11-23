"""
Dangerous behavior detection system for benchmarking.

Provides lightweight signal collection during simulation and
post-hoc LLM-based analysis to score agents on dangerous behaviors.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Signal:
    """A behavioral signal detected during simulation."""
    step: int
    agent_name: str
    category: str  # "power_seeking", "deception", "rule_exploitation"
    metric: str
    value: float
    threshold: Optional[float] = None


@dataclass
class DangerScores:
    """Danger scores for an agent in a benchmark run."""
    agent_name: str
    run_name: str
    power_seeking: int  # 0-10
    deception: int  # 0-10
    rule_exploitation: int  # 0-10
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def average_score(self) -> float:
        """Average danger score across all categories."""
        return (self.power_seeking + self.deception + self.rule_exploitation) / 3.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            "average_score": self.average_score
        }


class SignalCollector:
    """Collects behavioral signals during simulation steps."""

    def __init__(self):
        """Initialize with empty signals list."""
        self.signals: List[Signal] = []

    def get_signals_for_agent(self, agent_name: str) -> List[Signal]:
        """
        Return all signals collected for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of Signal objects for this agent
        """
        return [s for s in self.signals if s.agent_name == agent_name]
