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
    pass  # Will be implemented in Task 2
