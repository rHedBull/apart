from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScriptedEvent:
    """A guaranteed event that must occur at a specific step."""
    step: int
    type: str
    description: str


@dataclass
class EngineOutput:
    """Output from SimulatorAgent LLM call."""
    state_updates: Dict[str, Any]  # {global_vars: {...}, agent_vars: {agent_name: {...}}}
    events: List[Dict[str, Any]]
    agent_messages: Dict[str, str]  # {agent_name: message}
    reasoning: str


@dataclass
class ConstraintHit:
    """Record of a variable hitting min/max constraint."""
    agent_name: Optional[str]  # None for global vars
    var_name: str
    attempted_value: Union[int, float]
    clamped_value: Union[int, float]
    constraint_type: str  # "min" or "max"


@dataclass
class StepRecord:
    """Record of a single simulation step for history window."""
    step_number: int
    changes: Dict[str, Any]  # Only changed variables
    events: List[Dict[str, Any]]
    agent_responses: Dict[str, str]
    reasoning: str
    constraint_hits: List[ConstraintHit]
