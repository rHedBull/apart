"""
API models for the dashboard server.

These Pydantic models define the request/response schemas for the REST API.
"""

from pydantic import BaseModel, Field
from typing import Any
from enum import Enum


class SimulationStatus(str, Enum):
    """Status of a simulation run."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"  # Worker died mid-run, job back in queue


class AgentInfo(BaseModel):
    """Information about an agent in a simulation."""
    name: str
    llm_provider: str | None = None
    llm_model: str | None = None


class SimulationSummary(BaseModel):
    """Summary information about a simulation run."""
    run_id: str
    status: SimulationStatus
    scenario_name: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    current_step: int = 0
    max_steps: int | None = None
    agent_count: int = 0


class SimulationDetails(SimulationSummary):
    """Detailed information about a simulation run."""
    agents: list[AgentInfo] = Field(default_factory=list)
    error_message: str | None = None


class JobPriority(str, Enum):
    """Priority level for job queue (when enabled)."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class StartSimulationRequest(BaseModel):
    """Request to start a new simulation."""
    scenario_path: str
    run_id: str | None = None  # Optional run ID prefix (UUID suffix always appended for uniqueness)
    priority: JobPriority | None = JobPriority.NORMAL  # Job queue priority


class StartSimulationResponse(BaseModel):
    """Response after starting a simulation."""
    run_id: str
    status: SimulationStatus
    message: str


class DangerSignal(BaseModel):
    """A danger signal detected during simulation."""
    category: str  # power-seeking, deception, rule-exploit, etc.
    description: str
    confidence: float
    step: int
    agent_name: str | None = None
    timestamp: str


class DangerSummary(BaseModel):
    """Summary of danger signals for a run."""
    run_id: str
    total_signals: int
    by_category: dict[str, int] = Field(default_factory=dict)
    signals: list[DangerSignal] = Field(default_factory=list)


class PauseSimulationResponse(BaseModel):
    """Response after requesting simulation pause."""
    run_id: str
    status: str  # "pause_requested"
    message: str


class ResumeSimulationResponse(BaseModel):
    """Response after resuming a simulation."""
    run_id: str
    status: str  # "resumed"
    resuming_from_step: int
    message: str
