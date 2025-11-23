from pydantic import BaseModel, Field
from typing import Any
from utils.variables import VariableSet


class AgentState(BaseModel):
    """Represents the state of a single agent."""

    name: str
    responses: list[str] = Field(default_factory=list)
    active: bool = True
    resources: int = 0
    custom_data: dict[str, Any] = Field(default_factory=dict)
    variables: VariableSet = Field(default_factory=VariableSet)

    def add_response(self, response: str):
        """Add a response to the agent's history."""
        self.responses.append(response)

    def deactivate(self):
        """Mark agent as inactive."""
        self.active = False

    def get_var(self, name: str) -> int | float | bool:
        """Get a variable value."""
        return self.variables.get(name)

    def set_var(self, name: str, value: Any) -> None:
        """Set a variable value with validation."""
        self.variables.set(name, value)

    def update_vars(self, updates: dict[str, Any]) -> None:
        """Update multiple variables at once."""
        self.variables.update(updates)


class GameState(BaseModel):
    """Represents the full game state including all agents."""

    round: int = 0
    events: list[str] = Field(default_factory=list)
    agents: dict[str, AgentState] = Field(default_factory=dict)
    resources: int = 100
    difficulty: str = "normal"
    custom_data: dict[str, Any] = Field(default_factory=dict)
    variables: VariableSet = Field(default_factory=VariableSet)

    def add_event(self, event: str):
        """Add an event to the game history."""
        self.events.append(event)

    def get_agent(self, name: str) -> AgentState | None:
        """Get agent state by name."""
        return self.agents.get(name)

    def add_agent(self, name: str, **kwargs) -> AgentState:
        """Add a new agent to the game state."""
        agent_state = AgentState(name=name, **kwargs)
        self.agents[name] = agent_state
        return agent_state

    def advance_round(self):
        """Advance to the next round."""
        self.round += 1

    def get_var(self, name: str) -> int | float | bool:
        """Get a global variable value."""
        return self.variables.get(name)

    def set_var(self, name: str, value: Any) -> None:
        """Set a global variable value with validation."""
        self.variables.set(name, value)

    def update_vars(self, updates: dict[str, Any]) -> None:
        """Update multiple global variables at once."""
        self.variables.update(updates)

    def to_summary(self) -> dict:
        """Get a summary of the game state."""
        return {
            "round": self.round,
            "total_events": len(self.events),
            "active_agents": sum(1 for a in self.agents.values() if a.active),
            "total_agents": len(self.agents),
            "resources": self.resources,
            "difficulty": self.difficulty,
            "global_vars": self.variables.to_dict(),
        }
