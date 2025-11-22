from typing import Any, Dict, Optional
from core.state import GameState
from utils.config_parser import create_variable_set, create_variable_set_with_overrides, validate_config


class GameEngine:
    """Manages game state - pure state management, no simulation logic."""

    def __init__(self, config: dict):
        # Validate config before using it
        validate_config(config)

        self.config = config
        self.global_var_definitions = config.get("global_vars", {})
        self.agent_var_definitions = config.get("agent_vars", {})
        self.agent_configs = {agent["name"]: agent for agent in config.get("agents", [])}
        self.state = self._initialize_state()

    def _initialize_state(self) -> GameState:
        """Initialize the game state from configuration."""
        game_config = self.config.get("game_state", {})

        # Create global variables from config
        global_vars = create_variable_set(self.config.get("global_vars"))

        return GameState(
            resources=game_config.get("initial_resources", 100),
            difficulty=game_config.get("difficulty", "normal"),
            variables=global_vars,
        )

    def initialize_agent(self, agent_name: str, variable_overrides: Optional[Dict[str, Any]] = None):
        """Initialize an agent's state with variables."""
        if self.state.get_agent(agent_name) is not None:
            return  # Already initialized

        agent_config = self.agent_configs.get(agent_name, {})
        overrides = variable_overrides or agent_config.get("variables")

        self.state.add_agent(
            agent_name,
            variables=create_variable_set_with_overrides(
                self.config.get("agent_vars"),
                overrides
            )
        )

    def get_global_var(self, var_name: str) -> Any:
        """Get global variable value."""
        return self.state.get_var(var_name)

    def set_global_var(self, var_name: str, value: Any):
        """Set global variable with type validation."""
        self.state.set_var(var_name, value)

    def get_agent_var(self, agent_name: str, var_name: str) -> Any:
        """Get agent variable value."""
        agent_state = self.state.get_agent(agent_name)
        if agent_state is None:
            raise ValueError(f"Agent '{agent_name}' not initialized")
        return agent_state.get_var(var_name)

    def set_agent_var(self, agent_name: str, var_name: str, value: Any):
        """Set agent variable with type validation."""
        agent_state = self.state.get_agent(agent_name)
        if agent_state is None:
            raise ValueError(f"Agent '{agent_name}' not initialized")
        agent_state.set_var(var_name, value)

    def apply_state_updates(self, updates: Dict[str, Any]):
        """Apply state updates from SimulatorAgent."""
        # Apply global variable updates
        for var_name, value in updates.get("global_vars", {}).items():
            self.set_global_var(var_name, value)

        # Apply agent variable updates
        for agent_name, vars_dict in updates.get("agent_vars", {}).items():
            # Ensure agent is initialized
            if self.state.get_agent(agent_name) is None:
                self.initialize_agent(agent_name)

            for var_name, value in vars_dict.items():
                self.set_agent_var(agent_name, var_name, value)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state snapshot."""
        global_vars = self.state.variables.to_dict()

        agent_vars = {}
        for agent_name, agent_state in self.state.agents.items():
            agent_vars[agent_name] = agent_state.variables.to_dict()

        return {
            "global_vars": global_vars,
            "agent_vars": agent_vars
        }

    def get_state(self) -> GameState:
        """Get the full GameState object."""
        return self.state

    def advance_round(self):
        """Advance to the next round."""
        self.state.advance_round()

    def get_state_snapshot(self) -> dict:
        """Get a snapshot for persistence (backward compatibility)."""
        game_state = {
            "resources": self.state.resources,
            "difficulty": self.state.difficulty,
            "round": self.state.round
        }

        global_vars = self.state.variables.to_dict()

        agent_vars = {}
        for agent_name, agent_state in self.state.agents.items():
            agent_vars[agent_name] = agent_state.variables.to_dict()

        return {
            "game_state": game_state,
            "global_vars": global_vars,
            "agent_vars": agent_vars
        }
