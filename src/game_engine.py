from state import GameState
from config_parser import create_variable_set, validate_config


class GameEngine:
    """Manages game state and generates messages for agents."""

    def __init__(self, config: dict):
        # Validate config before using it
        validate_config(config)

        self.config = config
        self.agent_var_definitions = create_variable_set(config.get("agent_vars"))
        self.state = self._initialize_state()
        self.current_step = 0

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

    def get_message_for_agent(self, agent_name: str) -> str:
        """Generate a context-aware message for a specific agent based on current game state."""
        self.current_step += 1

        # Add event to state
        event = f"Step {self.current_step}: Request to {agent_name}"
        self.state.add_event(event)

        # Generate message based on game state
        base_message = self.config.get("orchestrator_message", "Continue")
        state_info = f"[Round {self.state.round}, Total events: {len(self.state.events)}]"

        return f"{base_message} {state_info}"

    def process_agent_response(self, agent_name: str, response: str):
        """Process agent response and update game state."""
        agent_state = self.state.get_agent(agent_name)
        if agent_state is None:
            # Initialize agent with variable definitions from config
            agent_state = self.state.add_agent(
                agent_name,
                variables=create_variable_set(self.config.get("agent_vars"))
            )

        agent_state.add_response(response)

    def advance_round(self):
        """Advance to the next round of the game."""
        self.state.advance_round()

    def get_state(self) -> GameState:
        """Get the current game state."""
        return self.state

    def is_game_over(self) -> bool:
        """Check if the game/simulation should end."""
        # For now, just relies on max_steps in orchestrator
        # Can add custom end conditions here later
        return False
