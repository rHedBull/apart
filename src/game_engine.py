from state import GameState


class GameEngine:
    """Manages game state and generates messages for agents."""

    def __init__(self, config: dict):
        self.config = config
        self.state = self._initialize_state()
        self.current_step = 0

    def _initialize_state(self) -> GameState:
        """Initialize the game state from configuration."""
        game_config = self.config.get("game_state", {})
        return GameState(
            resources=game_config.get("initial_resources", 100),
            difficulty=game_config.get("difficulty", "normal"),
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
            agent_state = self.state.add_agent(agent_name)

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
