import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from game_engine import GameEngine
from state import GameState


class TestGameEngine:
    """Tests for GameEngine."""

    def test_initialization_default(self):
        """Test game engine initialization with default config."""
        config = {}
        engine = GameEngine(config)
        assert engine.state.resources == 100
        assert engine.state.difficulty == "normal"
        assert engine.current_step == 0

    def test_initialization_from_config(self):
        """Test game engine initialization from config."""
        config = {
            "game_state": {
                "initial_resources": 200,
                "difficulty": "hard"
            }
        }
        engine = GameEngine(config)
        assert engine.state.resources == 200
        assert engine.state.difficulty == "hard"

    def test_get_message_for_agent(self):
        """Test message generation for agents."""
        config = {"orchestrator_message": "Test message"}
        engine = GameEngine(config)

        message = engine.get_message_for_agent("Agent1")
        assert "Test message" in message
        assert "Round 0" in message
        assert "Total events: 1" in message
        assert engine.current_step == 1
        assert len(engine.state.events) == 1

    def test_process_agent_response(self):
        """Test processing agent responses."""
        config = {}
        engine = GameEngine(config)

        engine.process_agent_response("Agent1", "Response 1")
        agent = engine.state.get_agent("Agent1")
        assert agent is not None
        assert len(agent.responses) == 1
        assert agent.responses[0] == "Response 1"

        engine.process_agent_response("Agent1", "Response 2")
        assert len(agent.responses) == 2

    def test_advance_round(self):
        """Test advancing rounds."""
        config = {}
        engine = GameEngine(config)
        assert engine.state.round == 0
        engine.advance_round()
        assert engine.state.round == 1

    def test_get_state(self):
        """Test getting current state."""
        config = {}
        engine = GameEngine(config)
        state = engine.get_state()
        assert isinstance(state, GameState)
        assert state.round == 0

    def test_is_game_over(self):
        """Test game over condition."""
        config = {}
        engine = GameEngine(config)
        # Currently always returns False
        assert engine.is_game_over() is False

    def test_full_simulation_cycle(self):
        """Test a complete simulation cycle."""
        config = {
            "orchestrator_message": "Proceed",
            "game_state": {
                "initial_resources": 150,
                "difficulty": "normal"
            }
        }
        engine = GameEngine(config)

        # Simulate 3 rounds with 2 agents
        for round_num in range(3):
            for agent_name in ["Alpha", "Beta"]:
                message = engine.get_message_for_agent(agent_name)
                response = f"{agent_name} response {round_num + 1}"
                engine.process_agent_response(agent_name, response)
            engine.advance_round()

        # Verify final state
        assert engine.state.round == 3
        assert len(engine.state.events) == 6  # 3 rounds * 2 agents
        assert len(engine.state.agents) == 2
        assert engine.state.get_agent("Alpha") is not None
        assert engine.state.get_agent("Beta") is not None
        assert len(engine.state.get_agent("Alpha").responses) == 3
