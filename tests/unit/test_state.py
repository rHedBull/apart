import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from core.state import AgentState, GameState


class TestAgentState:
    """Tests for AgentState model."""

    def test_agent_state_initialization(self):
        """Test creating a new agent state."""
        agent = AgentState(name="TestAgent")
        assert agent.name == "TestAgent"
        assert agent.responses == []
        assert agent.active is True
        assert agent.resources == 0
        assert agent.custom_data == {}

    def test_custom_initialization(self):
        """Test creating agent with custom values."""
        agent = AgentState(
            name="CustomAgent",
            resources=50,
            custom_data={"level": 5}
        )
        assert agent.name == "CustomAgent"
        assert agent.resources == 50
        assert agent.custom_data["level"] == 5


class TestGameState:
    """Tests for GameState model."""

    def test_game_state_initialization(self):
        """Test creating a new game state."""
        state = GameState()
        assert state.round == 0
        assert state.events == []
        assert state.agents == {}
        assert state.resources == 100
        assert state.difficulty == "normal"

    def test_add_agent(self):
        """Test adding agents to game state."""
        state = GameState()
        agent = state.add_agent("Agent1", resources=10)
        assert "Agent1" in state.agents
        assert state.agents["Agent1"].name == "Agent1"
        assert state.agents["Agent1"].resources == 10

    def test_get_agent(self):
        """Test retrieving agents from game state."""
        state = GameState()
        state.add_agent("Agent1")
        agent = state.get_agent("Agent1")
        assert agent is not None
        assert agent.name == "Agent1"

        non_existent = state.get_agent("NonExistent")
        assert non_existent is None

    def test_advance_round(self):
        """Test advancing game rounds."""
        state = GameState()
        assert state.round == 0
        state.advance_round()
        assert state.round == 1
        state.advance_round()
        assert state.round == 2

    def test_custom_initialization(self):
        """Test creating game state with custom values."""
        state = GameState(
            resources=500,
            difficulty="expert",
            custom_data={"mode": "competitive"}
        )
        assert state.resources == 500
        assert state.difficulty == "expert"
        assert state.custom_data["mode"] == "competitive"
