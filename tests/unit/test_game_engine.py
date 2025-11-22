import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from core.game_engine import GameEngine


class TestGameEngine:
    """Tests for simplified GameEngine (state management only)."""

    def test_initialization_default(self):
        """Test game engine initialization with minimal config."""
        config = {
            "global_vars": {},
            "agent_vars": {}
        }
        engine = GameEngine(config)
        assert engine.state.resources == 100
        assert engine.state.difficulty == "normal"

    def test_get_global_var(self):
        """Test getting global variable."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {}
        }
        engine = GameEngine(config)
        assert engine.get_global_var("tension") == 0.5

    def test_set_global_var(self):
        """Test setting global variable."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {}
        }
        engine = GameEngine(config)
        engine.set_global_var("tension", 0.8)
        assert engine.get_global_var("tension") == 0.8

    def test_get_agent_var(self):
        """Test getting agent variable."""
        config = {
            "global_vars": {},
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        # Initialize agent
        engine.initialize_agent("Agent A", {})
        assert engine.get_agent_var("Agent A", "health") == 100

    def test_set_agent_var(self):
        """Test setting agent variable."""
        config = {
            "global_vars": {},
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        engine.initialize_agent("Agent A", {})
        engine.set_agent_var("Agent A", "health", 95)
        assert engine.get_agent_var("Agent A", "health") == 95

    def test_apply_state_updates(self):
        """Test applying state updates from SimulatorAgent."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        engine.initialize_agent("Agent A", {})

        updates = {
            "global_vars": {"tension": 0.8},
            "agent_vars": {"Agent A": {"health": 95}}
        }

        engine.apply_state_updates(updates)

        assert engine.get_global_var("tension") == 0.8
        assert engine.get_agent_var("Agent A", "health") == 95

    def test_get_current_state(self):
        """Test getting current state snapshot."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        engine.initialize_agent("Agent A", {})

        state = engine.get_current_state()

        assert "global_vars" in state
        assert "agent_vars" in state
        assert state["global_vars"]["tension"] == 0.5
        assert state["agent_vars"]["Agent A"]["health"] == 100
