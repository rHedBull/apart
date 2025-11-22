import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import pytest
from core.simulator_agent import SimulatorAgent
from core.game_engine import GameEngine
from llm.llm_provider import LLMProvider


class MockSimulatorLLM(LLMProvider):
    """Mock LLM for testing SimulatorAgent."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.last_prompt = None

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        self.last_prompt = prompt
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.dumps(response)


class TestSimulatorAgent:
    """Tests for SimulatorAgent."""

    def test_initialization(self):
        """Test SimulatorAgent initialization."""
        config = {
            "global_vars": {},
            "agent_vars": {},
            "agents": []
        }
        game_engine = GameEngine(config)
        mock_llm = MockSimulatorLLM([])

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test prompt",
            simulation_plan="Test plan",
            realism_guidelines="Test guidelines",
            scripted_events=[],
            context_window_size=5
        )

        assert agent.llm_provider == mock_llm
        assert agent.game_engine == game_engine
        assert agent.context_window_size == 5
        assert len(agent.step_history) == 0

    def test_initialize_simulation(self):
        """Test simulation initialization."""
        config = {
            "global_vars": {},
            "agent_vars": {},
            "agents": [{"name": "Agent A"}, {"name": "Agent B"}]
        }
        game_engine = GameEngine(config)

        mock_response = {
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "agent_messages": {
                "Agent A": "Welcome Agent A",
                "Agent B": "Welcome Agent B"
            },
            "reasoning": "Initial setup"
        }
        mock_llm = MockSimulatorLLM([mock_response])

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        messages = agent.initialize_simulation(["Agent A", "Agent B"])

        assert "Agent A" in messages
        assert "Agent B" in messages
        assert messages["Agent A"] == "Welcome Agent A"
        assert messages["Agent B"] == "Welcome Agent B"

    def test_process_step_basic(self):
        """Test processing a single step."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        game_engine = GameEngine(config)
        game_engine.initialize_agent("Agent A", {})

        mock_response = {
            "state_updates": {
                "global_vars": {"tension": 0.6},
                "agent_vars": {"Agent A": {"health": 95}}
            },
            "events": [{"type": "combat", "description": "Minor skirmish"}],
            "agent_messages": {"Agent A": "You took damage"},
            "reasoning": "Combat occurred"
        }
        mock_llm = MockSimulatorLLM([mock_response])

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        agent_responses = {"Agent A": "I attack"}
        messages = agent.process_step(1, agent_responses)

        assert messages["Agent A"] == "You took damage"
        assert game_engine.get_global_var("tension") == 0.6
        assert game_engine.get_agent_var("Agent A", "health") == 95
        assert len(agent.step_history) == 1

    def test_context_window_limiting(self):
        """Test that history is limited to window size."""
        config = {
            "global_vars": {},
            "agent_vars": {},
            "agents": [{"name": "Agent A"}]
        }
        game_engine = GameEngine(config)
        game_engine.initialize_agent("Agent A", {})

        mock_response = {
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "agent_messages": {"Agent A": "Message"},
            "reasoning": "Step"
        }
        mock_llm = MockSimulatorLLM([mock_response] * 10)

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[],
            context_window_size=3
        )

        # Process 10 steps
        for i in range(1, 11):
            agent.process_step(i, {"Agent A": "Response"})

        # Should only keep last 3 steps
        assert len(agent.step_history) == 3
        assert agent.step_history[0].step_number == 8
        assert agent.step_history[-1].step_number == 10
