import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import pytest
from core.simulator_agent import SimulatorAgent
from core.game_engine import GameEngine
from llm.llm_provider import LLMProvider


class MockSimulatorLLM(LLMProvider):
    """Mock LLM for integration testing."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.dumps(response)


class TestSimulatorIntegration:
    """Integration tests for SimulatorAgent + GameEngine."""

    def test_full_simulation_flow(self):
        """Test complete simulation from initialization through multiple steps."""
        config = {
            "engine": {
                "provider": "mock",
                "model": "test",
                "system_prompt": "Test",
                "simulation_plan": "Test"
            },
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [
                {"name": "Agent A"},
                {"name": "Agent B"}
            ]
        }

        game_engine = GameEngine(config)

        # Mock responses for initialization + 3 steps
        mock_responses = [
            # Step 0: Initialization
            {
                "state_updates": {"global_vars": {}, "agent_vars": {}},
                "events": [],
                "agent_messages": {
                    "Agent A": "Welcome Agent A",
                    "Agent B": "Welcome Agent B"
                },
                "reasoning": "Initial setup"
            },
            # Step 1
            {
                "state_updates": {
                    "global_vars": {"tension": 0.6},
                    "agent_vars": {"Agent A": {"health": 95}}
                },
                "events": [{"type": "combat", "description": "Skirmish"}],
                "agent_messages": {
                    "Agent A": "You took damage",
                    "Agent B": "Tensions rise"
                },
                "reasoning": "Combat occurred"
            },
            # Step 2
            {
                "state_updates": {
                    "global_vars": {"tension": 0.7},
                    "agent_vars": {"Agent B": {"health": 90}}
                },
                "events": [],
                "agent_messages": {
                    "Agent A": "Conflict escalates",
                    "Agent B": "You suffer damage"
                },
                "reasoning": "Escalation"
            },
            # Step 3
            {
                "state_updates": {
                    "global_vars": {"tension": 0.8},
                    "agent_vars": {
                        "Agent A": {"health": 85},
                        "Agent B": {"health": 80}
                    }
                },
                "events": [{"type": "full_battle", "description": "Major combat"}],
                "agent_messages": {
                    "Agent A": "Major battle",
                    "Agent B": "Heavy fighting"
                },
                "reasoning": "Full war"
            }
        ]

        mock_llm = MockSimulatorLLM(mock_responses)

        simulator = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        # Initialize
        messages = simulator.initialize_simulation(["Agent A", "Agent B"])
        assert "Agent A" in messages
        assert "Agent B" in messages

        # Run 3 steps
        for step in range(1, 4):
            responses = {
                "Agent A": f"Agent A action {step}",
                "Agent B": f"Agent B action {step}"
            }
            messages = simulator.process_step(step, responses)

            assert "Agent A" in messages
            assert "Agent B" in messages

        # Verify final state
        assert game_engine.get_global_var("tension") == 0.8
        assert game_engine.get_agent_var("Agent A", "health") == 85
        assert game_engine.get_agent_var("Agent B", "health") == 80

        # Verify history
        assert len(simulator.step_history) == 3

    def test_constraint_enforcement(self):
        """Test that constraints are enforced and fed back."""
        config = {
            "engine": {
                "provider": "mock",
                "model": "test",
                "system_prompt": "Test",
                "simulation_plan": "Test"
            },
            "global_vars": {},
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }

        game_engine = GameEngine(config)

        # Response tries to set health below min
        mock_responses = [
            {
                "state_updates": {
                    "global_vars": {},
                    "agent_vars": {}
                },
                "events": [],
                "agent_messages": {"Agent A": "Start"},
                "reasoning": "Init"
            },
            {
                "state_updates": {
                    "global_vars": {},
                    "agent_vars": {"Agent A": {"health": -10}}  # Below min!
                },
                "events": [],
                "agent_messages": {"Agent A": "You died"},
                "reasoning": "Death"
            }
        ]

        mock_llm = MockSimulatorLLM(mock_responses)

        simulator = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        simulator.initialize_simulation(["Agent A"])
        simulator.process_step(1, {"Agent A": "I die"})

        # Should be clamped to 0
        assert game_engine.get_agent_var("Agent A", "health") == 0

        # Should have constraint hit in feedback
        assert len(simulator.constraint_feedback) == 1
        assert simulator.constraint_feedback[0].clamped_value == 0
