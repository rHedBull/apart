"""Integration tests for LLM agent scenarios."""

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.orchestrator import Orchestrator
from llm.mock_provider import MockLLMProvider


class TestLLMScenarioIntegration:
    """Test complete LLM scenario execution."""

    def test_llm_scenario_with_mock_provider(self, tmp_path, mock_engine_llm_provider):
        """Test LLM scenario runs successfully with mock provider."""
        # Create a test scenario with LLM agent
        scenario_content = """
max_steps: 2
orchestrator_message: "What is your strategy?"

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"

game_state:
  initial_resources: 100
  difficulty: "normal"

global_vars:
  interest_rate:
    type: float
    default: 0.05

agent_vars:
  economic_strength:
    type: float
    default: 1000.0

agents:
  - name: "AI Strategist"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: "You are a strategic advisor."
    response_template: "Analyzing situation"  # Fallback
    variables:
      economic_strength: 1500.0
"""
        scenario_file = tmp_path / "test_llm_scenario.yaml"
        scenario_file.write_text(scenario_content)

        # Mock GeminiProvider to return a mock that's available
        mock_agent_provider = MockLLMProvider(responses=["Strategic response"])
        mock_agent_provider._available = True

        with patch('core.orchestrator.GeminiProvider', return_value=mock_agent_provider):
            # Run orchestrator with mocked providers
            orchestrator = Orchestrator(str(scenario_file), "test_llm_scenario", save_frequency=0, engine_llm_provider=mock_engine_llm_provider)

            # Verify agent was created with LLM provider
            assert len(orchestrator.agents) == 1
            agent = orchestrator.agents[0]
            assert agent.name == "AI Strategist"
            assert agent.llm_provider is not None
            assert agent.response_template == "Analyzing situation"

            # Run simulation
            orchestrator.run()

            # Verify it completed without errors
            assert agent.step_count == 2

    def test_llm_agent_without_template_fails_gracefully(self, tmp_path):
        """Test agent without template or LLM fails during initialization."""
        scenario_content = """
max_steps: 1
orchestrator_message: "Test"

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"

game_state:
  initial_resources: 100

global_vars: {}
agent_vars: {}

agents:
  - name: "No Response Method Agent"
    system_prompt: "Test"
    # NO response_template and NO llm config - should fail
"""
        scenario_file = tmp_path / "no_fallback.yaml"
        scenario_file.write_text(scenario_content)

        # Should fail during initialization because agent has neither template nor LLM
        with pytest.raises(ValueError, match="must have either response_template or llm_provider"):
            orchestrator = Orchestrator(str(scenario_file), "no_fallback", save_frequency=0)


class TestLLMAgentWithMockProvider:
    """Test LLM agents with mock provider injected."""

    def test_orchestrator_with_mocked_llm_agent(self, tmp_path):
        """Test we can inject mock provider for testing."""
        from core.agent import Agent
        from core.game_engine import GameEngine
        from utils.persistence import RunPersistence

        # Create simple config
        config = {
            "max_steps": 2,
            "orchestrator_message": "Test message",
            "game_state": {"initial_resources": 100},
            "global_vars": {},
            "agent_vars": {},
            "agents": []
        }

        # Create agent with mock provider directly
        mock_provider = MockLLMProvider(
            responses=["Strategic move A", "Strategic move B"]
        )

        agent = Agent(
            name="Mock LLM Agent",
            llm_provider=mock_provider,
            system_prompt="You are a test agent"
        )

        # Test agent responds correctly
        response1 = agent.respond("First message")
        response2 = agent.respond("Second message")

        assert response1 == "Strategic move A"
        assert response2 == "Strategic move B"
        assert mock_provider.call_count == 2


class TestLLMExampleScenario:
    """Test the provided llm_example.yaml scenario."""

    def test_llm_example_scenario_requires_api_key(self, mock_engine_llm_provider):
        """Test the llm_example.yaml scenario works with mocked providers."""
        scenario_path = "scenarios/llm_example.yaml"

        # Mock GeminiProvider for agents
        mock_agent_provider = MockLLMProvider(responses=["Strategic response"])
        mock_agent_provider._available = True

        with patch('core.orchestrator.GeminiProvider', return_value=mock_agent_provider):
            # Should succeed with mocked providers
            orchestrator = Orchestrator(scenario_path, "llm_example_test", save_frequency=0, engine_llm_provider=mock_engine_llm_provider)
            assert len(orchestrator.agents) > 0

    def test_llm_example_with_fallback_template(self, tmp_path, mock_engine_llm_provider):
        """Test LLM scenario works with mocked provider."""
        # Create scenario with template
        scenario_content = """
max_steps: 2
orchestrator_message: "Test"
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"
game_state:
  initial_resources: 100
global_vars: {}
agent_vars: {}

agents:
  - name: "AI Strategist"
    llm:
      provider: "gemini"
      model: "gemini-1.5-flash"
    system_prompt: "You are a strategist."
    response_template: "Analyzing situation"
"""
        scenario_file = tmp_path / "with_mock.yaml"
        scenario_file.write_text(scenario_content)

        # Mock GeminiProvider for agents
        mock_agent_provider = MockLLMProvider(responses=["Strategic response"])
        mock_agent_provider._available = True

        with patch('core.orchestrator.GeminiProvider', return_value=mock_agent_provider):
            # Should work with mocked provider
            orchestrator = Orchestrator(str(scenario_file), "with_mock", save_frequency=0, engine_llm_provider=mock_engine_llm_provider)
            assert len(orchestrator.agents) == 1

            # Run simulation
            orchestrator.run()
            assert orchestrator.agents[0].step_count == 2


class TestUnifiedLLMProviderIntegration:
    """Test UnifiedLLMProvider integration in Orchestrator."""

    def test_unified_provider_in_orchestrator(self, tmp_path, mock_engine_llm_provider):
        """Test UnifiedLLMProvider works in Orchestrator."""
        scenario_content = """
max_steps: 1
orchestrator_message: "Test"

engine:
  provider: "gemini"
  model: "gemini-2.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"

game_state:
  initial_resources: 100

global_vars:
  test_var:
    type: float
    default: 1.0

agent_vars:
  agent_var:
    type: float
    default: 1.0

agents:
  - name: "Test Agent"
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
    system_prompt: "Test"
    response_template: "Fallback"
"""
        scenario_file = tmp_path / "test_unified.yaml"
        scenario_file.write_text(scenario_content)

        from llm.mock_provider import MockLLMProvider
        mock_agent_provider = MockLLMProvider(responses=["Test response"])
        mock_agent_provider._available = True

        with patch('core.orchestrator.UnifiedLLMProvider', return_value=mock_agent_provider):
            orchestrator = Orchestrator(
                str(scenario_file),
                "test_unified",
                save_frequency=0,
                engine_llm_provider=mock_engine_llm_provider
            )

            assert len(orchestrator.agents) == 1
            assert orchestrator.agents[0].llm_provider is not None
