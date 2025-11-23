"""Tests for LLM-powered agents using mock providers (no API calls)."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.agent import Agent
from llm.mock_provider import MockLLMProvider


class TestLLMAgent:
    """Test LLM-powered agents with mock provider (no real API calls)."""

    def test_agent_with_llm_provider(self):
        """Test agent uses LLM provider for responses."""
        provider = MockLLMProvider(responses=["Strategic decision A", "Strategic decision B"])
        agent = Agent(
            name="AI Agent",
            llm_provider=provider,
            system_prompt="You are a strategic advisor"
        )

        response = agent.respond("What should we do?")

        assert response == "Strategic decision A"
        assert provider.call_count == 1
        assert provider.last_prompt == "What should we do?"
        assert provider.last_system_prompt == "You are a strategic advisor"

    def test_agent_llm_cycles_through_responses(self):
        """Test agent cycles through mock responses."""
        provider = MockLLMProvider(responses=["Response 1", "Response 2"])
        agent = Agent(name="AI Agent", llm_provider=provider)

        response1 = agent.respond("Message 1")
        response2 = agent.respond("Message 2")
        response3 = agent.respond("Message 3")

        assert response1 == "Response 1"
        assert response2 == "Response 2"
        assert response3 == "Response 1"  # Cycles back
        assert provider.call_count == 3

    def test_agent_llm_fallback_to_template(self):
        """Test agent with unavailable LLM raises error (no fallback in new architecture)."""
        provider = MockLLMProvider(available=False)
        agent = Agent(
            name="Fallback Agent",
            llm_provider=provider,
            response_template="Template response",
            system_prompt="Test prompt"
        )

        # In the new architecture, if LLM is configured but unavailable, it raises an error
        with pytest.raises(ValueError, match="LLM provider is configured but not available"):
            agent.respond("Test message")

    def test_agent_llm_no_template_fails_when_unavailable(self):
        """Test agent without template fails when LLM unavailable."""
        provider = MockLLMProvider(available=False)
        agent = Agent(
            name="No Fallback Agent",
            llm_provider=provider,
            system_prompt="Test prompt"
        )

        # Error message changed in new architecture to be more specific
        with pytest.raises(ValueError, match="LLM provider is configured but not available"):
            agent.respond("Test message")

    def test_agent_llm_increments_step_count(self):
        """Test LLM agent still tracks step count."""
        provider = MockLLMProvider(responses=["Response"])
        agent = Agent(name="AI Agent", llm_provider=provider)

        assert agent.step_count == 0
        agent.respond("Message 1")
        assert agent.step_count == 1
        agent.respond("Message 2")
        assert agent.step_count == 2

    def test_agent_template_only_still_works(self):
        """Test traditional template-only agents still work."""
        agent = Agent(name="Template Agent", response_template="Acknowledged")

        response = agent.respond("Test message")

        assert response == "Acknowledged (step 1)"
        assert agent.step_count == 1

    def test_agent_requires_either_llm_or_template(self):
        """Test agent requires either LLM provider or template."""
        with pytest.raises(ValueError, match="must have either response_template or llm_provider"):
            Agent(name="Invalid Agent")

    def test_agent_llm_without_system_prompt(self):
        """Test LLM agent works without system prompt."""
        provider = MockLLMProvider(responses=["Response"])
        agent = Agent(name="AI Agent", llm_provider=provider)

        response = agent.respond("Test message")

        assert response == "Response"
        assert provider.last_system_prompt is None

    def test_multiple_llm_agents_independent(self):
        """Test multiple LLM agents maintain independent state."""
        provider1 = MockLLMProvider(responses=["Agent 1 response"])
        provider2 = MockLLMProvider(responses=["Agent 2 response"])

        agent1 = Agent(name="Agent 1", llm_provider=provider1)
        agent2 = Agent(name="Agent 2", llm_provider=provider2)

        response1 = agent1.respond("Message")
        response2 = agent2.respond("Message")

        assert response1 == "Agent 1 response"
        assert response2 == "Agent 2 response"
        assert provider1.call_count == 1
        assert provider2.call_count == 1
        assert agent1.step_count == 1
        assert agent2.step_count == 1


class TestMockLLMProvider:
    """Test the mock provider itself."""

    def test_mock_provider_available_by_default(self):
        """Test mock provider is available by default."""
        provider = MockLLMProvider()
        assert provider.is_available() is True

    def test_mock_provider_can_be_unavailable(self):
        """Test mock provider can be set unavailable."""
        provider = MockLLMProvider(available=False)
        assert provider.is_available() is False

    def test_mock_provider_returns_default_response(self):
        """Test mock provider returns default response."""
        provider = MockLLMProvider()
        response = provider.generate_response("Test prompt")
        assert response == "Mock LLM response"

    def test_mock_provider_returns_custom_responses(self):
        """Test mock provider returns custom responses."""
        provider = MockLLMProvider(responses=["Custom 1", "Custom 2"])
        assert provider.generate_response("Test") == "Custom 1"
        assert provider.generate_response("Test") == "Custom 2"

    def test_mock_provider_cycles_responses(self):
        """Test mock provider cycles through responses."""
        provider = MockLLMProvider(responses=["A", "B"])
        assert provider.generate_response("Test") == "A"
        assert provider.generate_response("Test") == "B"
        assert provider.generate_response("Test") == "A"  # Cycles
        assert provider.call_count == 3

    def test_mock_provider_stores_last_prompts(self):
        """Test mock provider stores prompts for verification."""
        provider = MockLLMProvider()
        provider.generate_response("User prompt", "System prompt")

        assert provider.last_prompt == "User prompt"
        assert provider.last_system_prompt == "System prompt"

    def test_mock_provider_reset(self):
        """Test mock provider reset works."""
        provider = MockLLMProvider(responses=["A", "B"])
        provider.generate_response("Test 1", "System")
        provider.generate_response("Test 2", "System")

        assert provider.call_count == 2

        provider.reset()

        assert provider.call_count == 0
        assert provider.last_prompt is None
        assert provider.last_system_prompt is None

    def test_mock_provider_raises_when_unavailable(self):
        """Test mock provider raises error when unavailable."""
        provider = MockLLMProvider(available=False)

        with pytest.raises(ValueError, match="Mock LLM provider not available"):
            provider.generate_response("Test")
