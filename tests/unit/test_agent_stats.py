"""Tests for agent stats awareness functionality."""
import pytest
from src.core.agent import Agent
from src.llm.mock_provider import MockLLMProvider


class TestAgentStats:
    """Test that agents can access their own stats."""

    def test_agent_updates_stats(self):
        """Test that agent.update_stats() properly updates stats."""
        mock_llm = MockLLMProvider(["response"])
        agent = Agent(
            name="Test Agent",
            llm_provider=mock_llm,
            system_prompt="You are a test agent."
        )

        # Initial stats should be empty
        assert agent.current_stats == {}

        # Update stats
        stats = {"score": 100, "health": 50}
        agent.update_stats(stats)

        # Stats should be stored
        assert agent.current_stats == stats

    def test_agent_stats_added_to_system_prompt(self):
        """Test that stats are appended to system prompt."""
        mock_llm = MockLLMProvider(["response"])
        base_prompt = "You are a test agent."
        agent = Agent(
            name="Test Agent",
            llm_provider=mock_llm,
            system_prompt=base_prompt
        )

        # Initial system prompt should be unchanged
        assert agent.system_prompt == base_prompt
        assert agent.base_system_prompt == base_prompt

        # Update stats
        stats = {"score": 100, "health": 50, "compute_resource": 500.0}
        agent.update_stats(stats)

        # System prompt should now include stats
        assert "=== YOUR CURRENT STATUS ===" in agent.system_prompt
        assert "score: 100" in agent.system_prompt
        assert "health: 50" in agent.system_prompt
        assert "compute_resource: 500.0" in agent.system_prompt

        # Base prompt should be preserved
        assert agent.base_system_prompt == base_prompt

    def test_agent_stats_update_preserves_base_prompt(self):
        """Test that updating stats multiple times preserves base prompt."""
        mock_llm = MockLLMProvider(["response"])
        base_prompt = "You are a test agent."
        agent = Agent(
            name="Test Agent",
            llm_provider=mock_llm,
            system_prompt=base_prompt
        )

        # Update stats first time
        agent.update_stats({"score": 100})
        first_update = agent.system_prompt

        # Update stats again with different values
        agent.update_stats({"score": 200, "health": 75})
        second_update = agent.system_prompt

        # Both should start with base prompt
        assert base_prompt in first_update
        assert base_prompt in second_update

        # Second update should have new values
        assert "score: 200" in second_update
        assert "health: 75" in second_update
        assert "score: 100" not in second_update

    def test_agent_without_llm_ignores_stats(self):
        """Test that template-only agents don't modify prompt with stats."""
        agent = Agent(
            name="Template Agent",
            response_template="Fixed response"
        )

        # Update stats
        agent.update_stats({"score": 100})

        # Stats should be stored but shouldn't affect anything
        assert agent.current_stats == {"score": 100}
        # No system prompt to modify
        assert agent.system_prompt is None

    def test_empty_stats_resets_to_base_prompt(self):
        """Test that empty stats reset system prompt to base."""
        mock_llm = MockLLMProvider(["response"])
        base_prompt = "You are a test agent."
        agent = Agent(
            name="Test Agent",
            llm_provider=mock_llm,
            system_prompt=base_prompt
        )

        # Add stats
        agent.update_stats({"score": 100})
        assert "YOUR CURRENT STATUS" in agent.system_prompt

        # Clear stats
        agent.update_stats({})
        assert agent.system_prompt == base_prompt
        assert "YOUR CURRENT STATUS" not in agent.system_prompt
