import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.agent import Agent


class TestAgent:
    """Test Agent class."""

    def test_agent_initialization(self):
        """Test agent is properly initialized."""
        agent = Agent(name="TestAgent", response_template="Acknowledged")

        assert agent.name == "TestAgent"
        assert agent.response_template == "Acknowledged"
        assert agent.step_count == 0

    def test_agent_respond(self):
        """Test agent generates response."""
        agent = Agent(name="Agent1", response_template="Processing")

        response = agent.respond("Do something")

        assert response == "Processing (step 1)"
        assert agent.step_count == 1

    def test_agent_increments_step_count(self):
        """Test that step_count increments with each response."""
        agent = Agent(name="Agent1", response_template="Done")

        agent.respond("Message 1")
        assert agent.step_count == 1

        agent.respond("Message 2")
        assert agent.step_count == 2

        agent.respond("Message 3")
        assert agent.step_count == 3

    def test_agent_response_includes_step(self):
        """Test that response includes current step number."""
        agent = Agent(name="Agent1", response_template="OK")

        response1 = agent.respond("Test")
        assert "(step 1)" in response1

        response2 = agent.respond("Test")
        assert "(step 2)" in response2

        response3 = agent.respond("Test")
        assert "(step 3)" in response3

    def test_agent_ignores_message_content(self):
        """Test that agent response doesn't depend on message content."""
        agent = Agent(name="Agent1", response_template="Ready")

        response1 = agent.respond("Do task A")
        response2 = agent.respond("Do task B")
        response3 = agent.respond("")

        # All should have same template, different step
        assert response1 == "Ready (step 1)"
        assert response2 == "Ready (step 2)"
        assert response3 == "Ready (step 3)"

    def test_multiple_agents_independent(self):
        """Test that multiple agents maintain independent state."""
        agent1 = Agent(name="Agent1", response_template="Agent1 ready")
        agent2 = Agent(name="Agent2", response_template="Agent2 ready")

        agent1.respond("Message")
        agent1.respond("Message")

        agent2.respond("Message")

        assert agent1.step_count == 2
        assert agent2.step_count == 1

        assert agent1.respond("Message") == "Agent1 ready (step 3)"
        assert agent2.respond("Message") == "Agent2 ready (step 2)"
