"""Shared fixtures for integration tests."""

import pytest
import sys
import json
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm.llm_provider import LLMProvider


class DynamicMockEngineProvider(LLMProvider):
    """Mock LLM provider that dynamically generates agent messages based on prompt."""

    def __init__(self):
        self.call_count = 0
        self.last_prompt = None

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a mock response with agent messages extracted from prompt."""
        self.last_prompt = prompt
        self.call_count += 1

        # Extract agent names from the prompt
        # Look for "agents": ["Name1", "Name2", ...] pattern first
        agents_array_pattern = r'"agents":\s*\[(.*?)\]'
        array_match = re.search(agents_array_pattern, prompt, re.DOTALL)
        agent_names = []

        if array_match:
            names_str = array_match.group(1)
            agent_names = re.findall(r'"([^"]+)"', names_str)

        # If no agents array found, look for agent names in other contexts
        if not agent_names:
            # Try to find agent names after "Agent names:" or similar patterns
            agent_list_pattern = r'(?:Agent names?|Agents?):\s*\[([^\]]+)\]'
            list_match = re.search(agent_list_pattern, prompt, re.IGNORECASE)
            if list_match:
                names_str = list_match.group(1)
                agent_names = [n.strip().strip('"\'') for n in names_str.split(',')]

        # Create response with agent_messages for each agent
        agent_messages = {name: f"Proceeding with step" for name in agent_names}
        agent_vars = {name: {} for name in agent_names}

        response = {
            "state_updates": {
                "global_vars": {},
                "agent_vars": agent_vars
            },
            "events": [],
            "agent_messages": agent_messages,
            "reasoning": "Mock simulation step"
        }

        return json.dumps(response, indent=2)


@pytest.fixture
def mock_engine_llm_provider():
    """Provide a mock LLM provider for engine that returns valid JSON responses."""
    return DynamicMockEngineProvider()
