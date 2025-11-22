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
        agent_names = []

        # Strategy 1: Look for "Agents: Name1, Name2, ..." pattern (initialization prompt)
        agents_pattern = r'Agents:\s*([^\n]+)'
        match = re.search(agents_pattern, prompt)
        if match:
            names_str = match.group(1)
            # Split by comma and clean up
            agent_names = [n.strip() for n in names_str.split(',')]

        # Strategy 2: Look for "Agent Responses from previous step:" section (step prompt)
        if not agent_names:
            # Find section like:
            # Agent Responses from previous step:
            #   Agent A: "..."
            #   Agent B: "..."
            response_section = re.search(
                r'Agent Responses from previous step:(.*?)(?=Return JSON format:|$)',
                prompt,
                re.DOTALL
            )
            if response_section:
                section_text = response_section.group(1)
                # Extract agent names from lines like "  Agent Name: "response""
                agent_names = re.findall(r'^\s+([^:]+):', section_text, re.MULTILINE)
                # Clean up names
                agent_names = [n.strip() for n in agent_names if n.strip()]

        # Strategy 3: Fallback to JSON pattern
        if not agent_names:
            agents_array_pattern = r'"agents":\s*\[(.*?)\]'
            array_match = re.search(agents_array_pattern, prompt, re.DOTALL)
            if array_match:
                names_str = array_match.group(1)
                agent_names = re.findall(r'"([^"]+)"', names_str)

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
