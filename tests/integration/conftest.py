"""Shared fixtures for integration tests."""

import pytest
import sys
import json
import re
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm.llm_provider import LLMProvider


# ============================================================================
# Server API Test Fixtures
# ============================================================================

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app with mocked Redis."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from fakeredis import FakeRedis
    from server.app import app
    import server.job_queue as job_queue_module

    # Create a fake Redis connection
    fake_redis = FakeRedis()

    # Mock the job queue initialization to use fake Redis
    def mock_init_job_queue(redis_url: str):
        from rq import Queue
        job_queue_module._redis_conn = fake_redis
        job_queue_module._queues = {
            "high": Queue("simulations-high", connection=fake_redis),
            "normal": Queue("simulations", connection=fake_redis),
            "low": Queue("simulations-low", connection=fake_redis),
        }

    with patch.object(job_queue_module, 'init_job_queue', mock_init_job_queue):
        with TestClient(app) as client:
            yield client

    # Reset job queue state after test
    job_queue_module._redis_conn = None
    job_queue_module._queues = {}


@pytest.fixture
def event_bus_reset(tmp_path):
    """Reset EventBus with temporary persistence for integration tests."""
    from server.event_bus import EventBus

    # Set test persist path
    EventBus._test_persist_path = tmp_path / "integration_events.jsonl"
    EventBus.reset_instance()
    bus = EventBus.get_instance()

    yield bus

    # Cleanup
    EventBus._test_persist_path = None
    EventBus.reset_instance()


@pytest.fixture
def sample_scenario(tmp_path):
    """Create a minimal test scenario file."""
    scenario_content = {
        "name": "Test Scenario",
        "max_steps": 2,
        "agents": [
            {
                "name": "TestAgent",
                "llm": {"provider": "mock", "model": "test"}
            }
        ],
        "initial_state": {
            "global_vars": {"test_var": 1}
        }
    }

    scenario_path = tmp_path / "test_scenario.yaml"
    import yaml
    with open(scenario_path, "w") as f:
        yaml.dump(scenario_content, f)

    return scenario_path


@pytest.fixture
def persist_path(tmp_path):
    """Provide a temporary path for event persistence testing."""
    return tmp_path / "test_events.jsonl"


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
                r'Agent Responses from previous step:(.*?)(?=Return ONLY valid JSON|Return JSON format:|=== CRITICAL|$)',
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
