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
    from server.run_state import RunStateManager
    import server.job_queue as job_queue_module

    # Create a fake Redis connection (decode_responses=True for consistency)
    fake_redis = FakeRedis(decode_responses=True)

    # Reset RunStateManager singleton before initializing
    RunStateManager.reset_instance()

    # Mock the job queue initialization to use fake Redis
    def mock_init_job_queue(redis_url: str):
        from rq import Queue
        job_queue_module._redis_conn = fake_redis
        job_queue_module._queues = {
            "high": Queue("simulations-high", connection=fake_redis),
            "normal": Queue("simulations", connection=fake_redis),
            "low": Queue("simulations-low", connection=fake_redis),
        }

    # Mock the state manager initialization to use our fake Redis
    def mock_init_state_manager():
        RunStateManager.initialize(fake_redis)

    with patch.object(job_queue_module, 'init_job_queue', mock_init_job_queue):
        with patch('server.app._initialize_state_manager', mock_init_state_manager):
            with TestClient(app) as client:
                yield client

    # Reset state after test
    job_queue_module._redis_conn = None
    job_queue_module._queues = {}
    RunStateManager.reset_instance()


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
def state_manager():
    """Provide access to RunStateManager for tests that need to create runs.

    This fixture provides the state manager singleton after it's been
    initialized by the test_client fixture.
    """
    from server.run_state import get_state_manager
    return get_state_manager()


def create_test_run(
    run_id: str,
    scenario_name: str = "test-scenario",
    scenario_path: str = "/test/scenario.yaml",
    status: str = "running",
    max_steps: int | None = None,
    current_step: int = 0,
    worker_id: str = "test-worker",
):
    """Helper to create a run in state manager for testing.

    Creates a run with the given parameters and optionally transitions
    to running status.

    Args:
        run_id: Unique run ID
        scenario_name: Human-readable scenario name
        scenario_path: Path to scenario file
        status: Target status (pending, running, completed, failed)
        max_steps: Total number of steps (if known)
        current_step: Current step (for running simulations)
        worker_id: Worker ID (for running simulations)

    Returns:
        The created RunState
    """
    from server.run_state import get_state_manager

    manager = get_state_manager()
    if manager is None:
        raise RuntimeError("RunStateManager not initialized - use test_client fixture first")

    # Create the run
    state = manager.create_run(
        run_id=run_id,
        scenario_path=scenario_path,
        scenario_name=scenario_name,
        total_steps=max_steps,
    )

    # Transition to target status if not pending
    if status == "running":
        state = manager.transition(run_id, "running", worker_id=worker_id)
        if current_step > 0:
            state = manager.update_progress(run_id, current_step=current_step)
    elif status == "completed":
        state = manager.transition(run_id, "running", worker_id=worker_id)
        state = manager.transition(run_id, "completed")
    elif status == "failed":
        state = manager.transition(run_id, "running", worker_id=worker_id)
        state = manager.transition(run_id, "failed", error="Test failure")
    elif status == "paused":
        state = manager.transition(run_id, "running", worker_id=worker_id)
        state = manager.transition(run_id, "paused")

    return state


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
