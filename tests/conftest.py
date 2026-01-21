"""Pytest configuration for tests."""

import sys
import shutil
import tempfile
from pathlib import Path
import pytest

# Add src directory to Python path for all tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def cleanup_results():
    """Automatically clean up results directory after each test."""
    # Run test
    yield

    # Cleanup after test
    results_dir = Path("results")
    if results_dir.exists():
        try:
            shutil.rmtree(results_dir)
        except Exception:
            pass  # Best effort cleanup


@pytest.fixture(autouse=True)
def reset_event_bus_for_tests(tmp_path):
    """Reset EventBus singleton and use temporary persistence for each test.

    This sets a class-level test persist path that is used whenever a new
    EventBus instance is created, including when tests call reset_instance().
    """
    from server.event_bus import EventBus

    # Set test persist path at class level - this persists across reset_instance() calls
    EventBus._test_persist_path = tmp_path / "test_events.jsonl"

    # Reset and get fresh instance
    EventBus.reset_instance()
    bus = EventBus.get_instance()

    yield bus

    # Cleanup after test
    EventBus._test_persist_path = None
    EventBus.reset_instance()
