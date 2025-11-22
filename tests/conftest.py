"""Pytest configuration for tests."""

import sys
import shutil
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
