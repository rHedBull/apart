"""Tests for core module definitions."""
import pytest
from modules.loader import ModuleLoader
from modules.models import ModuleLayer, Granularity


def test_agents_base_module_loads():
    """Test agents_base module loads correctly."""
    loader = ModuleLoader()
    module = loader.load("agents_base")

    assert module.name == "agents_base"
    assert module.layer == ModuleLayer.DOMAIN
    assert Granularity.MESO in module.granularity_support


def test_agents_base_has_core_variables():
    """Test agents_base provides core agent variables."""
    loader = ModuleLoader()
    module = loader.load("agents_base")

    # Should have agent-scoped communication variables
    agent_vars = [v for v in module.variables if v.scope == "agent"]
    var_names = [v.name for v in agent_vars]

    assert "stance" in var_names or "position" in var_names
