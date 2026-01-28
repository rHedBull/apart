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


def test_economic_base_has_new_schema_fields():
    """Test economic_base has layer and granularity fields."""
    loader = ModuleLoader()
    module = loader.load("economic_base")

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "economic"
    assert Granularity.MESO in module.granularity_support


def test_supply_chain_base_has_new_schema_fields():
    """Test supply_chain_base has layer and granularity fields."""
    loader = ModuleLoader()
    module = loader.load("supply_chain_base")

    assert module.layer == ModuleLayer.DETAIL
    assert module.domain == "economic"
    assert module.extends == "economic_base"
    assert Granularity.MESO in module.granularity_support


def test_diplomatic_base_module_loads():
    """Test diplomatic_base module loads correctly."""
    loader = ModuleLoader()
    module = loader.load("diplomatic_base")

    assert module.name == "diplomatic_base"
    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "diplomatic"


def test_diplomatic_base_has_alliance_variables():
    """Test diplomatic_base provides alliance tracking."""
    loader = ModuleLoader()
    module = loader.load("diplomatic_base")

    var_names = [v.name for v in module.variables]
    assert "alliances" in var_names or "treaties" in var_names


def test_trust_dynamics_module_loads():
    """Test trust_dynamics module loads correctly."""
    loader = ModuleLoader()
    module = loader.load("trust_dynamics")

    assert module.name == "trust_dynamics"
    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "social"


def test_trust_dynamics_has_trust_variables():
    """Test trust_dynamics provides trust tracking."""
    loader = ModuleLoader()
    module = loader.load("trust_dynamics")

    agent_vars = [v for v in module.variables if v.scope == "agent"]
    var_names = [v.name for v in agent_vars]

    assert "trust_scores" in var_names


def test_territory_graph_is_grounding_module():
    """Test territory_graph is properly categorized as grounding."""
    loader = ModuleLoader()
    module = loader.load("territory_graph")

    assert module.layer == ModuleLayer.GROUNDING
    assert module.domain is None  # Grounding modules are domain-agnostic
