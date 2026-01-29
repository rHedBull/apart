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


def test_all_core_modules_compose():
    """Test all 5 core modules can be loaded and composed together."""
    from modules.composer import ModuleComposer

    loader = ModuleLoader()
    core_modules = [
        "agents_base",
        "territory_graph",
        "economic_base",
        "diplomatic_base",
        "trust_dynamics",
    ]

    # Should load without dependency errors
    modules = loader.load_many(core_modules)
    assert len(modules) == 5

    # Should compose without errors
    composer = ModuleComposer()
    # Note: territory_graph needs map_file config, so we skip config for this test
    # Just verify modules load and have no conflicts
    composed = composer.compose(modules)
    assert composed is not None

    # Verify no conflicts between core modules
    all_conflicts = set()
    for m in modules:
        all_conflicts.update(m.conflicts_with)

    core_names = set(core_modules)
    conflicts_with_core = all_conflicts & core_names
    assert conflicts_with_core == set(), f"Core modules conflict with each other: {conflicts_with_core}"


def test_core_modules_share_meso_granularity():
    """Test all core modules support meso granularity."""
    from modules.loader import find_common_granularity

    loader = ModuleLoader()
    core_modules = [
        "agents_base",
        "territory_graph",
        "economic_base",
        "diplomatic_base",
        "trust_dynamics",
    ]

    modules = loader.load_many(core_modules)
    common = find_common_granularity(modules)

    assert Granularity.MESO in common, "All core modules should support meso granularity"


def test_military_base_module_loads():
    """Test military_base module loads correctly with new schema."""
    loader = ModuleLoader()
    module = loader.load("military_base")

    assert module.name == "military_base"
    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "military"


def test_military_base_has_all_granularities():
    """Test military_base supports all granularity levels."""
    loader = ModuleLoader()
    module = loader.load("military_base")

    assert Granularity.MACRO in module.granularity_support
    assert Granularity.MESO in module.granularity_support
    assert Granularity.MICRO in module.granularity_support


def test_military_base_has_deterrence_variables():
    """Test military_base provides deterrence and force variables."""
    loader = ModuleLoader()
    module = loader.load("military_base")

    var_names = [v.name for v in module.variables]

    assert "military_strength" in var_names
    assert "nuclear_capability" in var_names
    assert "deterrence_credibility" in var_names
    assert "force_projection" in var_names


def test_military_base_has_conflict_tracking():
    """Test military_base provides conflict and proxy war tracking."""
    loader = ModuleLoader()
    module = loader.load("military_base")

    global_vars = [v for v in module.variables if v.scope == "global"]
    global_var_names = [v.name for v in global_vars]

    assert "active_conflicts" in global_var_names
    assert "proxy_conflicts" in global_var_names
    assert "global_tension_level" in global_var_names


def test_military_base_composes_with_other_domains():
    """Test military_base composes with economic and diplomatic modules."""
    from modules.composer import ModuleComposer

    loader = ModuleLoader()
    modules = loader.load_many([
        "military_base",
        "economic_base",
        "diplomatic_base",
    ])

    composer = ModuleComposer()
    composed = composer.compose(modules)
    assert composed is not None

    # Verify all module variables are present (agent_variables is a dict)
    agent_var_names = list(composed.agent_variables.keys())
    assert "military_strength" in agent_var_names  # from military
    assert "gdp" in agent_var_names  # from economic
    assert "alliances" in agent_var_names  # from diplomatic
