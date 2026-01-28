"""Tests for module loader with new schema fields."""
import pytest
import tempfile
from pathlib import Path
from modules.loader import ModuleLoader
from modules.models import ModuleLayer, Granularity


@pytest.fixture
def temp_modules_dir():
    """Create a temporary directory with test module files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_loader_parses_layer_field(temp_modules_dir):
    """Test loader parses layer field from YAML."""
    module_yaml = """
module:
  name: test_module
  description: Test module
  layer: grounding
  domain: null
"""
    (temp_modules_dir / "test_module.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("test_module")

    assert module.layer == ModuleLayer.GROUNDING


def test_loader_parses_granularity_support(temp_modules_dir):
    """Test loader parses granularity_support field."""
    module_yaml = """
module:
  name: meso_only
  description: Meso-level only module
  granularity_support: [meso]
"""
    (temp_modules_dir / "meso_only.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("meso_only")

    assert module.granularity_support == [Granularity.MESO]


def test_loader_parses_extends_field(temp_modules_dir):
    """Test loader parses extends field."""
    module_yaml = """
module:
  name: economic_detailed
  description: Detailed economic module
  layer: detail
  domain: economic
  extends: economic_base
"""
    (temp_modules_dir / "economic_detailed.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("economic_detailed")

    assert module.extends == "economic_base"
    assert module.layer == ModuleLayer.DETAIL
    assert module.domain == "economic"


def test_loader_defaults_new_fields(temp_modules_dir):
    """Test loader applies defaults for new fields."""
    module_yaml = """
module:
  name: minimal
  description: Minimal module without new fields
"""
    (temp_modules_dir / "minimal.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("minimal")

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain is None
    assert len(module.granularity_support) == 3  # All granularities
    assert module.extends is None


def test_loader_parses_all_layer_types(temp_modules_dir):
    """Test loader parses all ModuleLayer enum values."""
    layers = ["meta", "grounding", "domain", "detail"]
    expected = [ModuleLayer.META, ModuleLayer.GROUNDING, ModuleLayer.DOMAIN, ModuleLayer.DETAIL]

    for layer_str, expected_enum in zip(layers, expected):
        module_yaml = f"""
module:
  name: {layer_str}_module
  description: Module with {layer_str} layer
  layer: {layer_str}
"""
        (temp_modules_dir / f"{layer_str}_module.yaml").write_text(module_yaml)
        loader = ModuleLoader(temp_modules_dir)
        loader.clear_cache()
        module = loader.load(f"{layer_str}_module")
        assert module.layer == expected_enum, f"Failed for layer: {layer_str}"


def test_loader_parses_all_granularity_types(temp_modules_dir):
    """Test loader parses all Granularity enum values."""
    module_yaml = """
module:
  name: all_granularities
  description: Module with all granularities
  granularity_support: [macro, meso, micro]
"""
    (temp_modules_dir / "all_granularities.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("all_granularities")

    assert Granularity.MACRO in module.granularity_support
    assert Granularity.MESO in module.granularity_support
    assert Granularity.MICRO in module.granularity_support


def test_loader_validates_extends_dependency(temp_modules_dir):
    """Test loader validates that extended module is loaded."""
    base_yaml = """
module:
  name: economic_base
  description: Base economic module
  layer: domain
"""
    detail_yaml = """
module:
  name: economic_detailed
  description: Detailed economic module
  layer: detail
  extends: economic_base
"""
    (temp_modules_dir / "economic_base.yaml").write_text(base_yaml)
    (temp_modules_dir / "economic_detailed.yaml").write_text(detail_yaml)

    loader = ModuleLoader(temp_modules_dir)

    # Loading detail without base should fail
    from modules.loader import ModuleDependencyError
    with pytest.raises(ModuleDependencyError, match="extends.*economic_base"):
        loader.load_many(["economic_detailed"])

    # Loading both should work
    loader.clear_cache()
    modules = loader.load_many(["economic_base", "economic_detailed"])
    assert len(modules) == 2


def test_loader_validates_granularity_compatibility(temp_modules_dir):
    """Test loader can check granularity compatibility."""
    macro_only = """
module:
  name: macro_module
  description: Macro only
  granularity_support: [macro]
"""
    micro_only = """
module:
  name: micro_module
  description: Micro only
  granularity_support: [micro]
"""
    (temp_modules_dir / "macro_module.yaml").write_text(macro_only)
    (temp_modules_dir / "micro_module.yaml").write_text(micro_only)

    loader = ModuleLoader(temp_modules_dir)
    modules = loader.load_many(["macro_module", "micro_module"])

    # Check that we can find common granularity (should be none)
    from modules.loader import find_common_granularity
    common = find_common_granularity(modules)
    assert common == []
