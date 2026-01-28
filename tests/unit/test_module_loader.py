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
