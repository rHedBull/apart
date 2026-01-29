"""Tests for enhanced module models."""
import pytest
from modules.models import (
    BehaviorModule,
    ModuleVariable,
    ModuleLayer,
    Granularity,
    VariableType,
)


def test_module_layer_enum():
    """Test ModuleLayer enum has required values."""
    assert ModuleLayer.META.value == "meta"
    assert ModuleLayer.GROUNDING.value == "grounding"
    assert ModuleLayer.DOMAIN.value == "domain"
    assert ModuleLayer.DETAIL.value == "detail"


def test_granularity_enum():
    """Test Granularity enum has required values."""
    assert Granularity.MACRO.value == "macro"
    assert Granularity.MESO.value == "meso"
    assert Granularity.MICRO.value == "micro"


def test_behavior_module_new_fields():
    """Test BehaviorModule has new taxonomy fields."""
    module = BehaviorModule(
        name="test_module",
        description="Test",
        layer=ModuleLayer.DOMAIN,
        domain="economic",
        granularity_support=[Granularity.MESO, Granularity.MICRO],
        extends="economic_base",
    )

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "economic"
    assert Granularity.MESO in module.granularity_support
    assert module.extends == "economic_base"


def test_behavior_module_defaults():
    """Test BehaviorModule new fields have sensible defaults."""
    module = BehaviorModule(
        name="minimal",
        description="Minimal module",
    )

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain is None
    assert module.granularity_support == [Granularity.MACRO, Granularity.MESO, Granularity.MICRO]
    assert module.extends is None
