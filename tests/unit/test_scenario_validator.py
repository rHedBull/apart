"""Tests for scenario validation."""
import pytest
import tempfile
from pathlib import Path
from utils.scenario_validator import ScenarioValidator, ValidationError


def test_validator_checks_required_agent_prompt_sections():
    """Test validator checks for required sections in agent prompts."""
    validator = ScenarioValidator()

    valid_prompt = """
# OBJECTIVES
- Goal 1
- Goal 2

# CONSTRAINTS
- Constraint 1

# INFORMATION ACCESS
- Full: own data
- Partial: enemy data
"""

    invalid_prompt = """
Just some text without required sections.
"""

    assert validator.validate_agent_prompt(valid_prompt) == []
    errors = validator.validate_agent_prompt(invalid_prompt)
    assert len(errors) > 0
    assert any("OBJECTIVES" in e for e in errors)


@pytest.fixture
def temp_modules_dir():
    """Create a temporary directory with test module files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a module that only supports macro and meso
        economic_base_yaml = """
module:
  name: economic_base
  description: Economic base module
  granularity_support: [macro, meso]
"""
        (tmpdir_path / "economic_base.yaml").write_text(economic_base_yaml)

        yield tmpdir_path


def test_validator_checks_granularity_compatibility(temp_modules_dir):
    """Test validator checks module granularity compatibility."""
    validator = ScenarioValidator(modules_dir=temp_modules_dir)

    config = {
        "meta": {
            "granularity": "micro",
        },
        "modules": ["economic_base"],  # Only supports macro, meso
    }

    errors = validator.validate_granularity(config)
    assert len(errors) > 0
    assert any("granularity" in e.lower() for e in errors)


def test_validator_passes_valid_granularity(temp_modules_dir):
    """Test validator passes when granularity is compatible."""
    validator = ScenarioValidator(modules_dir=temp_modules_dir)

    config = {
        "meta": {
            "granularity": "meso",
        },
        "modules": ["economic_base"],
    }

    errors = validator.validate_granularity(config)
    assert errors == []


def test_validator_handles_invalid_granularity_string():
    """Test validator reports error for invalid granularity string."""
    validator = ScenarioValidator()

    config = {
        "meta": {
            "granularity": "invalid_granularity",
        },
        "modules": [],
    }

    errors = validator.validate_granularity(config)
    assert len(errors) > 0
    assert any("invalid" in e.lower() for e in errors)


def test_validator_full_validation(temp_modules_dir):
    """Test full validation with valid config."""
    validator = ScenarioValidator(modules_dir=temp_modules_dir)

    config = {
        "meta": {
            "granularity": "meso",
        },
        "modules": ["economic_base"],
        "agents": [
            {
                "name": "TestAgent",
                "system_prompt": """
# OBJECTIVES
- Test objective

# CONSTRAINTS
- Test constraint

# INFORMATION ACCESS
- Full access
"""
            }
        ]
    }

    result = validator.validate(config)
    assert result.valid is True
    assert result.errors == []


def test_validator_full_validation_fails_with_invalid_prompt(temp_modules_dir):
    """Test full validation fails when agent prompt is invalid."""
    validator = ScenarioValidator(modules_dir=temp_modules_dir)

    config = {
        "meta": {
            "granularity": "meso",
        },
        "modules": ["economic_base"],
        "agents": [
            {
                "name": "BadAgent",
                "system_prompt": "No required sections here"
            }
        ]
    }

    result = validator.validate(config)
    assert result.valid is False
    assert len(result.errors) > 0
    assert any("BadAgent" in e for e in result.errors)


def test_validate_or_raise_raises_on_error(temp_modules_dir):
    """Test validate_or_raise raises ValidationError on invalid config."""
    validator = ScenarioValidator(modules_dir=temp_modules_dir)

    config = {
        "meta": {
            "granularity": "micro",  # Not supported by economic_base
        },
        "modules": ["economic_base"],
        "agents": []
    }

    with pytest.raises(ValidationError) as exc_info:
        validator.validate_or_raise(config)

    assert len(exc_info.value.errors) > 0


def test_validation_result_dataclass():
    """Test ValidationResult has expected attributes."""
    from utils.scenario_validator import ValidationResult

    result = ValidationResult(valid=True, errors=[], warnings=["A warning"])
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == ["A warning"]
