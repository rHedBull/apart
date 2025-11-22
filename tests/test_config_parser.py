import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from config_parser import (
    parse_variable_definitions,
    create_variable_set,
    validate_agent_vars_config,
    validate_global_vars_config,
    validate_config,
)


class TestParseVariableDefinitions:
    """Tests for parsing variable definitions from YAML."""

    def test_parse_valid_definitions(self):
        """Test parsing valid variable definitions."""
        config = {
            "economic_strength": {
                "type": "float",
                "default": 1000.0,
                "min": 0,
            },
            "interest_rate": {
                "type": "float",
                "default": 0.05,
            },
            "is_active": {
                "type": "bool",
                "default": True,
            },
        }

        definitions = parse_variable_definitions(config)

        assert "economic_strength" in definitions
        assert definitions["economic_strength"].type == "float"
        assert definitions["economic_strength"].default == 1000.0
        assert definitions["economic_strength"].min == 0

        assert "interest_rate" in definitions
        assert definitions["interest_rate"].default == 0.05

        assert "is_active" in definitions
        assert definitions["is_active"].type == "bool"

    def test_missing_type_fails(self):
        """Test that missing type field raises error."""
        config = {
            "bad_var": {
                "default": 100,
            }
        }

        with pytest.raises(ValueError, match="missing required field 'type'"):
            parse_variable_definitions(config)

    def test_missing_default_fails(self):
        """Test that missing default field raises error."""
        config = {
            "bad_var": {
                "type": "int",
            }
        }

        with pytest.raises(ValueError, match="missing required field 'default'"):
            parse_variable_definitions(config)

    def test_invalid_type_fails(self):
        """Test that invalid type raises error."""
        config = {
            "bad_var": {
                "type": "string",  # not a valid type
                "default": "hello",
            }
        }

        with pytest.raises(ValueError, match="Error parsing variable"):
            parse_variable_definitions(config)

    def test_non_dict_variable_fails(self):
        """Test that non-dict variable spec raises error."""
        config = {
            "bad_var": "not a dict"
        }

        with pytest.raises(ValueError, match="must be a dictionary"):
            parse_variable_definitions(config)

    def test_invalid_constraints_fails(self):
        """Test that invalid constraints raise error."""
        config = {
            "bad_var": {
                "type": "int",
                "default": 50,
                "min": 100,  # min > default
            }
        }

        with pytest.raises(ValueError, match="Error parsing variable"):
            parse_variable_definitions(config)


class TestCreateVariableSet:
    """Tests for creating VariableSet from config."""

    def test_create_from_valid_config(self):
        """Test creating variable set from valid config."""
        config = {
            "health": {
                "type": "int",
                "default": 100,
                "min": 0,
                "max": 100,
            },
            "speed": {
                "type": "float",
                "default": 1.0,
            },
        }

        var_set = create_variable_set(config)

        assert var_set.get("health") == 100
        assert var_set.get("speed") == 1.0

    def test_create_from_none_returns_empty(self):
        """Test that None config returns empty VariableSet."""
        var_set = create_variable_set(None)

        assert len(var_set.definitions) == 0
        assert len(var_set.values) == 0

    def test_create_from_empty_dict_returns_empty(self):
        """Test that empty dict returns empty VariableSet."""
        var_set = create_variable_set({})

        assert len(var_set.definitions) == 0
        assert len(var_set.values) == 0


class TestValidateConfig:
    """Tests for config validation."""

    def test_validate_valid_agent_vars(self):
        """Test that valid agent_vars passes validation."""
        config = {
            "agent_vars": {
                "economic_strength": {
                    "type": "float",
                    "default": 1000.0,
                    "min": 0,
                }
            }
        }

        # Should not raise
        validate_agent_vars_config(config)
        validate_config(config)

    def test_validate_valid_global_vars(self):
        """Test that valid global_vars passes validation."""
        config = {
            "global_vars": {
                "interest_rate": {
                    "type": "float",
                    "default": 0.05,
                }
            }
        }

        # Should not raise
        validate_global_vars_config(config)
        validate_config(config)

    def test_validate_both_vars(self):
        """Test that both agent and global vars can be validated."""
        config = {
            "agent_vars": {
                "health": {
                    "type": "int",
                    "default": 100,
                }
            },
            "global_vars": {
                "difficulty": {
                    "type": "float",
                    "default": 1.0,
                }
            }
        }

        # Should not raise
        validate_config(config)

    def test_invalid_agent_vars_type_fails(self):
        """Test that non-dict agent_vars fails validation."""
        config = {
            "agent_vars": "not a dict"
        }

        with pytest.raises(ValueError, match="agent_vars must be a dictionary"):
            validate_agent_vars_config(config)

    def test_invalid_global_vars_type_fails(self):
        """Test that non-dict global_vars fails validation."""
        config = {
            "global_vars": "not a dict"
        }

        with pytest.raises(ValueError, match="global_vars must be a dictionary"):
            validate_global_vars_config(config)

    def test_invalid_variable_definition_fails(self):
        """Test that invalid variable definitions fail validation."""
        config = {
            "agent_vars": {
                "bad_var": {
                    "type": "int",
                    # missing default
                }
            }
        }

        with pytest.raises(ValueError):
            validate_config(config)
