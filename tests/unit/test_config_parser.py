import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from utils.config_parser import (
    parse_variable_definitions,
    create_variable_set,
    create_variable_set_with_overrides,
    validate_agent_config,
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


class TestVariableOverrides:
    """Tests for agent variable overrides."""

    def test_create_with_valid_overrides(self):
        """Test creating variable set with valid overrides."""
        var_defs = {
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
        overrides = {
            "health": 75,
            "speed": 1.5,
        }

        var_set = create_variable_set_with_overrides(var_defs, overrides)

        assert var_set.get("health") == 75
        assert var_set.get("speed") == 1.5

    def test_create_with_partial_overrides(self):
        """Test that non-overridden variables use defaults."""
        var_defs = {
            "health": {
                "type": "int",
                "default": 100,
            },
            "speed": {
                "type": "float",
                "default": 1.0,
            },
        }
        overrides = {
            "health": 50,
        }

        var_set = create_variable_set_with_overrides(var_defs, overrides)

        assert var_set.get("health") == 50
        assert var_set.get("speed") == 1.0  # uses default

    def test_override_undefined_variable_fails(self):
        """Test that overriding undefined variable fails."""
        var_defs = {
            "health": {
                "type": "int",
                "default": 100,
            }
        }
        overrides = {
            "undefined_var": 50,
        }

        with pytest.raises(ValueError, match="Cannot override undefined variable"):
            create_variable_set_with_overrides(var_defs, overrides)

    def test_override_with_invalid_value_fails(self):
        """Test that invalid override value fails validation."""
        var_defs = {
            "health": {
                "type": "int",
                "default": 100,
                "min": 0,
                "max": 100,
            }
        }
        overrides = {
            "health": 150,  # exceeds max
        }

        with pytest.raises(ValueError, match="Invalid override value"):
            create_variable_set_with_overrides(var_defs, overrides)

    def test_override_with_no_definitions_fails(self):
        """Test that overrides fail when no definitions exist."""
        overrides = {
            "health": 100,
        }

        with pytest.raises(ValueError, match="Cannot override variables when no variable definitions exist"):
            create_variable_set_with_overrides(None, overrides)

    def test_validate_agent_config_with_valid_overrides(self):
        """Test validating agent config with valid variable overrides."""
        agent_config = {
            "name": "TestAgent",
            "variables": {
                "health": 75,
            }
        }
        agent_vars_defs = {
            "health": {
                "type": "int",
                "default": 100,
            }
        }

        # Should not raise
        validate_agent_config(agent_config, agent_vars_defs)

    def test_validate_agent_config_with_undefined_variable_fails(self):
        """Test that agent config with undefined variable override fails."""
        agent_config = {
            "name": "TestAgent",
            "variables": {
                "undefined_var": 50,
            }
        }
        agent_vars_defs = {
            "health": {
                "type": "int",
                "default": 100,
            }
        }

        with pytest.raises(ValueError, match="undefined variable.*in overrides"):
            validate_agent_config(agent_config, agent_vars_defs)

    def test_validate_full_config_with_agent_overrides(self):
        """Test full config validation with agent variable overrides."""
        config = {
            "agent_vars": {
                "economic_strength": {
                    "type": "float",
                    "default": 1000.0,
                    "min": 0.0,
                }
            },
            "agents": [
                {
                    "name": "Agent Alpha",
                    "response_template": "OK",
                    "variables": {
                        "economic_strength": 500.0,
                    }
                },
                {
                    "name": "Agent Beta",
                    "response_template": "OK",
                    # No overrides, uses defaults
                }
            ]
        }

        # Should not raise
        validate_config(config)


def test_validate_engine_config():
    """Test engine configuration validation."""
    from utils.config_parser import validate_engine_config

    # Valid engine config
    config = {
        "engine": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "system_prompt": "Test prompt",
            "simulation_plan": "Test plan"
        }
    }
    validate_engine_config(config)  # Should not raise

    # Missing required field
    config_missing = {
        "engine": {
            "provider": "gemini"
            # Missing model, system_prompt, simulation_plan
        }
    }
    with pytest.raises(ValueError, match="missing required field"):
        validate_engine_config(config_missing)

    # Invalid scripted event
    config_bad_event = {
        "engine": {
            "provider": "gemini",
            "model": "test",
            "system_prompt": "Test",
            "simulation_plan": "Test",
            "scripted_events": [
                {"step": "not_an_int", "type": "test", "description": "test"}
            ]
        }
    }
    with pytest.raises(ValueError):
        validate_engine_config(config_bad_event)


def test_parse_scripted_events():
    """Test parsing scripted events."""
    from utils.config_parser import parse_scripted_events
    from core.engine_models import ScriptedEvent

    events_config = [
        {"step": 20, "type": "major_war", "description": "War begins"},
        {"step": 35, "type": "disaster", "description": "Earthquake"}
    ]

    events = parse_scripted_events(events_config)

    assert len(events) == 2
    assert isinstance(events[0], ScriptedEvent)
    assert events[0].step == 20
    assert events[0].type == "major_war"
    assert events[1].step == 35
