from typing import Any
from variables import VariableDefinition, VariableSet


def parse_variable_definitions(var_config: dict[str, Any]) -> dict[str, VariableDefinition]:
    """
    Parse variable definitions from YAML config.

    Expected format:
    {
        "var_name": {
            "type": "float",
            "default": 1000.0,
            "min": 0,
            "max": 10000
        }
    }
    """
    definitions = {}

    for var_name, var_spec in var_config.items():
        if not isinstance(var_spec, dict):
            raise ValueError(f"Variable '{var_name}' must be a dictionary, got {type(var_spec).__name__}")

        if "type" not in var_spec:
            raise ValueError(f"Variable '{var_name}' missing required field 'type'")

        if "default" not in var_spec:
            raise ValueError(f"Variable '{var_name}' missing required field 'default'")

        try:
            definitions[var_name] = VariableDefinition(**var_spec)
        except Exception as e:
            raise ValueError(f"Error parsing variable '{var_name}': {e}") from e

    return definitions


def create_variable_set(var_config: dict[str, Any] | None) -> VariableSet:
    """
    Create a VariableSet from YAML config.

    Args:
        var_config: Dictionary of variable definitions from YAML

    Returns:
        VariableSet with definitions and default values
    """
    if not var_config:
        return VariableSet()

    definitions = parse_variable_definitions(var_config)
    return VariableSet(definitions=definitions)


def validate_agent_vars_config(config: dict[str, Any]) -> None:
    """Validate that agent_vars section is properly defined."""
    if "agent_vars" in config:
        if not isinstance(config["agent_vars"], dict):
            raise ValueError("agent_vars must be a dictionary")
        # Try to parse to validate
        create_variable_set(config["agent_vars"])


def validate_global_vars_config(config: dict[str, Any]) -> None:
    """Validate that global_vars section is properly defined."""
    if "global_vars" in config:
        if not isinstance(config["global_vars"], dict):
            raise ValueError("global_vars must be a dictionary")
        # Try to parse to validate
        create_variable_set(config["global_vars"])


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the entire configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    validate_agent_vars_config(config)
    validate_global_vars_config(config)
