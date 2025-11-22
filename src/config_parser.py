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


def create_variable_set_with_overrides(
    var_definitions_config: dict[str, Any] | None,
    overrides: dict[str, Any] | None
) -> VariableSet:
    """
    Create a VariableSet with optional value overrides.

    Args:
        var_definitions_config: Dictionary of variable definitions from YAML
        overrides: Dictionary of variable value overrides

    Returns:
        VariableSet with definitions and values (defaults or overridden)

    Raises:
        ValueError: If override references undefined variable or invalid value
    """
    if not var_definitions_config:
        if overrides:
            raise ValueError("Cannot override variables when no variable definitions exist")
        return VariableSet()

    definitions = parse_variable_definitions(var_definitions_config)
    var_set = VariableSet(definitions=definitions)

    # Apply overrides if provided
    if overrides:
        for var_name, value in overrides.items():
            if var_name not in definitions:
                raise ValueError(
                    f"Cannot override undefined variable '{var_name}'. "
                    f"Defined variables: {', '.join(definitions.keys())}"
                )
            try:
                var_set.set(var_name, value)
            except Exception as e:
                raise ValueError(
                    f"Invalid override value for variable '{var_name}': {e}"
                ) from e

    return var_set


def validate_agent_config(agent_config: dict[str, Any], agent_vars_definitions: dict[str, Any] | None) -> None:
    """
    Validate a single agent configuration.

    Args:
        agent_config: Agent configuration dictionary
        agent_vars_definitions: Global agent variable definitions

    Raises:
        ValueError: If agent config is invalid
    """
    if "variables" in agent_config:
        if not isinstance(agent_config["variables"], dict):
            raise ValueError(f"Agent '{agent_config.get('name', 'unknown')}': variables must be a dictionary")

        # Validate that all override variables are defined
        if agent_vars_definitions:
            definitions = parse_variable_definitions(agent_vars_definitions)
            for var_name in agent_config["variables"].keys():
                if var_name not in definitions:
                    raise ValueError(
                        f"Agent '{agent_config.get('name', 'unknown')}': "
                        f"undefined variable '{var_name}' in overrides. "
                        f"Defined variables: {', '.join(definitions.keys())}"
                    )


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the entire configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    validate_agent_vars_config(config)
    validate_global_vars_config(config)

    # Validate agent configurations
    if "agents" in config:
        agent_vars_defs = config.get("agent_vars")
        for agent_config in config["agents"]:
            validate_agent_config(agent_config, agent_vars_defs)
