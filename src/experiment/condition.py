"""
Condition modification utilities.

Apply dot-path modifications to scenario configurations.
"""

import copy
import re
from typing import Any, Dict


def apply_modifications(config: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply modifications to a configuration using dot-path keys.

    Creates a deep copy of the config and applies all modifications.

    Args:
        config: Base configuration dictionary
        modifications: Dict mapping dot-paths to new values

    Returns:
        Modified copy of the configuration

    Examples:
        >>> config = {"agent_vars": {"trust": {"default": 50}}}
        >>> mods = {"agent_vars.trust.default": 80}
        >>> result = apply_modifications(config, mods)
        >>> result["agent_vars"]["trust"]["default"]
        80

        >>> config = {"agents": [{"name": "A", "variables": {"x": 1}}]}
        >>> mods = {"agents.0.variables.x": 10}
        >>> result = apply_modifications(config, mods)
        >>> result["agents"][0]["variables"]["x"]
        10
    """
    result = copy.deepcopy(config)

    for path, value in modifications.items():
        _set_nested_value(result, path, value)

    return result


def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in nested dict/list using dot-path.

    Supports both dict keys and list indices:
    - "foo.bar.baz" -> data["foo"]["bar"]["baz"]
    - "agents.0.name" -> data["agents"][0]["name"]
    - "agents[0].name" -> data["agents"][0]["name"] (bracket notation)

    Args:
        data: Nested dictionary to modify (in place)
        path: Dot-separated path with optional bracket notation for indices
        value: Value to set

    Raises:
        KeyError: If intermediate path doesn't exist
        IndexError: If list index is out of range
        TypeError: If trying to index into non-dict/list
    """
    # Normalize bracket notation to dot notation: "agents[0].name" -> "agents.0.name"
    normalized_path = re.sub(r"\[(\d+)\]", r".\1", path)
    parts = normalized_path.split(".")

    current = data
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(
                    f"Path '{'.'.join(parts[:i+1])}' not found in config. "
                    f"Available keys: {list(current.keys())}"
                )
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except ValueError:
                raise TypeError(
                    f"Cannot use non-integer key '{part}' on list at '{'.'.join(parts[:i])}'"
                )
            except IndexError:
                raise IndexError(
                    f"Index {idx} out of range for list at '{'.'.join(parts[:i])}' "
                    f"(length: {len(current)})"
                )
        else:
            raise TypeError(
                f"Cannot traverse into {type(current).__name__} at '{'.'.join(parts[:i])}'"
            )

    # Set the final value
    final_key = parts[-1]
    if isinstance(current, dict):
        if final_key not in current:
            raise KeyError(
                f"Key '{final_key}' not found at '{'.'.join(parts[:-1])}'. "
                f"Available keys: {list(current.keys())}"
            )
        current[final_key] = value
    elif isinstance(current, list):
        try:
            idx = int(final_key)
            current[idx] = value
        except ValueError:
            raise TypeError(
                f"Cannot use non-integer key '{final_key}' on list at '{'.'.join(parts[:-1])}'"
            )
        except IndexError:
            raise IndexError(
                f"Index {idx} out of range for list at '{'.'.join(parts[:-1])}' "
                f"(length: {len(current)})"
            )
    else:
        raise TypeError(
            f"Cannot set value on {type(current).__name__} at '{'.'.join(parts[:-1])}'"
        )


def validate_modifications(config: Dict[str, Any], modifications: Dict[str, Any]) -> list[str]:
    """
    Validate that all modification paths exist in config.

    Args:
        config: Base configuration dictionary
        modifications: Dict mapping dot-paths to new values

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    test_config = copy.deepcopy(config)

    for path, value in modifications.items():
        try:
            _set_nested_value(test_config, path, value)
        except (KeyError, IndexError, TypeError) as e:
            errors.append(f"Invalid modification '{path}': {e}")

    return errors
