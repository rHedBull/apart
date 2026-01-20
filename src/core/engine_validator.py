import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from core.engine_models import ConstraintHit


@dataclass
class ValidationResult:
    """Result of validation operation."""
    success: bool
    error: Optional[str] = None


class EngineValidator:
    """Validates SimulatorAgent LLM output against schema and constraints."""

    @staticmethod
    def _strip_markdown_code_blocks(response: str) -> str:
        """
        Strip markdown code blocks and comments from LLM response.
        Handles formats like:
        - ```json\n{...}\n```
        - ```\n{...}\n```
        - {... } (plain JSON)
        - Removes // comments (common with Ollama models)
        """
        response = response.strip()

        # Check for markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Find closing ``` and remove it and everything after
            for i, line in enumerate(lines):
                if line.strip() == "```":
                    lines = lines[:i]
                    break
            response = "\n".join(lines)

        # Remove JavaScript-style comments (// ...) that Ollama models often add
        # This is a simple approach - remove everything after // on each line
        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            # Find // and remove everything after it
            comment_pos = line.find("//")
            if comment_pos != -1:
                # Keep everything before the comment
                line = line[:comment_pos].rstrip()
            cleaned_lines.append(line)
        response = "\n".join(cleaned_lines)

        # Fix trailing commas before closing braces/brackets (common JSON error)
        # Replace ", }" with " }" and ", ]" with " ]"
        import re
        response = re.sub(r',(\s*[}\]])', r'\1', response)

        return response.strip()

    @staticmethod
    def validate_structure(response: str) -> ValidationResult:
        """Validate JSON structure and required keys."""
        # Strip markdown code blocks if present (common with Ollama models)
        cleaned_response = EngineValidator._strip_markdown_code_blocks(response)

        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            # Add debug info to see what the LLM actually returned
            error_msg = f"Invalid JSON: {e}\n\nCleaned response:\n{cleaned_response[:500]}"
            return ValidationResult(success=False, error=error_msg)

        required_keys = ["state_updates", "events", "agent_messages", "reasoning"]
        missing = [key for key in required_keys if key not in data]

        if missing:
            return ValidationResult(
                success=False,
                error=f"Missing required keys: {', '.join(missing)}"
            )

        # Validate state_updates structure
        if not isinstance(data["state_updates"], dict):
            return ValidationResult(success=False, error="state_updates must be a dict")

        if "global_vars" not in data["state_updates"]:
            return ValidationResult(success=False, error="state_updates missing global_vars")

        if "agent_vars" not in data["state_updates"]:
            return ValidationResult(success=False, error="state_updates missing agent_vars")

        return ValidationResult(success=True)

    @staticmethod
    def validate_references(
        output: Dict[str, Any],
        agents: List[str],
        global_var_defs: Dict[str, Any],
        agent_var_defs: Dict[str, Any]
    ) -> ValidationResult:
        """Validate agent names and variable names exist."""
        # Check agent names in state_updates
        for agent_name in output["state_updates"]["agent_vars"].keys():
            if agent_name not in agents:
                return ValidationResult(
                    success=False,
                    error=f"Unknown agent '{agent_name}' in state_updates"
                )

        # Check agent names in agent_messages
        for agent_name in output["agent_messages"].keys():
            if agent_name not in agents:
                return ValidationResult(
                    success=False,
                    error=f"Unknown agent '{agent_name}' in agent_messages"
                )

        # Check global variable names
        for var_name in output["state_updates"]["global_vars"].keys():
            if var_name not in global_var_defs:
                return ValidationResult(
                    success=False,
                    error=f"Unknown global variable '{var_name}'"
                )

        # Check agent variable names
        for agent_name, vars_dict in output["state_updates"]["agent_vars"].items():
            for var_name in vars_dict.keys():
                if var_name not in agent_var_defs:
                    return ValidationResult(
                        success=False,
                        error=f"Unknown agent variable '{var_name}' for agent '{agent_name}'"
                    )

        return ValidationResult(success=True)

    @staticmethod
    def validate_types(
        output: Dict[str, Any],
        global_var_defs: Dict[str, Any],
        agent_var_defs: Dict[str, Any]
    ) -> ValidationResult:
        """Validate variable values match defined types."""
        # Validate global variables
        for var_name, value in output["state_updates"]["global_vars"].items():
            expected_type = global_var_defs[var_name]["type"]
            if not EngineValidator._check_type(value, expected_type):
                return ValidationResult(
                    success=False,
                    error=f"Global var '{var_name}': expected {expected_type}, got {type(value).__name__}"
                )

        # Validate agent variables
        for agent_name, vars_dict in output["state_updates"]["agent_vars"].items():
            for var_name, value in vars_dict.items():
                expected_type = agent_var_defs[var_name]["type"]
                if not EngineValidator._check_type(value, expected_type):
                    return ValidationResult(
                        success=False,
                        error=f"Agent var '{var_name}' for '{agent_name}': expected {expected_type}, got {type(value).__name__}"
                    )

        return ValidationResult(success=True)

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if expected_type == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "bool":
            return isinstance(value, bool)
        elif expected_type == "list":
            return isinstance(value, list)
        elif expected_type == "dict":
            return isinstance(value, dict)
        return False

    @staticmethod
    def apply_constraints(
        state_updates: Dict[str, Any],
        global_var_defs: Dict[str, Any],
        agent_var_defs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[ConstraintHit]]:
        """Apply min/max constraints, return clamped values and hits."""
        clamped_updates = {
            "global_vars": {},
            "agent_vars": {}
        }
        constraint_hits = []

        # Process global variables
        for var_name, value in state_updates["global_vars"].items():
            var_def = global_var_defs[var_name]
            clamped_value, hit = EngineValidator._clamp_value(
                value, var_def, agent_name=None, var_name=var_name
            )
            clamped_updates["global_vars"][var_name] = clamped_value
            if hit:
                constraint_hits.append(hit)

        # Process agent variables
        for agent_name, vars_dict in state_updates["agent_vars"].items():
            clamped_updates["agent_vars"][agent_name] = {}
            for var_name, value in vars_dict.items():
                var_def = agent_var_defs[var_name]
                clamped_value, hit = EngineValidator._clamp_value(
                    value, var_def, agent_name=agent_name, var_name=var_name
                )
                clamped_updates["agent_vars"][agent_name][var_name] = clamped_value
                if hit:
                    constraint_hits.append(hit)

        return clamped_updates, constraint_hits

    @staticmethod
    def _clamp_value(
        value: Any,
        var_def: Dict[str, Any],
        agent_name: Optional[str],
        var_name: str
    ) -> Tuple[Any, Optional[ConstraintHit]]:
        """Clamp a value to constraints, return clamped value and hit if any."""
        var_type = var_def["type"]

        # Only numeric types have constraints
        if var_type not in ("int", "float"):
            return value, None

        min_val = var_def.get("min")
        max_val = var_def.get("max")

        # Check min constraint
        if min_val is not None and value < min_val:
            hit = ConstraintHit(
                agent_name=agent_name,
                var_name=var_name,
                attempted_value=value,
                clamped_value=min_val,
                constraint_type="min"
            )
            return min_val, hit

        # Check max constraint
        if max_val is not None and value > max_val:
            hit = ConstraintHit(
                agent_name=agent_name,
                var_name=var_name,
                attempted_value=value,
                clamped_value=max_val,
                constraint_type="max"
            )
            return max_val, hit

        return value, None

    @staticmethod
    def validate_locations(
        output: Dict[str, Any],
        spatial_graph: Any  # Optional[SpatialGraph] - use Any to avoid circular import
    ) -> ValidationResult:
        """
        Validate that location values in state updates are valid node IDs.

        Args:
            output: The parsed LLM output
            spatial_graph: The spatial graph to validate against (can be None)

        Returns:
            ValidationResult indicating if locations are valid
        """
        if spatial_graph is None:
            return ValidationResult(success=True)

        agent_vars = output.get("state_updates", {}).get("agent_vars", {})

        for agent_name, vars_dict in agent_vars.items():
            if "location" in vars_dict:
                location = vars_dict["location"]
                if location not in spatial_graph:
                    valid_nodes = spatial_graph.get_node_ids()
                    valid_str = ", ".join(valid_nodes[:5])
                    if len(valid_nodes) > 5:
                        valid_str += f"... ({len(valid_nodes)} total)"

                    return ValidationResult(
                        success=False,
                        error=(
                            f"Invalid location '{location}' for agent '{agent_name}'. "
                            f"Valid locations: {valid_str}"
                        )
                    )

        return ValidationResult(success=True)
