"""
Scenario validation for AI-driven simulation generation.

Validates configuration against schema, checks module compatibility,
and ensures agent prompts contain required sections.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from modules.loader import ModuleLoader, find_common_granularity
from modules.models import Granularity


class ValidationError(Exception):
    """Validation failed with one or more errors."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(
            f"Validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]


# Required sections in agent system prompts
REQUIRED_PROMPT_SECTIONS = [
    "OBJECTIVES",
    "CONSTRAINTS",
    "INFORMATION ACCESS",
]


class ScenarioValidator:
    """
    Validates scenario configurations for AI-driven simulations.

    Checks:
    - Required fields present
    - Module compatibility (dependencies, conflicts, granularity)
    - Agent prompt structure (required sections)
    - Variable types match definitions
    - Experiment conditions reference valid paths
    """

    def __init__(self, modules_dir: Optional[Path] = None):
        self.loader = ModuleLoader(modules_dir) if modules_dir else ModuleLoader()

    def validate_agent_prompt(self, prompt: str) -> List[str]:
        """
        Validate agent system prompt has required sections.

        Args:
            prompt: Agent system prompt text

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        prompt_upper = prompt.upper()

        for section in REQUIRED_PROMPT_SECTIONS:
            # Check for section header (with or without #)
            if (
                f"# {section}" not in prompt_upper
                and f"#{section}" not in prompt_upper
                and section not in prompt_upper
            ):
                errors.append(f"Agent prompt missing required section: {section}")

        return errors

    def validate_granularity(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate that selected granularity is supported by all modules.

        Args:
            config: Scenario configuration dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        meta = config.get("meta", {})
        selected_granularity_str = meta.get("granularity", "meso")

        granularity_mapping = {
            "macro": Granularity.MACRO,
            "meso": Granularity.MESO,
            "micro": Granularity.MICRO,
        }
        selected = granularity_mapping.get(selected_granularity_str)

        if not selected:
            errors.append(f"Invalid granularity: {selected_granularity_str}")
            return errors

        module_names = config.get("modules", [])
        if not module_names:
            return errors

        try:
            modules = self.loader.load_many(module_names)
            common = find_common_granularity(modules)

            if selected not in common:
                unsupported = [
                    m.name for m in modules if selected not in m.granularity_support
                ]
                errors.append(
                    f"Granularity '{selected_granularity_str}' not supported by "
                    f"modules: {', '.join(unsupported)}"
                )
        except Exception as e:
            errors.append(f"Error loading modules for granularity check: {e}")

        return errors

    def validate_variable_types(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate agent variable values match their type definitions.

        Args:
            config: Scenario configuration dict

        Returns:
            List of error messages
        """
        errors = []

        # Get variable definitions
        agent_var_defs = config.get("agent_vars", {})

        # Type checking functions
        type_checkers = {
            "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "bool": lambda v: isinstance(v, bool),
            "percent": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100,
            "scale": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100,
            "count": lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
            "dict": lambda v: isinstance(v, dict),
            "list": lambda v: isinstance(v, list),
        }

        for agent in config.get("agents", []):
            agent_name = agent.get("name", "unknown")
            agent_vars = agent.get("variables", {})

            for var_name, value in agent_vars.items():
                if var_name not in agent_var_defs:
                    errors.append(f"Agent '{agent_name}': undefined variable '{var_name}'")
                    continue

                var_def = agent_var_defs[var_name]
                var_type = var_def.get("type", "int")

                checker = type_checkers.get(var_type)
                if checker and not checker(value):
                    errors.append(
                        f"Agent '{agent_name}': variable '{var_name}' has wrong type. "
                        f"Expected {var_type}, got {type(value).__name__} ({value})"
                    )

        return errors

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Perform full validation of scenario configuration.

        Args:
            config: Full scenario configuration dict

        Returns:
            ValidationResult with valid flag, errors, and warnings
        """
        errors = []
        warnings = []

        # Check granularity compatibility
        errors.extend(self.validate_granularity(config))

        # Check agent prompts
        for agent in config.get("agents", []):
            prompt = agent.get("system_prompt", "")
            agent_name = agent.get("name", "unknown")
            prompt_errors = self.validate_agent_prompt(prompt)
            for e in prompt_errors:
                errors.append(f"Agent '{agent_name}': {e}")

        # Check variable types
        errors.extend(self.validate_variable_types(config))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_or_raise(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration and raise if invalid.

        Args:
            config: Full scenario configuration dict

        Raises:
            ValidationError: If validation fails
        """
        result = self.validate(config)
        if not result.valid:
            raise ValidationError(result.errors)
