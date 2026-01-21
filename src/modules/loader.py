"""
Module loader for behavior modules.

Loads YAML module definitions from the modules/definitions directory
and converts them to BehaviorModule objects.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set

from modules.models import (
    BehaviorModule,
    ModuleVariable,
    ModuleDynamic,
    ModuleConstraint,
    ModuleAgentEffect,
    VariableType,
    ReinforcementType,
)


# Default directory for built-in module definitions
DEFAULT_MODULES_DIR = Path(__file__).parent / "definitions"


class ModuleLoadError(Exception):
    """Error loading a behavior module."""
    pass


class ModuleDependencyError(Exception):
    """Error with module dependencies."""
    pass


class ModuleLoader:
    """
    Loads behavior module definitions from YAML files.

    Usage:
        loader = ModuleLoader()
        module = loader.load("military_operations")
        modules = loader.load_many(["military_operations", "fog_of_war"])
    """

    def __init__(self, modules_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            modules_dir: Directory containing module YAML files.
                         Defaults to src/modules/definitions/
        """
        self.modules_dir = Path(modules_dir) if modules_dir else DEFAULT_MODULES_DIR
        self._cache: Dict[str, BehaviorModule] = {}

    def load(self, module_name: str) -> BehaviorModule:
        """
        Load a single module by name.

        Args:
            module_name: Name of the module (without .yaml extension)

        Returns:
            BehaviorModule object

        Raises:
            ModuleLoadError: If module file not found or invalid
        """
        if module_name in self._cache:
            return self._cache[module_name]

        module_path = self.modules_dir / f"{module_name}.yaml"

        if not module_path.exists():
            available = self._list_available_modules()
            raise ModuleLoadError(
                f"Module '{module_name}' not found at {module_path}. "
                f"Available modules: {', '.join(available) if available else '(none)'}"
            )

        try:
            with open(module_path, "r") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ModuleLoadError(f"Invalid YAML in module '{module_name}': {e}")

        if not raw or "module" not in raw:
            raise ModuleLoadError(
                f"Module '{module_name}' missing required 'module' key"
            )

        module = self._parse_module(raw["module"], module_name)
        self._cache[module_name] = module
        return module

    def load_many(self, module_names: List[str]) -> List[BehaviorModule]:
        """
        Load multiple modules with dependency validation.

        Args:
            module_names: List of module names to load

        Returns:
            List of BehaviorModule objects

        Raises:
            ModuleLoadError: If any module fails to load
            ModuleDependencyError: If dependencies are violated
        """
        modules = []
        loaded_names: Set[str] = set()

        for name in module_names:
            module = self.load(name)
            modules.append(module)
            loaded_names.add(name)

        # Validate dependencies
        self._validate_dependencies(modules, loaded_names)

        return modules

    def _parse_module(self, data: dict, source_name: str) -> BehaviorModule:
        """Parse raw YAML data into a BehaviorModule."""
        name = data.get("name", source_name)

        # Parse variables
        variables = []
        vars_data = data.get("variables", {})

        for scope in ["global", "agent"]:
            scope_vars = vars_data.get(scope, [])
            for var_data in scope_vars:
                variables.append(self._parse_variable(var_data, scope))

        # Parse dynamics
        dynamics = []
        for dyn_data in data.get("dynamics", []):
            dynamics.append(self._parse_dynamic(dyn_data))

        # Parse constraints
        constraints = []
        for con_data in data.get("constraints", []):
            constraints.append(self._parse_constraint(con_data))

        # Parse agent effects
        agent_effects = []
        for effect_data in data.get("agent_effects", []):
            agent_effects.append(self._parse_agent_effect(effect_data))

        return BehaviorModule(
            name=name,
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            variables=variables,
            dynamics=dynamics,
            constraints=constraints,
            agent_effects=agent_effects,
            requires=data.get("requires", []),
            conflicts_with=data.get("conflicts_with", []),
            simulator_prompt_section=data.get("simulator_prompt_section"),
            agent_prompt_section=data.get("agent_prompt_section"),
            event_types=data.get("event_types", []),
            event_probabilities=data.get("event_probabilities", {}),
        )

    def _parse_variable(self, data: dict, scope: str) -> ModuleVariable:
        """Parse a variable definition."""
        var_type_str = data.get("type", "int")

        # Map string to enum
        type_mapping = {
            "int": VariableType.INT,
            "float": VariableType.FLOAT,
            "bool": VariableType.BOOL,
            "percent": VariableType.PERCENT,
            "scale": VariableType.SCALE,
            "count": VariableType.COUNT,
            "dict": VariableType.DICT,
            "list": VariableType.LIST,
        }

        var_type = type_mapping.get(var_type_str, VariableType.INT)

        return ModuleVariable(
            name=data["name"],
            type=var_type,
            description=data.get("description", ""),
            scope=scope,
            default=data.get("default"),
            min=data.get("min"),
            max=data.get("max"),
            derived_from=data.get("derived_from"),
            affects=data.get("affects", []),
        )

    def _parse_dynamic(self, data: dict) -> ModuleDynamic:
        """Parse a dynamic definition."""
        return ModuleDynamic(
            description=data.get("description", ""),
            priority=data.get("priority", 5),
            conditions=data.get("conditions", []),
            examples=data.get("examples", []),
        )

    def _parse_constraint(self, data: dict) -> ModuleConstraint:
        """Parse a constraint definition."""
        enforcement_str = data.get("enforcement", "guided")

        enforcement_mapping = {
            "hard": ReinforcementType.HARD,
            "soft": ReinforcementType.SOFT,
            "guided": ReinforcementType.GUIDED,
        }

        enforcement = enforcement_mapping.get(enforcement_str, ReinforcementType.GUIDED)

        return ModuleConstraint(
            description=data.get("description", ""),
            enforcement=enforcement,
            variable=data.get("variable"),
            rule=data.get("rule"),
            probability_per_step=data.get("probability_per_step"),
        )

    def _parse_agent_effect(self, data: dict) -> ModuleAgentEffect:
        """Parse an agent effect definition."""
        return ModuleAgentEffect(
            description=data.get("description", ""),
            applies_to=data.get("applies_to", "all"),
            agent_types=data.get("agent_types", []),
            information_bias=data.get("information_bias"),
            confidence_modifier=data.get("confidence_modifier"),
        )

    def _validate_dependencies(
        self,
        modules: List[BehaviorModule],
        loaded_names: Set[str]
    ) -> None:
        """Validate module dependencies."""
        for module in modules:
            # Check required modules
            for required in module.requires:
                if required not in loaded_names:
                    raise ModuleDependencyError(
                        f"Module '{module.name}' requires '{required}' which is not loaded. "
                        f"Add '{required}' to your modules list."
                    )

            # Check conflicts
            for conflict in module.conflicts_with:
                if conflict in loaded_names:
                    raise ModuleDependencyError(
                        f"Module '{module.name}' conflicts with '{conflict}'. "
                        f"These modules cannot be used together."
                    )

    def _list_available_modules(self) -> List[str]:
        """List available module names."""
        if not self.modules_dir.exists():
            return []
        return [
            p.stem for p in self.modules_dir.glob("*.yaml")
        ]

    def list_modules(self) -> List[str]:
        """
        List all available module names.

        Returns:
            List of module names (without .yaml extension)
        """
        return self._list_available_modules()

    def clear_cache(self) -> None:
        """Clear the module cache."""
        self._cache.clear()


class ModuleRegistry:
    """
    Registry of available behavior modules.

    Provides discovery and metadata for all available modules.
    """

    def __init__(self, modules_dir: Optional[Path] = None):
        self.loader = ModuleLoader(modules_dir)

    def list_available(self) -> List[str]:
        """List all available module names."""
        return self.loader.list_modules()

    def get_module_info(self, module_name: str) -> Dict:
        """
        Get module metadata without full loading.

        Args:
            module_name: Name of the module

        Returns:
            Dictionary with module metadata
        """
        module = self.loader.load(module_name)
        return {
            "name": module.name,
            "version": module.version,
            "description": module.description,
            "requires": module.requires,
            "conflicts_with": module.conflicts_with,
            "num_variables": len(module.variables),
            "num_dynamics": len(module.dynamics),
            "num_constraints": len(module.constraints),
        }

    def get_all_module_info(self) -> List[Dict]:
        """Get metadata for all available modules."""
        return [
            self.get_module_info(name)
            for name in self.list_available()
        ]
