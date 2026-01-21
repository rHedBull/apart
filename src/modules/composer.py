"""
Module composer for combining multiple behavior modules.

Merges variables, dynamics, constraints, and generates prompt sections
from multiple modules into a unified ComposedModules object.
"""

from typing import Any, Dict, List, Optional

from modules.models import (
    BehaviorModule,
    ComposedModules,
    ModuleVariable,
)
from modules.loader import ModuleLoader


class ModuleComposer:
    """
    Composes multiple behavior modules into a unified structure.

    Usage:
        loader = ModuleLoader()
        modules = loader.load_many(["military_operations", "fog_of_war"])
        composer = ModuleComposer()
        composed = composer.compose(modules)
    """

    def compose(self, modules: List[BehaviorModule]) -> ComposedModules:
        """
        Merge all modules into a unified ComposedModules object.

        The ComposedModules class handles the actual merging in __post_init__.
        If territory_graph module is present with map_file config, loads the map.

        Args:
            modules: List of BehaviorModule objects to compose

        Returns:
            ComposedModules with all merged content
        """
        composed = ComposedModules(modules=modules)

        # Load spatial graph if territory_graph module has map_file config
        for module in modules:
            if module.name == "territory_graph" and module.config_values.get("map_file"):
                self._load_map_for_module(module, composed)
                break

        return composed

    def _load_map_for_module(
        self,
        module: BehaviorModule,
        composed: ComposedModules
    ) -> None:
        """Load map file and populate spatial graph in composed modules."""
        from modules.map_loader import load_map_file, create_movement_config

        map_file = module.config_values.get("map_file")
        if not map_file:
            return

        spatial_graph, metadata, geojson = load_map_file(map_file)
        movement_config = create_movement_config(module.config_values)

        composed.spatial_graph = spatial_graph
        composed.map_metadata = metadata
        composed.movement_config = movement_config
        composed.geojson = geojson

    def to_var_definitions(
        self,
        composed: ComposedModules,
        scope: str = "all"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert module variables to config_parser format.

        Args:
            composed: ComposedModules object
            scope: "global", "agent", or "all"

        Returns:
            Dictionary of variable definitions in config_parser format
        """
        result = {}

        if scope in ("global", "all"):
            result.update(composed.to_global_var_definitions())

        if scope in ("agent", "all"):
            result.update(composed.to_agent_var_definitions())

        return result

    def get_global_var_definitions(
        self,
        composed: ComposedModules
    ) -> Dict[str, Dict[str, Any]]:
        """Get global variable definitions for config merge."""
        return composed.to_global_var_definitions()

    def get_agent_var_definitions(
        self,
        composed: ComposedModules
    ) -> Dict[str, Dict[str, Any]]:
        """Get agent variable definitions for config merge."""
        return composed.to_agent_var_definitions()

    def to_prompt_sections(self, composed: ComposedModules) -> Dict[str, str]:
        """
        Generate all prompt sections for injection.

        Args:
            composed: ComposedModules object

        Returns:
            Dictionary with keys: "dynamics", "constraints", "agent_effects"
        """
        sections = {}

        dynamics = composed.to_dynamics_prompt()
        if dynamics:
            sections["dynamics"] = dynamics

        constraints = composed.to_constraints_prompt()
        if constraints:
            sections["constraints"] = constraints

        agent_effects = composed.to_agent_effects_prompt()
        if agent_effects:
            sections["agent_effects"] = agent_effects

        return sections


def compose_modules(
    module_names: List[str],
    modules_dir: Optional[str] = None
) -> ComposedModules:
    """
    Convenience function to load and compose modules in one step.

    Args:
        module_names: List of module names to load
        modules_dir: Optional custom modules directory

    Returns:
        ComposedModules object

    Example:
        composed = compose_modules(["military_operations", "fog_of_war"])
    """
    from pathlib import Path

    loader = ModuleLoader(Path(modules_dir) if modules_dir else None)
    modules = loader.load_many(module_names)
    return ModuleComposer().compose(modules)
