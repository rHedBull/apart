"""
Modular behavior system for simulation scenarios.

This module provides composable behavior components that can be mixed
and matched to create realistic simulations with minimal configuration.
"""

from modules.models import (
    BehaviorModule,
    ModuleVariable,
    ModuleDynamic,
    ModuleConstraint,
    ModuleAgentEffect,
    ModuleConfigField,
    ComposedModules,
)
from modules.loader import ModuleLoader, ModuleRegistry
from modules.composer import ModuleComposer
from modules.map_loader import load_map_file, MapLoadError

__all__ = [
    "BehaviorModule",
    "ModuleVariable",
    "ModuleDynamic",
    "ModuleConstraint",
    "ModuleAgentEffect",
    "ModuleConfigField",
    "ComposedModules",
    "ModuleLoader",
    "ModuleRegistry",
    "ModuleComposer",
    "load_map_file",
    "MapLoadError",
]
