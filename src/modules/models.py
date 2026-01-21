"""
Data models for behavior modules.

Behavior modules are composable components that add variables, dynamics,
constraints, and agent effects to a simulation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum


class VariableType(str, Enum):
    """Supported variable types for modules."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    PERCENT = "percent"  # 0-100 int
    SCALE = "scale"  # 0-100 int with semantic meaning
    COUNT = "count"  # non-negative int
    DICT = "dict"
    LIST = "list"


class ReinforcementType(str, Enum):
    """How a constraint is enforced."""
    HARD = "hard"  # Clamped, cannot be violated
    SOFT = "soft"  # Warning logged, but allowed
    GUIDED = "guided"  # Injected into prompt as guidance


@dataclass
class ModuleConfigField:
    """
    A configuration field for a behavior module.

    Config fields define what configuration the module needs from the scenario.
    For example, territory_graph needs a map_file path.
    """
    name: str
    type: str  # "string", "int", "float", "bool", "dict", "list"
    description: str
    required: bool = False
    default: Any = None


@dataclass
class ModuleVariable:
    """
    A variable provided by a behavior module.

    Variables are automatically added to the scenario when a module is enabled.
    They can be global (simulation-wide) or per-agent.
    """
    name: str
    type: VariableType
    description: str
    scope: Literal["global", "agent"] = "global"
    default: Any = None
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None

    # Optional: how this variable relates to other variables
    derived_from: Optional[str] = None  # e.g., "casualties = f(escalation_level)"
    affects: List[str] = field(default_factory=list)  # Variables this influences

    def to_var_definition(self) -> Dict[str, Any]:
        """Convert to the format expected by GameEngine."""
        defn = {
            "type": self._map_type(),
            "default": self._get_default(),
            "description": self.description,
        }
        if self.min is not None:
            defn["min"] = self.min
        if self.max is not None:
            defn["max"] = self.max
        return defn

    def _map_type(self) -> str:
        """Map module types to engine types."""
        mapping = {
            VariableType.INT: "int",
            VariableType.FLOAT: "float",
            VariableType.BOOL: "bool",
            VariableType.PERCENT: "int",
            VariableType.SCALE: "int",
            VariableType.COUNT: "int",
            VariableType.DICT: "dict",
            VariableType.LIST: "list",
        }
        return mapping[self.type]

    def _get_default(self) -> Any:
        """Get default value, applying type-specific defaults."""
        if self.default is not None:
            return self.default
        defaults = {
            VariableType.INT: 0,
            VariableType.FLOAT: 0.0,
            VariableType.BOOL: False,
            VariableType.PERCENT: 50,
            VariableType.SCALE: 50,
            VariableType.COUNT: 0,
            VariableType.DICT: {"value": "unknown"},
            VariableType.LIST: [],
        }
        return defaults[self.type]


@dataclass
class ModuleDynamic:
    """
    A behavioral dynamic that the simulator should model.

    Dynamics are natural language descriptions of how the simulation
    should behave. They're injected into the simulator's prompt.
    """
    description: str
    priority: int = 5  # 1-10, higher = more important
    conditions: List[str] = field(default_factory=list)  # When this applies
    examples: List[str] = field(default_factory=list)  # Concrete examples

    def to_prompt_section(self) -> str:
        """Format as a prompt section."""
        lines = [f"- {self.description}"]
        if self.conditions:
            lines.append(f"  Applies when: {', '.join(self.conditions)}")
        if self.examples:
            for ex in self.examples[:2]:  # Limit examples
                lines.append(f"  Example: {ex}")
        return "\n".join(lines)


@dataclass
class ModuleConstraint:
    """
    A realism constraint that limits simulation behavior.

    Constraints can be hard (enforced by validator) or soft (guidance only).
    """
    description: str
    enforcement: ReinforcementType = ReinforcementType.GUIDED

    # For hard constraints: validation rules
    variable: Optional[str] = None
    rule: Optional[str] = None  # e.g., "max_change_per_step: 10"

    # For probability constraints
    probability_per_step: Optional[float] = None

    def to_prompt_section(self) -> str:
        """Format as a prompt constraint."""
        prefix = ""
        if self.enforcement == ReinforcementType.HARD:
            prefix = "[ENFORCED] "
        elif self.enforcement == ReinforcementType.SOFT:
            prefix = "[WARNING] "
        return f"- {prefix}{self.description}"


@dataclass
class ModuleAgentEffect:
    """
    How a module affects agent behavior.

    Agent effects modify how agents perceive and respond to situations.
    """
    description: str
    applies_to: Literal["all", "specific"] = "all"
    agent_types: List[str] = field(default_factory=list)  # If specific

    # Perspective bias
    information_bias: Optional[str] = None  # How info is filtered
    confidence_modifier: Optional[float] = None  # +/- to confidence

    def to_prompt_section(self) -> str:
        """Format for agent prompt injection."""
        return f"- {self.description}"


@dataclass
class BehaviorModule:
    """
    A composable behavior module for simulations.

    Modules encapsulate a domain of behavior (military, economics, etc.)
    with all the variables, dynamics, constraints, and effects needed.
    """
    name: str
    description: str
    version: str = "1.0.0"

    # What this module provides
    variables: List[ModuleVariable] = field(default_factory=list)
    dynamics: List[ModuleDynamic] = field(default_factory=list)
    constraints: List[ModuleConstraint] = field(default_factory=list)
    agent_effects: List[ModuleAgentEffect] = field(default_factory=list)

    # Configuration schema (what the scenario must provide)
    config_schema: List[ModuleConfigField] = field(default_factory=list)
    # Actual config values (populated when module is loaded with config)
    config_values: Dict[str, Any] = field(default_factory=dict)

    # Dependencies on other modules
    requires: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)

    # Optional: module-specific prompt sections
    simulator_prompt_section: Optional[str] = None
    agent_prompt_section: Optional[str] = None

    # Event generation
    event_types: List[str] = field(default_factory=list)
    event_probabilities: Dict[str, float] = field(default_factory=dict)

    def get_global_variables(self) -> List[ModuleVariable]:
        """Get all global-scope variables."""
        return [v for v in self.variables if v.scope == "global"]

    def get_agent_variables(self) -> List[ModuleVariable]:
        """Get all agent-scope variables."""
        return [v for v in self.variables if v.scope == "agent"]

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value by key."""
        return self.config_values.get(key, default)

    def has_config_schema(self) -> bool:
        """Check if this module requires configuration."""
        return len(self.config_schema) > 0

    def get_required_config_fields(self) -> List[str]:
        """Get names of required config fields."""
        return [f.name for f in self.config_schema if f.required]

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate config values against schema.

        Returns list of error messages (empty if valid).
        """
        errors = []

        # Check required fields
        for field in self.config_schema:
            if field.required and field.name not in config:
                errors.append(
                    f"Module '{self.name}' requires config field '{field.name}'"
                )

        # Check for unknown fields
        known_fields = {f.name for f in self.config_schema}
        for key in config:
            if key not in known_fields:
                errors.append(
                    f"Unknown config field '{key}' for module '{self.name}'"
                )

        return errors

    def apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply config values, using defaults for missing optional fields.
        """
        for field in self.config_schema:
            if field.name in config:
                self.config_values[field.name] = config[field.name]
            elif field.default is not None:
                self.config_values[field.name] = field.default

    def to_prompt_sections(self) -> Dict[str, str]:
        """Generate prompt sections for this module."""
        sections = {}

        # Dynamics section
        if self.dynamics:
            dynamics_lines = [f"=== {self.name.upper()} DYNAMICS ==="]
            sorted_dynamics = sorted(self.dynamics, key=lambda d: -d.priority)
            for dynamic in sorted_dynamics:
                dynamics_lines.append(dynamic.to_prompt_section())
            sections["dynamics"] = "\n".join(dynamics_lines)

        # Constraints section
        if self.constraints:
            constraint_lines = [f"=== {self.name.upper()} CONSTRAINTS ==="]
            for constraint in self.constraints:
                constraint_lines.append(constraint.to_prompt_section())
            sections["constraints"] = "\n".join(constraint_lines)

        # Agent effects
        if self.agent_effects:
            effect_lines = [f"=== {self.name.upper()} EFFECTS ==="]
            for effect in self.agent_effects:
                effect_lines.append(effect.to_prompt_section())
            sections["agent_effects"] = "\n".join(effect_lines)

        # Custom sections
        if self.simulator_prompt_section:
            sections["custom_simulator"] = self.simulator_prompt_section
        if self.agent_prompt_section:
            sections["custom_agent"] = self.agent_prompt_section

        return sections


@dataclass
class ComposedModules:
    """
    The result of composing multiple modules together.

    Provides unified access to all variables, dynamics, constraints,
    and generated prompt sections.
    """
    modules: List[BehaviorModule]

    # Merged outputs
    global_variables: Dict[str, ModuleVariable] = field(default_factory=dict)
    agent_variables: Dict[str, ModuleVariable] = field(default_factory=dict)
    all_dynamics: List[ModuleDynamic] = field(default_factory=list)
    all_constraints: List[ModuleConstraint] = field(default_factory=list)
    all_agent_effects: List[ModuleAgentEffect] = field(default_factory=list)

    # Event generation
    event_probabilities: Dict[str, float] = field(default_factory=dict)

    # Spatial graph (loaded from territory module's map_file)
    spatial_graph: Optional[Any] = None  # SpatialGraph, avoid circular import
    map_metadata: Dict[str, Any] = field(default_factory=dict)
    movement_config: Optional[Any] = None  # MovementConfig
    geojson: Optional[Dict[str, Any]] = None  # GeoJSON data for map overlay

    def __post_init__(self):
        """Merge all module components."""
        for module in self.modules:
            # Merge variables
            for var in module.get_global_variables():
                if var.name not in self.global_variables:
                    self.global_variables[var.name] = var
            for var in module.get_agent_variables():
                if var.name not in self.agent_variables:
                    self.agent_variables[var.name] = var

            # Merge dynamics, constraints, effects
            self.all_dynamics.extend(module.dynamics)
            self.all_constraints.extend(module.constraints)
            self.all_agent_effects.extend(module.agent_effects)

            # Merge event probabilities
            self.event_probabilities.update(module.event_probabilities)

        # Sort dynamics by priority
        self.all_dynamics.sort(key=lambda d: -d.priority)

    def to_global_var_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Convert to format expected by config parser."""
        return {
            name: var.to_var_definition()
            for name, var in self.global_variables.items()
        }

    def to_agent_var_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Convert to format expected by config parser."""
        return {
            name: var.to_var_definition()
            for name, var in self.agent_variables.items()
        }

    def to_dynamics_prompt(self) -> str:
        """Generate unified dynamics prompt section."""
        if not self.all_dynamics:
            return ""

        lines = ["=== SIMULATION DYNAMICS ==="]
        lines.append("The following dynamics govern this simulation:\n")

        # Group by priority
        high_priority = [d for d in self.all_dynamics if d.priority >= 7]
        medium_priority = [d for d in self.all_dynamics if 4 <= d.priority < 7]
        low_priority = [d for d in self.all_dynamics if d.priority < 4]

        if high_priority:
            lines.append("CRITICAL DYNAMICS:")
            for d in high_priority:
                lines.append(d.to_prompt_section())
            lines.append("")

        if medium_priority:
            lines.append("IMPORTANT DYNAMICS:")
            for d in medium_priority:
                lines.append(d.to_prompt_section())
            lines.append("")

        if low_priority:
            lines.append("ADDITIONAL DYNAMICS:")
            for d in low_priority:
                lines.append(d.to_prompt_section())

        return "\n".join(lines)

    def to_constraints_prompt(self) -> str:
        """Generate unified constraints prompt section."""
        if not self.all_constraints:
            return ""

        lines = ["=== REALISM CONSTRAINTS ==="]
        lines.append("Enforce these constraints for realistic simulation:\n")

        # Group by enforcement type
        hard = [c for c in self.all_constraints if c.enforcement == ReinforcementType.HARD]
        soft = [c for c in self.all_constraints if c.enforcement == ReinforcementType.SOFT]
        guided = [c for c in self.all_constraints if c.enforcement == ReinforcementType.GUIDED]

        if hard:
            lines.append("HARD CONSTRAINTS (will be enforced):")
            for c in hard:
                lines.append(c.to_prompt_section())
            lines.append("")

        if soft:
            lines.append("SOFT CONSTRAINTS (warnings if violated):")
            for c in soft:
                lines.append(c.to_prompt_section())
            lines.append("")

        if guided:
            lines.append("GUIDANCE:")
            for c in guided:
                lines.append(c.to_prompt_section())

        return "\n".join(lines)

    def to_agent_effects_prompt(self) -> str:
        """Generate agent effects prompt section."""
        if not self.all_agent_effects:
            return ""

        lines = ["=== AGENT BEHAVIOR MODIFIERS ==="]
        for effect in self.all_agent_effects:
            lines.append(effect.to_prompt_section())

        return "\n".join(lines)

    def get_module_names(self) -> List[str]:
        """Get names of all active modules."""
        return [m.name for m in self.modules]
