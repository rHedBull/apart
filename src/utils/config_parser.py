from pathlib import Path
from typing import Any, Optional, Tuple
from utils.variables import VariableDefinition, VariableSet
from utils.persona_loader import PersonaLoader, resolve_persona_in_agent
from core.engine_models import ScriptedEvent
from utils.spatial_graph import SpatialGraph, Node, Edge
from utils.movement_validator import MovementConfig
from modules.models import ComposedModules
from modules.loader import ModuleLoader
from modules.composer import ModuleComposer


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


def parse_scripted_events(events_config: list[dict] | None) -> list[ScriptedEvent]:
    """
    Parse scripted events from YAML config.

    Args:
        events_config: List of event dictionaries

    Returns:
        List of ScriptedEvent objects
    """
    if not events_config:
        return []

    events = []
    for event_dict in events_config:
        if not isinstance(event_dict, dict):
            raise ValueError("Each scripted event must be a dictionary")

        required_fields = ["step", "type", "description"]
        for field in required_fields:
            if field not in event_dict:
                raise ValueError(f"Scripted event missing required field '{field}'")

        if not isinstance(event_dict["step"], int):
            raise ValueError("Scripted event 'step' must be an integer")

        events.append(ScriptedEvent(
            step=event_dict["step"],
            type=event_dict["type"],
            description=event_dict["description"]
        ))

    return events


def validate_engine_config(config: dict[str, Any]) -> None:
    """
    Validate engine configuration.

    Raises:
        ValueError: If engine configuration is invalid or missing
    """
    if "engine" not in config:
        raise ValueError("Configuration missing required 'engine' section")

    engine = config["engine"]

    if not isinstance(engine, dict):
        raise ValueError("'engine' must be a dictionary")

    required_fields = ["provider", "model", "system_prompt", "simulation_plan"]
    for field in required_fields:
        if field not in engine:
            raise ValueError(f"Engine configuration missing required field '{field}'")

    # Validate scripted events if present
    if "scripted_events" in engine:
        parse_scripted_events(engine["scripted_events"])


def parse_geography(geography_config: dict | None) -> dict:
    """
    Parse geography configuration from YAML.

    Args:
        geography_config: Geography configuration dictionary

    Returns:
        Parsed geography with normalized structure
    """
    if not geography_config:
        return {}

    if not isinstance(geography_config, dict):
        raise ValueError("Geography configuration must be a dictionary")

    geography = {}

    # Region (optional)
    if "region" in geography_config:
        geography["region"] = str(geography_config["region"])

    # Locations (optional - list of discrete locations)
    if "locations" in geography_config:
        locations = geography_config["locations"]
        if not isinstance(locations, list):
            raise ValueError("Geography 'locations' must be a list")

        parsed_locations = []
        for loc in locations:
            if isinstance(loc, dict):
                # Structured location with name, description, conditions
                if "name" not in loc:
                    raise ValueError("Each location must have a 'name' field")
                parsed_locations.append({
                    "name": str(loc["name"]),
                    "description": str(loc.get("description", "")),
                    "conditions": loc.get("conditions", [])
                })
            elif isinstance(loc, str):
                # Simple string location
                parsed_locations.append({
                    "name": loc,
                    "description": "",
                    "conditions": []
                })
            else:
                raise ValueError("Each location must be a string or dictionary")

        geography["locations"] = parsed_locations

    # Travel information (optional)
    if "travel" in geography_config:
        travel = geography_config["travel"]
        if isinstance(travel, dict):
            geography["travel"] = travel
        elif isinstance(travel, str):
            geography["travel"] = {"description": travel}
        else:
            raise ValueError("Geography 'travel' must be a string or dictionary")

    # Additional context (optional)
    if "context" in geography_config:
        geography["context"] = str(geography_config["context"])

    return geography


def parse_spatial_graph(
    geography_config: dict | None
) -> Tuple[Optional[SpatialGraph], Optional[MovementConfig]]:
    """
    Parse spatial graph configuration from YAML.

    Args:
        geography_config: Geography configuration dictionary with spatial_model: "graph"

    Returns:
        Tuple of (SpatialGraph, MovementConfig) or (None, None) if not using graph mode
    """
    if not geography_config:
        return None, None

    if not isinstance(geography_config, dict):
        return None, None

    # Check if spatial model is "graph"
    spatial_model = geography_config.get("spatial_model", "narrative")
    if spatial_model != "graph":
        return None, None

    graph = SpatialGraph()

    # Parse nodes
    nodes_config = geography_config.get("nodes", [])
    if not isinstance(nodes_config, list):
        raise ValueError("geography.nodes must be a list")

    for node_data in nodes_config:
        if not isinstance(node_data, dict):
            raise ValueError("Each node must be a dictionary")

        if "id" not in node_data:
            raise ValueError("Each node must have an 'id' field")

        node = Node(
            id=str(node_data["id"]),
            name=str(node_data.get("name", node_data["id"])),
            type=str(node_data.get("type", "location")),
            properties=node_data.get("properties", {}),
            conditions=node_data.get("conditions", [])
        )
        graph.add_node(node)

    # Parse edges
    edges_config = geography_config.get("edges", [])
    if not isinstance(edges_config, list):
        raise ValueError("geography.edges must be a list")

    for edge_data in edges_config:
        if not isinstance(edge_data, dict):
            raise ValueError("Each edge must be a dictionary")

        if "from" not in edge_data or "to" not in edge_data:
            raise ValueError("Each edge must have 'from' and 'to' fields")

        from_node = str(edge_data["from"])
        to_node = str(edge_data["to"])

        # Validate node references
        if from_node not in graph:
            raise ValueError(f"Edge references unknown node '{from_node}'")
        if to_node not in graph:
            raise ValueError(f"Edge references unknown node '{to_node}'")

        edge = Edge(
            from_node=from_node,
            to_node=to_node,
            type=str(edge_data.get("type", "connection")),
            directed=edge_data.get("directed", False),
            properties=edge_data.get("properties", {})
        )
        graph.add_edge(edge)

    # Parse movement configuration
    movement_config = geography_config.get("movement", {})
    config = MovementConfig(
        default_budget_per_step=float(movement_config.get("default_budget_per_step", 20.0)),
        allow_multi_hop=movement_config.get("allow_multi_hop", True),
        blocked_edge_types=movement_config.get("blocked_edge_types", [])
    )

    return graph, config


def validate_spatial_graph_config(config: dict[str, Any]) -> None:
    """Validate spatial graph configuration if present."""
    if "geography" not in config:
        return

    geography = config["geography"]
    if not isinstance(geography, dict):
        return

    spatial_model = geography.get("spatial_model", "narrative")
    if spatial_model != "graph":
        return

    # Validate by attempting to parse
    try:
        parse_spatial_graph(geography)
    except Exception as e:
        raise ValueError(f"Invalid spatial graph configuration: {e}") from e


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the entire configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    validate_engine_config(config)
    validate_agent_vars_config(config)
    validate_global_vars_config(config)

    # Validate agent configurations
    if "agents" in config:
        agent_vars_defs = config.get("agent_vars")
        for agent_config in config["agents"]:
            validate_agent_config(agent_config, agent_vars_defs)

    # Validate geography if present
    if "geography" in config:
        parse_geography(config["geography"])
        validate_spatial_graph_config(config)


def resolve_personas_in_config(
    config: dict[str, Any],
    personas_dir: Path | str | None = None
) -> dict[str, Any]:
    """
    Resolve all persona references in a configuration.

    Processes all agents with 'persona' keys, loading and merging
    the persona definitions.

    Args:
        config: Full scenario configuration
        personas_dir: Directory containing persona files (optional)

    Returns:
        Configuration with persona references resolved

    Example:
        Input config with:
            agents:
              - persona: geopolitical/usa_state_dept
                variables:
                  diplomatic_leverage: 90

        Output config with:
            agents:
              - name: "US State Department"
                system_prompt: "..."
                variables:
                  diplomatic_leverage: 90  # Agent override
                  military_readiness: 70   # From persona
    """
    if "agents" not in config:
        return config

    loader = PersonaLoader(personas_dir)
    result = config.copy()

    resolved_agents = []
    for agent_config in config["agents"]:
        resolved = resolve_persona_in_agent(agent_config, loader)
        resolved_agents.append(resolved)

    result["agents"] = resolved_agents
    return result


class ModuleConfigError(Exception):
    """Error with module configuration."""
    pass


def parse_modules(
    config: dict[str, Any],
    modules_dir: Path | str | None = None
) -> Optional[ComposedModules]:
    """
    Load and compose modules specified in configuration.

    Args:
        config: Full scenario configuration dictionary
        modules_dir: Optional custom directory for module definitions

    Returns:
        ComposedModules if modules are specified, None otherwise

    Raises:
        ModuleConfigError: If required config is missing or invalid

    Example:
        config = {
            "modules": ["territory_graph", "military_base"],
            "module_config": {
                "territory_graph": {
                    "map_file": "modules/maps/sample_conflict.yaml"
                }
            }
        }
        composed = parse_modules(config)
    """
    module_names = config.get("modules", [])
    if not module_names:
        return None

    modules_path = Path(modules_dir) if modules_dir else None
    loader = ModuleLoader(modules_path)
    modules = loader.load_many(module_names)

    # Get module config from scenario
    module_config = config.get("module_config", {})

    # Validate and apply config to each module
    all_errors = []
    for module in modules:
        if module.has_config_schema():
            module_cfg = module_config.get(module.name, {})
            errors = module.validate_config(module_cfg)
            if errors:
                all_errors.extend(errors)
            else:
                module.apply_config(module_cfg)

    if all_errors:
        raise ModuleConfigError(
            "Module configuration errors:\n  " + "\n  ".join(all_errors)
        )

    composer = ModuleComposer()
    return composer.compose(modules)


def merge_module_variables(
    config: dict[str, Any],
    composed: ComposedModules
) -> dict[str, Any]:
    """
    Merge module variables into config global_vars/agent_vars.

    Module variables are added to the config if they don't already exist.
    Existing config variables take precedence.

    Args:
        config: Full scenario configuration dictionary
        composed: ComposedModules with variables to merge

    Returns:
        Updated configuration dictionary with module variables merged

    Example:
        config = parse_modules(raw_config)
        composed = parse_modules(raw_config)
        config = merge_module_variables(config, composed)
    """
    result = config.copy()

    # Merge global variables
    global_var_defs = composed.to_global_var_definitions()
    if global_var_defs:
        existing_global = result.get("global_vars", {})
        merged_global = {}

        # Add module variables first (lower precedence)
        for var_name, var_def in global_var_defs.items():
            merged_global[var_name] = var_def

        # Override with existing config variables (higher precedence)
        for var_name, var_def in existing_global.items():
            merged_global[var_name] = var_def

        result["global_vars"] = merged_global

    # Merge agent variables
    agent_var_defs = composed.to_agent_var_definitions()
    if agent_var_defs:
        existing_agent = result.get("agent_vars", {})
        merged_agent = {}

        # Add module variables first (lower precedence)
        for var_name, var_def in agent_var_defs.items():
            merged_agent[var_name] = var_def

        # Override with existing config variables (higher precedence)
        for var_name, var_def in existing_agent.items():
            merged_agent[var_name] = var_def

        result["agent_vars"] = merged_agent

    return result
