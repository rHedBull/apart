"""
Map file loader for territory modules.

Loads map YAML files and converts them to SpatialGraph objects.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.spatial_graph import SpatialGraph, Node, Edge
from utils.movement_validator import MovementConfig


class MapLoadError(Exception):
    """Error loading a map file."""
    pass


def load_map_file(
    map_path: str | Path,
    base_dir: Path | None = None
) -> Tuple[SpatialGraph, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load a map file and create a SpatialGraph.

    Args:
        map_path: Path to the map YAML file
        base_dir: Base directory for relative paths (defaults to src/)

    Returns:
        Tuple of (SpatialGraph, map_metadata, geojson_data)

    Raises:
        MapLoadError: If map file not found or invalid
    """
    import json

    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # src/

    full_path = base_dir / map_path

    if not full_path.exists():
        raise MapLoadError(f"Map file not found: {full_path}")

    try:
        with open(full_path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise MapLoadError(f"Invalid YAML in map file: {e}")

    if not raw:
        raise MapLoadError(f"Empty map file: {full_path}")

    graph, metadata = parse_map_data(raw, str(full_path))

    # Load GeoJSON if specified
    geojson_data = None
    geojson_path = raw.get("map", {}).get("geojson")
    if geojson_path:
        geojson_full_path = full_path.parent / geojson_path
        if not geojson_full_path.exists():
            raise MapLoadError(f"GeoJSON file not found: {geojson_full_path}")

        try:
            with open(geojson_full_path, "r") as f:
                geojson_data = json.load(f)
        except json.JSONDecodeError as e:
            raise MapLoadError(f"Invalid JSON in GeoJSON file: {e}")

        # Validate that region nodes have matching features
        feature_ids = set()
        for feature in geojson_data.get("features", []):
            fid = feature.get("id") or feature.get("properties", {}).get("id")
            if fid:
                feature_ids.add(str(fid))

        for node_data in raw.get("nodes", []):
            if node_data.get("type") == "region":
                node_id = str(node_data["id"])
                if node_id not in feature_ids:
                    raise MapLoadError(
                        f"Region node '{node_id}' has no matching GeoJSON feature"
                    )

    return graph, metadata, geojson_data


def parse_map_data(data: Dict[str, Any], source: str = "unknown") -> Tuple[SpatialGraph, Dict[str, Any]]:
    """
    Parse map data into a SpatialGraph.

    Args:
        data: Parsed YAML data
        source: Source file path for error messages

    Returns:
        Tuple of (SpatialGraph, map_metadata)
    """
    graph = SpatialGraph()

    # Extract metadata
    metadata = data.get("map", {})

    # Parse nodes
    nodes_config = data.get("nodes", [])
    if not isinstance(nodes_config, list):
        raise MapLoadError(f"'nodes' must be a list in {source}")

    for node_data in nodes_config:
        if not isinstance(node_data, dict):
            raise MapLoadError(f"Each node must be a dictionary in {source}")

        if "id" not in node_data:
            raise MapLoadError(f"Each node must have an 'id' field in {source}")

        # Parse coordinates if present
        coords = node_data.get("coordinates")
        coordinates = tuple(coords) if coords else None

        node = Node(
            id=str(node_data["id"]),
            name=str(node_data.get("name", node_data["id"])),
            type=str(node_data.get("type", "location")),
            properties=node_data.get("properties", {}),
            conditions=node_data.get("conditions", []),
            coordinates=coordinates,
        )
        graph.add_node(node)

    # Parse edges
    edges_config = data.get("edges", [])
    if not isinstance(edges_config, list):
        raise MapLoadError(f"'edges' must be a list in {source}")

    for edge_data in edges_config:
        if not isinstance(edge_data, dict):
            raise MapLoadError(f"Each edge must be a dictionary in {source}")

        if "from" not in edge_data or "to" not in edge_data:
            raise MapLoadError(f"Each edge must have 'from' and 'to' fields in {source}")

        from_node = str(edge_data["from"])
        to_node = str(edge_data["to"])

        # Validate node references
        if from_node not in graph:
            raise MapLoadError(f"Edge references unknown node '{from_node}' in {source}")
        if to_node not in graph:
            raise MapLoadError(f"Edge references unknown node '{to_node}' in {source}")

        # Map 'distance' to 'traversal_cost' if not explicitly set
        properties = edge_data.get("properties", {})
        if "traversal_cost" not in properties and "travel_cost" in properties:
            properties["traversal_cost"] = properties["travel_cost"]

        edge = Edge(
            from_node=from_node,
            to_node=to_node,
            type=str(edge_data.get("type", "connection")),
            directed=edge_data.get("directed", False),
            properties=properties
        )
        graph.add_edge(edge)

    # Add initial_control to metadata
    if "initial_control" in data:
        metadata["initial_control"] = data["initial_control"]

    return graph, metadata


def create_movement_config(module_config: Dict[str, Any]) -> MovementConfig:
    """
    Create MovementConfig from module config.

    Args:
        module_config: Module's config_values dict

    Returns:
        MovementConfig object
    """
    movement = module_config.get("movement", {})

    return MovementConfig(
        default_budget_per_step=float(movement.get("budget_per_step", 20.0)),
        allow_multi_hop=movement.get("allow_multi_hop", True),
        blocked_edge_types=movement.get("blocked_edge_types", [])
    )
