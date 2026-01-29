"""
Infrastructure file loader for trade_infrastructure module.

Loads infrastructure YAML files defining shipping lanes, ports,
canals, straits, and their connections.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class InfrastructureLoadError(Exception):
    """Error loading an infrastructure file."""
    pass


@dataclass
class InfrastructureNode:
    """A node in the trade infrastructure network."""
    id: str
    name: str
    type: str  # canal, strait, port, pipeline, hub
    properties: Dict[str, Any] = field(default_factory=dict)
    strategic_value: str = "medium"  # low, medium, high, critical
    controlling_actor: Optional[str] = None
    notes: str = ""

    @property
    def is_chokepoint(self) -> bool:
        """Check if this node is a chokepoint (canal or strait)."""
        return self.type in ("canal", "strait")


@dataclass
class InfrastructureEdge:
    """A connection in the trade infrastructure network."""
    id: str
    from_node: str
    to_node: str
    type: str  # shipping_lane, pipeline, rail, air_freight
    via: List[str] = field(default_factory=list)  # intermediate nodes
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def transit_days(self) -> float:
        """Get transit time in days."""
        return self.properties.get("transit_days", 0)

    @property
    def all_nodes(self) -> List[str]:
        """Get all nodes this edge passes through."""
        return [self.from_node] + self.via + [self.to_node]


@dataclass
class ReroutingOption:
    """An alternative route when a chokepoint is blocked."""
    blocked_node: str
    alternative_route: Optional[str]
    penalty_days: float
    cost_increase_pct: float
    constraint: str = ""
    notes: str = ""


@dataclass
class TradeInfrastructure:
    """
    Complete trade infrastructure representation.

    Contains nodes (ports, canals, straits), edges (shipping lanes),
    and rerouting options.
    """
    name: str
    description: str

    nodes: Dict[str, InfrastructureNode] = field(default_factory=dict)
    edges: Dict[str, InfrastructureEdge] = field(default_factory=dict)
    rerouting: Dict[str, List[ReroutingOption]] = field(default_factory=dict)

    # Raw metadata for extensions
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Optional[InfrastructureNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[InfrastructureEdge]:
        """Get edge by ID."""
        return self.edges.get(edge_id)

    def get_chokepoints(self) -> List[InfrastructureNode]:
        """Get all chokepoint nodes (canals and straits)."""
        return [n for n in self.nodes.values() if n.is_chokepoint]

    def get_critical_chokepoints(self) -> List[InfrastructureNode]:
        """Get chokepoints with critical strategic value."""
        return [
            n for n in self.nodes.values()
            if n.is_chokepoint and n.strategic_value == "critical"
        ]

    def get_ports(self) -> List[InfrastructureNode]:
        """Get all port nodes."""
        return [n for n in self.nodes.values() if n.type == "port"]

    def get_edges_through(self, node_id: str) -> List[InfrastructureEdge]:
        """Get all edges that pass through a node."""
        return [
            e for e in self.edges.values()
            if node_id in e.all_nodes
        ]

    def get_rerouting_options(self, blocked_node: str) -> List[ReroutingOption]:
        """Get rerouting options if a node is blocked."""
        return self.rerouting.get(blocked_node, [])

    def get_controlled_by(self, actor: str) -> List[InfrastructureNode]:
        """Get all nodes controlled by an actor."""
        return [
            n for n in self.nodes.values()
            if n.controlling_actor == actor
        ]

    def calculate_disruption_impact(self, node_id: str) -> Dict[str, Any]:
        """
        Calculate the impact of disrupting a node.

        Returns dict with affected_edges, rerouting_penalty, affected_trade_pct.
        """
        affected_edges = self.get_edges_through(node_id)
        rerouting = self.get_rerouting_options(node_id)

        min_penalty = min((r.penalty_days for r in rerouting), default=float('inf'))
        min_cost = min((r.cost_increase_pct for r in rerouting), default=float('inf'))

        return {
            "affected_edges": len(affected_edges),
            "affected_edge_ids": [e.id for e in affected_edges],
            "rerouting_available": len(rerouting) > 0,
            "min_rerouting_penalty_days": min_penalty if rerouting else None,
            "min_cost_increase_pct": min_cost if rerouting else None,
            "no_alternative": len(rerouting) == 0 or all(r.alternative_route is None for r in rerouting),
        }


def load_infrastructure_file(
    infra_path: str | Path,
    base_dir: Path | None = None
) -> TradeInfrastructure:
    """
    Load a trade infrastructure file.

    Args:
        infra_path: Path to the infrastructure YAML file
        base_dir: Base directory for relative paths (defaults to src/)

    Returns:
        TradeInfrastructure object

    Raises:
        InfrastructureLoadError: If file not found or invalid
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # src/

    full_path = base_dir / infra_path

    if not full_path.exists():
        raise InfrastructureLoadError(f"Infrastructure file not found: {full_path}")

    try:
        with open(full_path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InfrastructureLoadError(f"Invalid YAML in infrastructure file: {e}")

    if not raw:
        raise InfrastructureLoadError(f"Empty infrastructure file: {full_path}")

    return parse_infrastructure_data(raw, str(full_path))


def parse_infrastructure_data(
    data: Dict[str, Any],
    source: str = "unknown"
) -> TradeInfrastructure:
    """
    Parse infrastructure data into a TradeInfrastructure object.

    Args:
        data: Parsed YAML data
        source: Source file path for error messages

    Returns:
        TradeInfrastructure object
    """
    infra_meta = data.get("infrastructure", {})

    infra = TradeInfrastructure(
        name=infra_meta.get("name", "Unnamed Infrastructure"),
        description=infra_meta.get("description", ""),
    )

    # Parse nodes
    for node_data in data.get("nodes", []):
        if not isinstance(node_data, dict) or "id" not in node_data:
            raise InfrastructureLoadError(f"Invalid node in {source}: {node_data}")

        node = InfrastructureNode(
            id=node_data["id"],
            name=node_data.get("name", node_data["id"]),
            type=node_data.get("type", "hub"),
            properties=node_data.get("properties", {}),
            strategic_value=node_data.get("strategic_value", "medium"),
            controlling_actor=node_data.get("controlling_actor"),
            notes=node_data.get("notes", ""),
        )
        infra.nodes[node.id] = node

    # Parse edges
    for edge_data in data.get("edges", []):
        if not isinstance(edge_data, dict):
            raise InfrastructureLoadError(f"Invalid edge in {source}: {edge_data}")

        # Generate ID if not provided
        edge_id = edge_data.get("id", f"{edge_data.get('from', 'unknown')}_{edge_data.get('to', 'unknown')}")

        edge = InfrastructureEdge(
            id=edge_id,
            from_node=edge_data.get("from", ""),
            to_node=edge_data.get("to", ""),
            type=edge_data.get("type", "shipping_lane"),
            via=edge_data.get("via", []),
            properties=edge_data.get("properties", {}),
        )
        infra.edges[edge.id] = edge

    # Parse rerouting options
    rerouting_data = data.get("rerouting", {})
    for blocked_node, options in rerouting_data.items():
        infra.rerouting[blocked_node] = []

        if isinstance(options, dict):
            # Single alternative
            if options.get("alternative") is not None:
                infra.rerouting[blocked_node].append(ReroutingOption(
                    blocked_node=blocked_node,
                    alternative_route=options.get("alternative"),
                    penalty_days=options.get("penalty_days", 0),
                    cost_increase_pct=options.get("cost_increase_pct", 0),
                    constraint=options.get("capacity_constraint", options.get("constraint", "")),
                    notes=options.get("notes", ""),
                ))
            elif options.get("alternatives"):
                # Multiple alternatives
                for alt in options["alternatives"]:
                    infra.rerouting[blocked_node].append(ReroutingOption(
                        blocked_node=blocked_node,
                        alternative_route=alt.get("route"),
                        penalty_days=alt.get("penalty_days", 0),
                        cost_increase_pct=alt.get("cost_increase_pct", 0),
                        constraint=alt.get("constraint", ""),
                    ))
            else:
                # No alternative (e.g., Hormuz)
                infra.rerouting[blocked_node].append(ReroutingOption(
                    blocked_node=blocked_node,
                    alternative_route=None,
                    penalty_days=float('inf'),
                    cost_increase_pct=float('inf'),
                    notes=options.get("notes", "No alternative route"),
                ))

    # Store remaining data as metadata
    known_keys = {"infrastructure", "nodes", "edges", "rerouting"}
    for key, value in data.items():
        if key not in known_keys:
            infra.metadata[key] = value

    return infra
