"""
LLM-friendly spatial query formatting.

Provides a query engine that formats spatial graph information
in a way that's easy for LLMs to understand and use in prompts.
"""

from typing import Any, Dict, List, Optional
from utils.spatial_graph import SpatialGraph, PathResult


class SpatialQueryEngine:
    """
    Formats spatial graph queries for LLM consumption.

    Provides human-readable descriptions of:
    - Neighboring locations and their connections
    - Paths between locations with costs
    - Reachability information
    - Nodes within range
    """

    def __init__(self, graph: SpatialGraph):
        self.graph = graph

    def query_neighbors(self, node_id: str) -> str:
        """
        Get a formatted description of neighboring nodes.

        Args:
            node_id: The node to query neighbors for

        Returns:
            Human-readable description of adjacent locations
        """
        node = self.graph.get_node(node_id)
        if not node:
            return f"Unknown location: {node_id}"

        neighbors = self.graph.get_neighbors(node_id)

        if not neighbors:
            return f"{node.name} has no accessible connections."

        lines = [f"Adjacent to {node.name}:"]

        for neighbor_id, edge in neighbors:
            neighbor = self.graph.get_node(neighbor_id)
            neighbor_name = neighbor.name if neighbor else neighbor_id

            # Format edge properties
            props = []
            if edge.distance_km > 0:
                props.append(f"{edge.distance_km:.0f}km")
            if edge.travel_time_hours > 0:
                props.append(f"{edge.travel_time_hours:.1f}h")
            if edge.traversal_cost > 0:
                props.append(f"cost:{edge.traversal_cost:.0f}")

            props_str = f" ({', '.join(props)})" if props else ""
            lines.append(f"  - {neighbor_name} [{edge.type}]{props_str}")

        return "\n".join(lines)

    def query_path(self, from_node: str, to_node: str) -> str:
        """
        Get a formatted description of the path between two nodes.

        Args:
            from_node: Starting node ID
            to_node: Destination node ID

        Returns:
            Human-readable path description with costs
        """
        from_node_obj = self.graph.get_node(from_node)
        to_node_obj = self.graph.get_node(to_node)

        from_name = from_node_obj.name if from_node_obj else from_node
        to_name = to_node_obj.name if to_node_obj else to_node

        if from_node == to_node:
            return f"Already at {from_name}."

        result = self.graph.shortest_path(from_node, to_node)

        if not result.exists:
            return f"No path exists from {from_name} to {to_name}."

        # Build path description
        path_names = []
        for node_id in result.path:
            node = self.graph.get_node(node_id)
            path_names.append(node.name if node else node_id)

        lines = [f"Path from {from_name} to {to_name}:"]
        lines.append(f"  Route: {' -> '.join(path_names)}")
        lines.append(f"  Total distance: {result.total_distance:.0f} km")
        lines.append(f"  Total travel time: {result.total_time:.1f} hours")
        lines.append(f"  Total movement cost: {result.total_cost:.0f}")

        # Detail each leg
        if len(result.edges_used) > 1:
            lines.append("  Legs:")
            for edge in result.edges_used:
                from_n = self.graph.get_node(edge.from_node)
                to_n = self.graph.get_node(edge.to_node)
                from_name_leg = from_n.name if from_n else edge.from_node
                to_name_leg = to_n.name if to_n else edge.to_node
                lines.append(
                    f"    {from_name_leg} -> {to_name_leg} "
                    f"[{edge.type}] ({edge.distance_km:.0f}km, cost:{edge.traversal_cost:.0f})"
                )

        return "\n".join(lines)

    def query_reachable(self, from_node: str, to_node: str) -> str:
        """
        Check and explain reachability between two nodes.

        Args:
            from_node: Starting node ID
            to_node: Destination node ID

        Returns:
            Human-readable reachability explanation
        """
        from_node_obj = self.graph.get_node(from_node)
        to_node_obj = self.graph.get_node(to_node)

        from_name = from_node_obj.name if from_node_obj else from_node
        to_name = to_node_obj.name if to_node_obj else to_node

        if from_node == to_node:
            return f"Already at {from_name}."

        result = self.graph.shortest_path(from_node, to_node)

        if result.exists:
            return (
                f"Yes, {to_name} is reachable from {from_name}. "
                f"Shortest path: {len(result.path) - 1} hop(s), "
                f"{result.total_distance:.0f}km, cost {result.total_cost:.0f}."
            )
        else:
            # Check if blocked edge types might be the reason
            blocked = self.graph.get_blocked_edge_types()
            if blocked:
                return (
                    f"No, {to_name} is NOT reachable from {from_name}. "
                    f"Blocked edge types: {', '.join(blocked)}."
                )
            return f"No, {to_name} is NOT reachable from {from_name}."

    def query_within_range(
        self,
        node_id: str,
        max_hops: int
    ) -> str:
        """
        Get all nodes reachable within N hops.

        Args:
            node_id: Starting node ID
            max_hops: Maximum number of hops

        Returns:
            Human-readable list of reachable locations by hop count
        """
        node = self.graph.get_node(node_id)
        if not node:
            return f"Unknown location: {node_id}"

        reachable = self.graph.nodes_within_hops(node_id, max_hops)

        if len(reachable) <= 1:
            return f"No locations reachable within {max_hops} hop(s) from {node.name}."

        # Group by hop count
        by_hops: Dict[int, List[str]] = {}
        for n_id, hops in reachable.items():
            if n_id == node_id:
                continue
            if hops not in by_hops:
                by_hops[hops] = []
            n = self.graph.get_node(n_id)
            by_hops[hops].append(n.name if n else n_id)

        lines = [f"Locations reachable from {node.name} within {max_hops} hop(s):"]
        for hops in sorted(by_hops.keys()):
            hop_str = "hop" if hops == 1 else "hops"
            locations = ", ".join(sorted(by_hops[hops]))
            lines.append(f"  {hops} {hop_str}: {locations}")

        return "\n".join(lines)

    def query_within_budget(
        self,
        node_id: str,
        budget: float
    ) -> str:
        """
        Get all nodes reachable within a movement budget.

        Args:
            node_id: Starting node ID
            budget: Maximum movement cost

        Returns:
            Human-readable list of reachable locations with costs
        """
        node = self.graph.get_node(node_id)
        if not node:
            return f"Unknown location: {node_id}"

        reachable = self.graph.nodes_within_cost(node_id, budget)

        if len(reachable) <= 1:
            return f"No locations reachable within budget {budget:.0f} from {node.name}."

        lines = [f"Locations reachable from {node.name} with budget {budget:.0f}:"]

        # Sort by cost
        sorted_nodes = sorted(
            [(n_id, cost) for n_id, cost in reachable.items() if n_id != node_id],
            key=lambda x: x[1]
        )

        for n_id, cost in sorted_nodes:
            n = self.graph.get_node(n_id)
            name = n.name if n else n_id
            remaining = budget - cost
            lines.append(f"  - {name}: cost {cost:.0f} (remaining: {remaining:.0f})")

        return "\n".join(lines)

    def get_spatial_summary(
        self,
        agent_locations: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Get a complete spatial context summary for LLM prompts.

        Args:
            agent_locations: Optional dict mapping agent names to their current location IDs

        Returns:
            Comprehensive spatial context for prompt inclusion
        """
        lines = ["=== SPATIAL GRAPH ==="]

        # List all locations
        nodes = self.graph.get_nodes()
        if not nodes:
            return "No spatial graph configured."

        lines.append("\nLocations:")
        for node in sorted(nodes, key=lambda n: n.name):
            # Check if agents are here
            agents_here = []
            if agent_locations:
                for agent_name, loc_id in agent_locations.items():
                    if loc_id == node.id:
                        agents_here.append(agent_name)

            agents_str = f" [Agents: {', '.join(agents_here)}]" if agents_here else ""
            props_str = ""
            if node.properties:
                props = [f"{k}={v}" for k, v in node.properties.items()]
                props_str = f" ({', '.join(props)})"

            lines.append(f"  {node.name} ({node.type}){props_str}{agents_str}")

            if node.conditions:
                for cond in node.conditions:
                    lines.append(f"    - {cond}")

        # List connections
        lines.append("\nConnections:")
        seen_edges = set()
        for node_id in self.graph.get_node_ids():
            for neighbor_id, edge in self.graph.get_neighbors(node_id, include_blocked=True):
                # For undirected edges, only show once
                edge_key = tuple(sorted([edge.from_node, edge.to_node])) + (edge.type,)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                from_n = self.graph.get_node(edge.from_node)
                to_n = self.graph.get_node(neighbor_id)
                from_name = from_n.name if from_n else edge.from_node
                to_name = to_n.name if to_n else neighbor_id

                blocked_str = " [BLOCKED]" if self.graph.is_edge_type_blocked(edge.type) else ""
                direction = "->" if edge.directed else "<->"

                lines.append(
                    f"  {from_name} {direction} {to_name}: "
                    f"{edge.type}{blocked_str} "
                    f"({edge.distance_km:.0f}km, {edge.travel_time_hours:.1f}h, cost:{edge.traversal_cost:.0f})"
                )

        # Show blocked edge types
        blocked = self.graph.get_blocked_edge_types()
        if blocked:
            lines.append(f"\nBlocked edge types: {', '.join(sorted(blocked))}")

        # Agent location tracking note
        if agent_locations:
            lines.append("\nAgent Locations:")
            for agent_name, loc_id in sorted(agent_locations.items()):
                node = self.graph.get_node(loc_id)
                loc_name = node.name if node else loc_id
                lines.append(f"  {agent_name}: {loc_name}")

        # Spatial queries available
        lines.append("""
SPATIAL RULES:
- Agents can only move to adjacent locations (connected by edges)
- Movement costs are deducted from per-step budget
- Blocked edge types cannot be traversed
- Update agent 'location' variable to move them""")

        return "\n".join(lines)

    def format_movement_options(
        self,
        agent_name: str,
        current_location: str,
        budget: float
    ) -> str:
        """
        Format available movement options for an agent.

        Args:
            agent_name: Name of the agent
            current_location: Current location node ID
            budget: Remaining movement budget

        Returns:
            Formatted list of movement options
        """
        node = self.graph.get_node(current_location)
        if not node:
            return f"{agent_name} is at unknown location: {current_location}"

        reachable = self.graph.nodes_within_cost(current_location, budget)

        lines = [f"{agent_name} at {node.name} (budget: {budget:.0f}):"]

        if len(reachable) <= 1:
            lines.append("  No reachable destinations within budget.")
            return "\n".join(lines)

        lines.append("  Can move to:")
        sorted_nodes = sorted(
            [(n_id, cost) for n_id, cost in reachable.items() if n_id != current_location],
            key=lambda x: x[1]
        )

        for n_id, cost in sorted_nodes:
            n = self.graph.get_node(n_id)
            name = n.name if n else n_id
            lines.append(f"    - {name} (cost: {cost:.0f})")

        return "\n".join(lines)
