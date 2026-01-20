"""
Graph-based spatial modeling for geopolitical simulations.

Provides data structures and algorithms for representing locations as nodes
and connections between them as edges with properties like distance, travel time,
and traversal costs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import heapq


@dataclass
class Node:
    """Represents a location in the spatial graph."""
    id: str
    name: str
    type: str  # nation, city, region, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)


@dataclass
class Edge:
    """Represents a connection between two locations."""
    from_node: str
    to_node: str
    type: str  # maritime, land, air, etc.
    directed: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def distance_km(self) -> float:
        """Get distance in kilometers."""
        return self.properties.get("distance_km", 0)

    @property
    def travel_time_hours(self) -> float:
        """Get travel time in hours."""
        return self.properties.get("travel_time_hours", 0)

    @property
    def traversal_cost(self) -> float:
        """Get traversal cost for movement budget."""
        return self.properties.get("traversal_cost", 1)


@dataclass
class PathResult:
    """Result of a pathfinding operation."""
    exists: bool
    path: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_distance: float = 0.0
    total_time: float = 0.0
    edges_used: List[Edge] = field(default_factory=list)


class SpatialGraph:
    """
    Graph representation of spatial relationships between locations.

    Supports:
    - Directed and undirected edges
    - Multiple edge types (maritime, land, air)
    - Dynamic edge blocking (e.g., blockades)
    - Pathfinding with Dijkstra's algorithm
    - BFS for nodes within N hops
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._adjacency: Dict[str, List[Edge]] = {}
        self._blocked_edge_types: Set[str] = set()

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # Ensure nodes exist
        if edge.from_node not in self._nodes:
            raise ValueError(f"Node '{edge.from_node}' does not exist")
        if edge.to_node not in self._nodes:
            raise ValueError(f"Node '{edge.to_node}' does not exist")

        # Add forward edge
        self._adjacency[edge.from_node].append(edge)

        # Add reverse edge if undirected
        if not edge.directed:
            reverse_edge = Edge(
                from_node=edge.to_node,
                to_node=edge.from_node,
                type=edge.type,
                directed=False,
                properties=edge.properties.copy()
            )
            self._adjacency[edge.to_node].append(reverse_edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[Node]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_node_ids(self) -> List[str]:
        """Get all node IDs."""
        return list(self._nodes.keys())

    def get_neighbors(self, node_id: str, include_blocked: bool = False) -> List[Tuple[str, Edge]]:
        """
        Get neighboring nodes and their connecting edges.

        Args:
            node_id: The node to get neighbors for
            include_blocked: If True, include edges with blocked types

        Returns:
            List of (neighbor_id, edge) tuples
        """
        if node_id not in self._adjacency:
            return []

        neighbors = []
        for edge in self._adjacency[node_id]:
            if include_blocked or edge.type not in self._blocked_edge_types:
                neighbors.append((edge.to_node, edge))

        return neighbors

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get the edge between two nodes (if exists)."""
        if from_node not in self._adjacency:
            return None

        for edge in self._adjacency[from_node]:
            if edge.to_node == to_node and edge.type not in self._blocked_edge_types:
                return edge

        return None

    def block_edge_type(self, edge_type: str) -> None:
        """Block all edges of a given type (e.g., 'maritime' for blockade)."""
        self._blocked_edge_types.add(edge_type)

    def unblock_edge_type(self, edge_type: str) -> None:
        """Unblock all edges of a given type."""
        self._blocked_edge_types.discard(edge_type)

    def get_blocked_edge_types(self) -> Set[str]:
        """Get the set of currently blocked edge types."""
        return self._blocked_edge_types.copy()

    def is_edge_type_blocked(self, edge_type: str) -> bool:
        """Check if an edge type is blocked."""
        return edge_type in self._blocked_edge_types

    def shortest_path(
        self,
        from_node: str,
        to_node: str,
        cost_key: str = "traversal_cost"
    ) -> PathResult:
        """
        Find the shortest path between two nodes using Dijkstra's algorithm.

        Args:
            from_node: Starting node ID
            to_node: Destination node ID
            cost_key: Property key to use for path cost (default: traversal_cost)

        Returns:
            PathResult with path details
        """
        if from_node not in self._nodes or to_node not in self._nodes:
            return PathResult(exists=False)

        if from_node == to_node:
            return PathResult(exists=True, path=[from_node])

        # Dijkstra's algorithm
        distances: Dict[str, float] = {from_node: 0}
        previous: Dict[str, Tuple[str, Edge]] = {}
        pq = [(0, from_node)]
        visited: Set[str] = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == to_node:
                break

            for neighbor, edge in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                cost = edge.properties.get(cost_key, 1)
                new_dist = current_dist + cost

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = (current, edge)
                    heapq.heappush(pq, (new_dist, neighbor))

        # Check if path exists
        if to_node not in previous and from_node != to_node:
            return PathResult(exists=False)

        # Reconstruct path
        path = []
        edges_used = []
        current = to_node

        while current != from_node:
            path.append(current)
            prev_node, edge = previous[current]
            edges_used.append(edge)
            current = prev_node

        path.append(from_node)
        path.reverse()
        edges_used.reverse()

        # Calculate totals
        total_cost = sum(e.traversal_cost for e in edges_used)
        total_distance = sum(e.distance_km for e in edges_used)
        total_time = sum(e.travel_time_hours for e in edges_used)

        return PathResult(
            exists=True,
            path=path,
            total_cost=total_cost,
            total_distance=total_distance,
            total_time=total_time,
            edges_used=edges_used
        )

    def nodes_within_hops(self, start_node: str, max_hops: int) -> Dict[str, int]:
        """
        Find all nodes reachable within N hops using BFS.

        Args:
            start_node: Starting node ID
            max_hops: Maximum number of hops

        Returns:
            Dictionary mapping node IDs to their hop distance from start
        """
        if start_node not in self._nodes:
            return {}

        reachable: Dict[str, int] = {start_node: 0}
        queue = [(start_node, 0)]

        while queue:
            current, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            for neighbor, _ in self.get_neighbors(current):
                if neighbor not in reachable:
                    reachable[neighbor] = hops + 1
                    queue.append((neighbor, hops + 1))

        return reachable

    def is_reachable(self, from_node: str, to_node: str) -> bool:
        """Check if a path exists between two nodes."""
        return self.shortest_path(from_node, to_node).exists

    def nodes_within_cost(
        self,
        start_node: str,
        max_cost: float,
        cost_key: str = "traversal_cost"
    ) -> Dict[str, float]:
        """
        Find all nodes reachable within a given cost budget.

        Args:
            start_node: Starting node ID
            max_cost: Maximum total cost
            cost_key: Property key to use for cost

        Returns:
            Dictionary mapping reachable node IDs to their minimum cost from start
        """
        if start_node not in self._nodes:
            return {}

        costs: Dict[str, float] = {start_node: 0}
        pq = [(0, start_node)]
        visited: Set[str] = set()

        while pq:
            current_cost, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            for neighbor, edge in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                edge_cost = edge.properties.get(cost_key, 1)
                new_cost = current_cost + edge_cost

                if new_cost <= max_cost:
                    if neighbor not in costs or new_cost < costs[neighbor]:
                        costs[neighbor] = new_cost
                        heapq.heappush(pq, (new_cost, neighbor))

        return costs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "properties": n.properties,
                    "conditions": n.conditions
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "from": e.from_node,
                    "to": e.to_node,
                    "type": e.type,
                    "directed": e.directed,
                    "properties": e.properties
                }
                for edges in self._adjacency.values()
                for e in edges
                if e.directed or e.from_node < e.to_node  # Avoid duplicating undirected edges
            ],
            "blocked_edge_types": list(self._blocked_edge_types)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialGraph":
        """Deserialize graph from dictionary."""
        graph = cls()

        # Add nodes
        for node_data in data.get("nodes", []):
            node = Node(
                id=node_data["id"],
                name=node_data["name"],
                type=node_data["type"],
                properties=node_data.get("properties", {}),
                conditions=node_data.get("conditions", [])
            )
            graph.add_node(node)

        # Add edges
        for edge_data in data.get("edges", []):
            edge = Edge(
                from_node=edge_data["from"],
                to_node=edge_data["to"],
                type=edge_data["type"],
                directed=edge_data.get("directed", False),
                properties=edge_data.get("properties", {})
            )
            graph.add_edge(edge)

        # Restore blocked edge types
        for edge_type in data.get("blocked_edge_types", []):
            graph.block_edge_type(edge_type)

        return graph

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes
