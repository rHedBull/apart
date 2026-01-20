"""
Movement validation for spatial graph-based simulations.

Validates agent movements against the spatial graph, ensuring:
- Movements follow valid paths
- Movement costs are within budget
- Edge types are not blocked
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from utils.spatial_graph import SpatialGraph, PathResult


@dataclass
class MovementResult:
    """Result of a movement validation or execution."""
    valid: bool
    from_location: str
    to_location: str
    path: List[str]
    cost: float
    error: Optional[str] = None
    remaining_budget: float = 0.0


@dataclass
class MovementConfig:
    """Configuration for movement validation."""
    default_budget_per_step: float = 20.0
    allow_multi_hop: bool = True
    blocked_edge_types: List[str] = None

    def __post_init__(self):
        if self.blocked_edge_types is None:
            self.blocked_edge_types = []


class MovementValidator:
    """
    Validates and tracks agent movements on the spatial graph.

    Features:
    - Per-step movement budgets
    - Path validation (direct or multi-hop)
    - Dynamic edge blocking support
    - Movement cost tracking
    """

    def __init__(
        self,
        graph: SpatialGraph,
        config: Optional[MovementConfig] = None
    ):
        self.graph = graph
        self.config = config or MovementConfig()

        # Apply initial blocked edge types from config
        for edge_type in self.config.blocked_edge_types:
            self.graph.block_edge_type(edge_type)

        # Track agent budgets (reset each step)
        self._agent_budgets: Dict[str, float] = {}

        # Track current agent locations
        self._agent_locations: Dict[str, str] = {}

    def reset_budgets(self, agent_names: Optional[List[str]] = None) -> None:
        """
        Reset movement budgets for a new step.

        Args:
            agent_names: Optional list of agent names to reset (defaults to all)
        """
        if agent_names:
            for name in agent_names:
                self._agent_budgets[name] = self.config.default_budget_per_step
        else:
            for name in self._agent_budgets:
                self._agent_budgets[name] = self.config.default_budget_per_step

    def set_agent_location(self, agent_name: str, location: str) -> None:
        """
        Set an agent's current location.

        Args:
            agent_name: Name of the agent
            location: Node ID of the location
        """
        self._agent_locations[agent_name] = location
        if agent_name not in self._agent_budgets:
            self._agent_budgets[agent_name] = self.config.default_budget_per_step

    def get_agent_location(self, agent_name: str) -> Optional[str]:
        """Get an agent's current location."""
        return self._agent_locations.get(agent_name)

    def get_agent_budget(self, agent_name: str) -> float:
        """Get an agent's remaining movement budget."""
        return self._agent_budgets.get(agent_name, self.config.default_budget_per_step)

    def update_locations_from_state(self, agent_vars: Dict[str, Dict[str, Any]]) -> None:
        """
        Update agent locations from game state.

        Handles both direct string values and dict format {"value": "location"}.

        Args:
            agent_vars: Dictionary of agent variables containing 'location' keys
        """
        for agent_name, vars_dict in agent_vars.items():
            if "location" in vars_dict:
                location = vars_dict["location"]
                # Handle dict format {"value": "location_id"}
                if isinstance(location, dict) and "value" in location:
                    location = location["value"]
                self._agent_locations[agent_name] = location
                if agent_name not in self._agent_budgets:
                    self._agent_budgets[agent_name] = self.config.default_budget_per_step

    def validate_movement(
        self,
        agent_name: str,
        from_location: str,
        to_location: str
    ) -> MovementResult:
        """
        Validate a proposed movement without executing it.

        Args:
            agent_name: Name of the agent attempting to move
            from_location: Starting node ID
            to_location: Destination node ID

        Returns:
            MovementResult indicating if the movement is valid
        """
        # Same location - no movement needed
        if from_location == to_location:
            return MovementResult(
                valid=True,
                from_location=from_location,
                to_location=to_location,
                path=[from_location],
                cost=0,
                remaining_budget=self.get_agent_budget(agent_name)
            )

        # Check if locations exist
        if from_location not in self.graph:
            return MovementResult(
                valid=False,
                from_location=from_location,
                to_location=to_location,
                path=[],
                cost=0,
                error=f"Invalid source location: {from_location}"
            )

        if to_location not in self.graph:
            return MovementResult(
                valid=False,
                from_location=from_location,
                to_location=to_location,
                path=[],
                cost=0,
                error=f"Invalid destination location: {to_location}"
            )

        # Check if direct connection exists (if multi-hop disabled)
        if not self.config.allow_multi_hop:
            edge = self.graph.get_edge(from_location, to_location)
            if not edge:
                return MovementResult(
                    valid=False,
                    from_location=from_location,
                    to_location=to_location,
                    path=[],
                    cost=0,
                    error=f"No direct connection from {from_location} to {to_location}"
                )
            cost = edge.traversal_cost
            path = [from_location, to_location]
        else:
            # Find shortest path
            path_result = self.graph.shortest_path(from_location, to_location)
            if not path_result.exists:
                return MovementResult(
                    valid=False,
                    from_location=from_location,
                    to_location=to_location,
                    path=[],
                    cost=0,
                    error=f"No path exists from {from_location} to {to_location}"
                )
            cost = path_result.total_cost
            path = path_result.path

        # Check budget
        budget = self.get_agent_budget(agent_name)
        if cost > budget:
            return MovementResult(
                valid=False,
                from_location=from_location,
                to_location=to_location,
                path=path,
                cost=cost,
                error=f"Movement cost {cost:.0f} exceeds budget {budget:.0f}"
            )

        return MovementResult(
            valid=True,
            from_location=from_location,
            to_location=to_location,
            path=path,
            cost=cost,
            remaining_budget=budget - cost
        )

    def execute_movement(
        self,
        agent_name: str,
        from_location: str,
        to_location: str
    ) -> MovementResult:
        """
        Validate and execute a movement, deducting from budget.

        Args:
            agent_name: Name of the agent attempting to move
            from_location: Starting node ID
            to_location: Destination node ID

        Returns:
            MovementResult with execution status
        """
        result = self.validate_movement(agent_name, from_location, to_location)

        if result.valid and result.cost > 0:
            # Deduct cost from budget
            self._agent_budgets[agent_name] = self.get_agent_budget(agent_name) - result.cost
            # Update location
            self._agent_locations[agent_name] = to_location

        return result

    def get_valid_destinations(
        self,
        agent_name: str,
        from_location: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get all valid destinations for an agent within their budget.

        Args:
            agent_name: Name of the agent
            from_location: Starting location (defaults to agent's current location)

        Returns:
            Dictionary mapping destination node IDs to their movement cost
        """
        location = from_location or self._agent_locations.get(agent_name)
        if not location:
            return {}

        budget = self.get_agent_budget(agent_name)
        return self.graph.nodes_within_cost(location, budget)

    def validate_location_updates(
        self,
        state_updates: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate location updates in state changes and reject invalid ones.

        Handles both direct string values and dict format {"value": "location"}.

        Args:
            state_updates: State updates from LLM including agent_vars

        Returns:
            Tuple of (corrected_updates, list of warning messages)
        """
        warnings = []
        corrected = {
            "global_vars": state_updates.get("global_vars", {}),
            "agent_vars": {}
        }

        agent_vars = state_updates.get("agent_vars", {})

        for agent_name, vars_dict in agent_vars.items():
            corrected["agent_vars"][agent_name] = vars_dict.copy()

            if "location" not in vars_dict:
                continue

            new_location = vars_dict["location"]
            # Handle dict format {"value": "location_id"}
            if isinstance(new_location, dict) and "value" in new_location:
                new_location = new_location["value"]

            current_location = self._agent_locations.get(agent_name)

            # If no current location tracked, accept any valid node
            if not current_location:
                if new_location in self.graph:
                    self._agent_locations[agent_name] = new_location
                else:
                    warnings.append(
                        f"{agent_name}: Invalid location '{new_location}' - "
                        f"not a valid node in spatial graph"
                    )
                    del corrected["agent_vars"][agent_name]["location"]
                continue

            # Validate movement
            result = self.validate_movement(agent_name, current_location, new_location)

            if not result.valid:
                warnings.append(
                    f"{agent_name}: Invalid movement from {current_location} to {new_location} - "
                    f"{result.error}. Location unchanged."
                )
                # Remove the invalid location update
                del corrected["agent_vars"][agent_name]["location"]
            else:
                # Valid movement - deduct budget and update tracking
                if result.cost > 0:
                    self._agent_budgets[agent_name] = result.remaining_budget
                self._agent_locations[agent_name] = new_location

        return corrected, warnings

    def is_valid_location(self, location: str) -> bool:
        """Check if a location ID is valid in the graph."""
        return location in self.graph

    def get_location_name(self, location_id: str) -> str:
        """Get the display name for a location."""
        node = self.graph.get_node(location_id)
        return node.name if node else location_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize validator state."""
        return {
            "agent_budgets": self._agent_budgets.copy(),
            "agent_locations": self._agent_locations.copy(),
            "config": {
                "default_budget_per_step": self.config.default_budget_per_step,
                "allow_multi_hop": self.config.allow_multi_hop,
                "blocked_edge_types": self.config.blocked_edge_types
            }
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore validator state from dictionary."""
        self._agent_budgets = data.get("agent_budgets", {})
        self._agent_locations = data.get("agent_locations", {})
