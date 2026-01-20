"""Unit tests for spatial graph components."""

import pytest
from utils.spatial_graph import SpatialGraph, Node, Edge, PathResult
from utils.spatial_queries import SpatialQueryEngine
from utils.movement_validator import MovementValidator, MovementConfig, MovementResult


class TestNode:
    """Tests for Node dataclass."""

    def test_node_creation(self):
        node = Node(
            id="taiwan",
            name="Taiwan",
            type="nation",
            properties={"population": 24000000},
            conditions=["democratic", "island"]
        )
        assert node.id == "taiwan"
        assert node.name == "Taiwan"
        assert node.type == "nation"
        assert node.properties["population"] == 24000000
        assert len(node.conditions) == 2

    def test_node_defaults(self):
        node = Node(id="test", name="Test", type="location")
        assert node.properties == {}
        assert node.conditions == []


class TestEdge:
    """Tests for Edge dataclass."""

    def test_edge_creation(self):
        edge = Edge(
            from_node="taiwan",
            to_node="fujian",
            type="maritime",
            directed=False,
            properties={
                "distance_km": 180,
                "travel_time_hours": 4,
                "traversal_cost": 10
            }
        )
        assert edge.from_node == "taiwan"
        assert edge.to_node == "fujian"
        assert edge.type == "maritime"
        assert not edge.directed
        assert edge.distance_km == 180
        assert edge.travel_time_hours == 4
        assert edge.traversal_cost == 10

    def test_edge_property_defaults(self):
        edge = Edge(from_node="a", to_node="b", type="road")
        assert edge.distance_km == 0
        assert edge.travel_time_hours == 0
        assert edge.traversal_cost == 1  # default from properties.get


class TestSpatialGraph:
    """Tests for SpatialGraph class."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        graph = SpatialGraph()

        # Add nodes
        graph.add_node(Node(id="a", name="A", type="city"))
        graph.add_node(Node(id="b", name="B", type="city"))
        graph.add_node(Node(id="c", name="C", type="city"))
        graph.add_node(Node(id="d", name="D", type="city"))

        # Add edges
        # A <-> B (cost 5)
        graph.add_edge(Edge(
            from_node="a", to_node="b", type="road",
            properties={"traversal_cost": 5, "distance_km": 100}
        ))
        # B <-> C (cost 3)
        graph.add_edge(Edge(
            from_node="b", to_node="c", type="road",
            properties={"traversal_cost": 3, "distance_km": 60}
        ))
        # A <-> C (cost 10) - longer direct route
        graph.add_edge(Edge(
            from_node="a", to_node="c", type="highway",
            properties={"traversal_cost": 10, "distance_km": 150}
        ))
        # D is isolated
        return graph

    @pytest.fixture
    def taiwan_graph(self):
        """Create a Taiwan Strait test graph."""
        graph = SpatialGraph()

        # Add nodes
        graph.add_node(Node(id="taiwan", name="Taiwan", type="nation"))
        graph.add_node(Node(id="fujian", name="Fujian Province", type="region"))
        graph.add_node(Node(id="guangdong", name="Guangdong Province", type="region"))
        graph.add_node(Node(id="japan", name="Japan", type="nation"))
        graph.add_node(Node(id="philippines", name="Philippines", type="nation"))

        # Add edges
        graph.add_edge(Edge(
            from_node="taiwan", to_node="fujian", type="maritime",
            properties={"distance_km": 180, "travel_time_hours": 4, "traversal_cost": 10}
        ))
        graph.add_edge(Edge(
            from_node="taiwan", to_node="japan", type="maritime",
            properties={"distance_km": 200, "travel_time_hours": 5, "traversal_cost": 12}
        ))
        graph.add_edge(Edge(
            from_node="taiwan", to_node="philippines", type="maritime",
            properties={"distance_km": 350, "travel_time_hours": 8, "traversal_cost": 15}
        ))
        graph.add_edge(Edge(
            from_node="fujian", to_node="guangdong", type="land",
            properties={"distance_km": 500, "travel_time_hours": 6, "traversal_cost": 8}
        ))

        return graph

    def test_add_node(self, simple_graph):
        assert len(simple_graph) == 4
        assert "a" in simple_graph
        assert "e" not in simple_graph

    def test_get_node(self, simple_graph):
        node = simple_graph.get_node("a")
        assert node is not None
        assert node.name == "A"

        missing = simple_graph.get_node("missing")
        assert missing is None

    def test_get_neighbors(self, simple_graph):
        neighbors = simple_graph.get_neighbors("a")
        neighbor_ids = [n_id for n_id, _ in neighbors]

        assert "b" in neighbor_ids
        assert "c" in neighbor_ids
        assert "d" not in neighbor_ids

    def test_get_edge(self, simple_graph):
        edge = simple_graph.get_edge("a", "b")
        assert edge is not None
        assert edge.type == "road"

        no_edge = simple_graph.get_edge("a", "d")
        assert no_edge is None

    def test_shortest_path(self, simple_graph):
        # Direct path A -> C via B is cheaper (5+3=8) than direct (10)
        result = simple_graph.shortest_path("a", "c")
        assert result.exists
        assert result.path == ["a", "b", "c"]
        assert result.total_cost == 8

    def test_shortest_path_same_node(self, simple_graph):
        result = simple_graph.shortest_path("a", "a")
        assert result.exists
        assert result.path == ["a"]
        assert result.total_cost == 0

    def test_shortest_path_no_path(self, simple_graph):
        result = simple_graph.shortest_path("a", "d")
        assert not result.exists
        assert result.path == []

    def test_nodes_within_hops(self, simple_graph):
        # From A, within 1 hop: B, C
        reachable = simple_graph.nodes_within_hops("a", 1)
        assert "a" in reachable
        assert "b" in reachable
        assert "c" in reachable
        assert "d" not in reachable

    def test_nodes_within_cost(self, simple_graph):
        # From A with budget 5: B (cost 5)
        reachable = simple_graph.nodes_within_cost("a", 5)
        assert "a" in reachable
        assert "b" in reachable
        assert "c" not in reachable  # Would need 8 via B

    def test_is_reachable(self, simple_graph):
        assert simple_graph.is_reachable("a", "c")
        assert not simple_graph.is_reachable("a", "d")

    def test_block_edge_type(self, taiwan_graph):
        # Initially can reach fujian from taiwan
        assert taiwan_graph.is_reachable("taiwan", "fujian")

        # Block maritime edges
        taiwan_graph.block_edge_type("maritime")
        assert taiwan_graph.is_edge_type_blocked("maritime")

        # Now cannot reach
        assert not taiwan_graph.is_reachable("taiwan", "fujian")

        # Unblock
        taiwan_graph.unblock_edge_type("maritime")
        assert not taiwan_graph.is_edge_type_blocked("maritime")
        assert taiwan_graph.is_reachable("taiwan", "fujian")

    def test_serialization(self, taiwan_graph):
        # Serialize
        data = taiwan_graph.to_dict()

        # Deserialize
        restored = SpatialGraph.from_dict(data)

        # Verify
        assert len(restored) == len(taiwan_graph)
        assert "taiwan" in restored
        assert restored.is_reachable("taiwan", "fujian")

    def test_directed_edge(self):
        graph = SpatialGraph()
        graph.add_node(Node(id="a", name="A", type="city"))
        graph.add_node(Node(id="b", name="B", type="city"))

        # Add directed edge A -> B
        graph.add_edge(Edge(
            from_node="a", to_node="b", type="one_way",
            directed=True,
            properties={"traversal_cost": 5}
        ))

        # Can go A -> B
        assert graph.is_reachable("a", "b")

        # Cannot go B -> A
        assert not graph.is_reachable("b", "a")


class TestSpatialQueryEngine:
    """Tests for SpatialQueryEngine class."""

    @pytest.fixture
    def engine(self):
        graph = SpatialGraph()
        graph.add_node(Node(id="a", name="City A", type="city"))
        graph.add_node(Node(id="b", name="City B", type="city"))
        graph.add_node(Node(id="c", name="City C", type="city"))

        graph.add_edge(Edge(
            from_node="a", to_node="b", type="highway",
            properties={"distance_km": 100, "travel_time_hours": 1, "traversal_cost": 5}
        ))
        graph.add_edge(Edge(
            from_node="b", to_node="c", type="road",
            properties={"distance_km": 50, "travel_time_hours": 0.5, "traversal_cost": 3}
        ))

        return SpatialQueryEngine(graph)

    def test_query_neighbors(self, engine):
        result = engine.query_neighbors("a")
        assert "City A" in result
        assert "City B" in result
        assert "highway" in result

    def test_query_path(self, engine):
        result = engine.query_path("a", "c")
        assert "City A" in result
        assert "City C" in result
        assert "150 km" in result or "150.0 km" in result or "150" in result

    def test_query_reachable(self, engine):
        result = engine.query_reachable("a", "c")
        assert "Yes" in result or "reachable" in result

    def test_query_within_range(self, engine):
        result = engine.query_within_range("a", 2)
        assert "City B" in result
        assert "City C" in result

    def test_get_spatial_summary(self, engine):
        agent_locations = {"Agent 1": "a", "Agent 2": "b"}
        result = engine.get_spatial_summary(agent_locations)
        assert "City A" in result
        assert "Agent 1" in result
        assert "SPATIAL RULES" in result


class TestMovementValidator:
    """Tests for MovementValidator class."""

    @pytest.fixture
    def validator(self):
        graph = SpatialGraph()
        graph.add_node(Node(id="base", name="Base", type="location"))
        graph.add_node(Node(id="outpost", name="Outpost", type="location"))
        graph.add_node(Node(id="remote", name="Remote", type="location"))

        # Base <-> Outpost (cost 5)
        graph.add_edge(Edge(
            from_node="base", to_node="outpost", type="road",
            properties={"traversal_cost": 5}
        ))
        # Outpost <-> Remote (cost 8)
        graph.add_edge(Edge(
            from_node="outpost", to_node="remote", type="road",
            properties={"traversal_cost": 8}
        ))

        config = MovementConfig(default_budget_per_step=10)
        return MovementValidator(graph, config)

    def test_validate_valid_movement(self, validator):
        validator.set_agent_location("agent1", "base")
        result = validator.validate_movement("agent1", "base", "outpost")

        assert result.valid
        assert result.cost == 5
        assert result.remaining_budget == 5

    def test_validate_over_budget(self, validator):
        validator.set_agent_location("agent1", "base")
        # Direct to remote would cost 5+8=13, but budget is 10
        result = validator.validate_movement("agent1", "base", "remote")

        assert not result.valid
        assert "exceeds budget" in result.error

    def test_validate_no_path(self, validator):
        validator.set_agent_location("agent1", "base")
        result = validator.validate_movement("agent1", "base", "nonexistent")

        assert not result.valid
        assert "Invalid destination" in result.error

    def test_execute_movement(self, validator):
        validator.set_agent_location("agent1", "base")
        result = validator.execute_movement("agent1", "base", "outpost")

        assert result.valid
        assert validator.get_agent_location("agent1") == "outpost"
        assert validator.get_agent_budget("agent1") == 5

    def test_reset_budgets(self, validator):
        validator.set_agent_location("agent1", "base")
        validator.execute_movement("agent1", "base", "outpost")
        assert validator.get_agent_budget("agent1") == 5

        validator.reset_budgets(["agent1"])
        assert validator.get_agent_budget("agent1") == 10

    def test_validate_location_updates(self, validator):
        validator.set_agent_location("agent1", "base")
        validator.reset_budgets(["agent1"])

        state_updates = {
            "global_vars": {},
            "agent_vars": {
                "agent1": {"location": "outpost", "other_var": 42}
            }
        }

        corrected, warnings = validator.validate_location_updates(state_updates)

        assert len(warnings) == 0
        assert corrected["agent_vars"]["agent1"]["location"] == "outpost"
        assert validator.get_agent_location("agent1") == "outpost"

    def test_validate_invalid_location_update(self, validator):
        validator.set_agent_location("agent1", "base")
        validator.reset_budgets(["agent1"])

        state_updates = {
            "global_vars": {},
            "agent_vars": {
                "agent1": {"location": "nonexistent"}
            }
        }

        corrected, warnings = validator.validate_location_updates(state_updates)

        assert len(warnings) == 1
        assert "location" not in corrected["agent_vars"]["agent1"]

    def test_blocked_edge_type(self, validator):
        validator.graph.block_edge_type("road")
        validator.set_agent_location("agent1", "base")

        result = validator.validate_movement("agent1", "base", "outpost")

        assert not result.valid
        assert "No path exists" in result.error


class TestMovementConfig:
    """Tests for MovementConfig dataclass."""

    def test_default_values(self):
        config = MovementConfig()
        assert config.default_budget_per_step == 20.0
        assert config.allow_multi_hop is True
        assert config.blocked_edge_types == []

    def test_custom_values(self):
        config = MovementConfig(
            default_budget_per_step=50.0,
            allow_multi_hop=False,
            blocked_edge_types=["maritime"]
        )
        assert config.default_budget_per_step == 50.0
        assert config.allow_multi_hop is False
        assert config.blocked_edge_types == ["maritime"]
