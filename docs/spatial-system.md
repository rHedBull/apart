# Spatial System

The spatial system provides graph-based geographic modeling for simulations.

## Overview

Locations are represented as **nodes** (cities, regions, nations) connected by **edges** (routes, borders). This enables:

- Pathfinding between locations
- Movement validation with budgets
- Dynamic blockades (e.g., naval blockades)
- Distance and travel time calculations

## Configuration

### In Scenario YAML

```yaml
geography:
  spatial_model: graph

  nodes:
    - id: taiwan
      name: "Taiwan"
      type: region
      coordinates: [121.5, 23.5]  # [lng, lat]
      properties:
        population: 24000000

    - id: beijing
      name: "Beijing"
      type: city
      coordinates: [116.4, 39.9]

    - id: taiwan_strait
      name: "Taiwan Strait"
      type: sea_zone

  edges:
    - from: taiwan
      to: taiwan_strait
      type: maritime
      properties:
        distance_km: 100
        travel_time_hours: 4
        traversal_cost: 2.0

    - from: beijing
      to: taiwan_strait
      type: air
      directed: true  # One-way route
      properties:
        distance_km: 1200

  movement:
    default_budget_per_step: 20.0
    allow_multi_hop: true
```

### Node Types

| Type | Description | Example |
|------|-------------|---------|
| `nation` | Country or sovereign state | USA, China |
| `region` | Geographic region | Taiwan, Crimea |
| `city` | Major city | Taipei, Beijing |
| `sea_zone` | Maritime area | Taiwan Strait, Pacific |
| `air_base` | Military air installation | Kadena Air Base |
| `port` | Naval port | Kaohsiung Port |

### Edge Types

| Type | Description | Can Block |
|------|-------------|-----------|
| `maritime` | Sea routes | Yes (blockades) |
| `land` | Ground routes | Yes |
| `air` | Air routes | Yes (no-fly zones) |
| `rail` | Railway connections | Yes |
| `diplomatic` | Political relationships | No |

## Programmatic Usage

### Creating a Graph

```python
from utils.spatial_graph import SpatialGraph, Node, Edge

graph = SpatialGraph()

# Add nodes
graph.add_node(Node(
    id="taipei",
    name="Taipei",
    type="city",
    coordinates=(121.5, 25.0),
    properties={"population": 2700000}
))

graph.add_node(Node(
    id="beijing",
    name="Beijing",
    type="city",
    coordinates=(116.4, 39.9)
))

# Add edge
graph.add_edge(Edge(
    from_node="taipei",
    to_node="beijing",
    type="air",
    directed=False,
    properties={
        "distance_km": 1700,
        "travel_time_hours": 2.5,
        "traversal_cost": 5.0
    }
))
```

### Pathfinding

```python
# Find shortest path by cost
result = graph.find_path("taipei", "beijing")

if result.exists:
    print(f"Path: {' -> '.join(result.path)}")
    print(f"Total cost: {result.total_cost}")
    print(f"Distance: {result.total_distance} km")
    print(f"Time: {result.total_time} hours")
```

### Blocking Routes

```python
# Block all maritime routes (simulating blockade)
graph.block_edge_type("maritime")

# Check if path still exists
result = graph.find_path("taiwan", "usa")
if not result.exists:
    print("Route blocked by blockade")

# Unblock
graph.unblock_edge_type("maritime")
```

### Querying Neighbors

```python
# Get all neighbors within reach
neighbors = graph.get_neighbors("taiwan")
for node_id, edge in neighbors:
    print(f"{node_id} via {edge.type} ({edge.distance_km} km)")

# Get nodes within N hops
nearby = graph.get_nodes_within_hops("taiwan", max_hops=2)
```

## Movement Validation

The movement validator enforces budget constraints:

```python
from utils.movement_validator import MovementValidator

validator = MovementValidator(graph, budget_per_step=20.0)

# Validate a move
result = validator.validate_move(
    agent="China",
    from_node="beijing",
    to_node="taiwan_strait",
    step=1
)

if result.valid:
    print(f"Move allowed, remaining budget: {result.remaining_budget}")
else:
    print(f"Move blocked: {result.reason}")
```

## Spatial Queries

Advanced queries for simulation logic:

```python
from utils.spatial_queries import SpatialQueries

queries = SpatialQueries(graph)

# Find all sea zones
sea_zones = queries.find_nodes_by_type("sea_zone")

# Find nodes within distance
nearby = queries.find_nodes_within_distance("taiwan", max_km=500)

# Check line of sight (no blocking terrain)
has_los = queries.has_line_of_sight("taipei", "xiamen")

# Find chokepoints
chokepoints = queries.find_chokepoints()
```

## Loading from Files

### Map Files

```yaml
# modules/maps/east_asia.yaml
nodes:
  - id: taiwan
    name: Taiwan
    type: region
    coordinates: [121.5, 23.5]

edges:
  - from: taiwan
    to: taiwan_strait
    type: maritime
```

```python
from modules.map_loader import load_map

graph = load_map("modules/maps/east_asia.yaml")
```

### GeoJSON Integration

For dashboard visualization:

```python
from modules.map_loader import graph_to_geojson

geojson = graph_to_geojson(graph)
# Returns FeatureCollection with polygons/points for each node
```

## Integration with Modules

The `territory_graph` module integrates spatial data with simulations:

```yaml
modules:
  - territory_graph

module_config:
  territory_graph:
    map_file: modules/maps/east_asia.yaml
```

This automatically:
- Loads the spatial graph
- Adds location variables to agents
- Injects movement constraints into the simulation
- Provides spatial context to the engine

## Data Structures

### Node

```python
@dataclass
class Node:
    id: str                           # Unique identifier
    name: str                         # Display name
    type: str                         # Node type
    properties: Dict[str, Any]        # Custom properties
    conditions: List[str]             # Active conditions
    coordinates: Optional[Tuple[float, float]]  # (lng, lat)
```

### Edge

```python
@dataclass
class Edge:
    from_node: str                    # Source node ID
    to_node: str                      # Target node ID
    type: str                         # Edge type
    directed: bool                    # One-way if True
    properties: Dict[str, Any]        # distance_km, travel_time_hours, traversal_cost
```

### PathResult

```python
@dataclass
class PathResult:
    exists: bool                      # Path found?
    path: List[str]                   # Node IDs in order
    total_cost: float                 # Sum of traversal costs
    total_distance: float             # Sum of distances (km)
    total_time: float                 # Sum of travel times (hours)
    edges_used: List[Edge]            # Edges traversed
```

## Algorithm Details

### Pathfinding

Uses Dijkstra's algorithm with traversal cost as weight:

```python
def find_path(self, from_node: str, to_node: str) -> PathResult:
    # Priority queue: (cost, node_id, path)
    # Expands lowest-cost nodes first
    # Respects blocked edge types
```

### Reachability

BFS-based reachability within hop count:

```python
def get_nodes_within_hops(self, start: str, max_hops: int) -> Set[str]:
    # Returns all node IDs reachable within max_hops edges
```
