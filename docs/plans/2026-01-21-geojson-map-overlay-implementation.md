# GeoJSON Map Overlay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add geographic map visualization using Leaflet and GeoJSON while keeping the graph as the underlying data structure.

**Architecture:** Extend the existing Node/SpatialGraph data models to support coordinates, load GeoJSON files alongside map YAML files, pass GeoJSON through the API, and render with Leaflet in a new GeoMapVisualization component that conditionally replaces the existing force-directed MapVisualization.

**Tech Stack:** Python (dataclasses, json), TypeScript, React, Leaflet/react-leaflet, GeoJSON

---

## Task 1: Extend Node Dataclass with Coordinates

**Files:**
- Modify: `src/utils/spatial_graph.py:14-22`

**Step 1: Add coordinates field to Node**

In `src/utils/spatial_graph.py`, modify the Node dataclass:

```python
@dataclass
class Node:
    """Represents a location in the spatial graph."""
    id: str
    name: str
    type: str  # nation, city, region, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    coordinates: Optional[Tuple[float, float]] = None  # (lng, lat) for point nodes
```

**Step 2: Update to_dict to include coordinates**

In the `SpatialGraph.to_dict` method (around line 330-356), update the nodes serialization:

```python
"nodes": [
    {
        "id": n.id,
        "name": n.name,
        "type": n.type,
        "properties": n.properties,
        "conditions": n.conditions,
        "coordinates": n.coordinates,  # Add this line
    }
    for n in self._nodes.values()
],
```

**Step 3: Update from_dict to parse coordinates**

In `SpatialGraph.from_dict` (around line 358-389), update node creation:

```python
# Parse coordinates if present
coords = node_data.get("coordinates")
coordinates = tuple(coords) if coords else None

node = Node(
    id=node_data["id"],
    name=node_data["name"],
    type=node_data["type"],
    properties=node_data.get("properties", {}),
    conditions=node_data.get("conditions", []),
    coordinates=coordinates,
)
```

**Step 4: Commit**

```bash
git add src/utils/spatial_graph.py
git commit -m "feat(spatial): add coordinates field to Node dataclass"
```

---

## Task 2: Extend Map Loader to Handle GeoJSON

**Files:**
- Modify: `src/modules/map_loader.py`

**Step 1: Update load_map_file signature and add GeoJSON loading**

Replace the entire `load_map_file` function:

```python
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
```

**Step 2: Update parse_map_data to handle coordinates**

In `parse_map_data`, update the node creation (around line 85-92):

```python
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
```

**Step 3: Commit**

```bash
git add src/modules/map_loader.py
git commit -m "feat(map-loader): add GeoJSON loading and coordinate parsing"
```

---

## Task 3: Update ComposedModules to Store GeoJSON

**Files:**
- Modify: `src/modules/models.py:307-331`
- Modify: `src/modules/composer.py:52-69`

**Step 1: Add geojson field to ComposedModules**

In `src/modules/models.py`, add to the ComposedModules dataclass (around line 327-330):

```python
# Spatial graph (loaded from territory module's map_file)
spatial_graph: Optional[Any] = None  # SpatialGraph, avoid circular import
map_metadata: Dict[str, Any] = field(default_factory=dict)
movement_config: Optional[Any] = None  # MovementConfig
geojson: Optional[Dict[str, Any]] = None  # GeoJSON data for map overlay
```

**Step 2: Update _load_map_for_module in composer.py**

Replace the `_load_map_for_module` method:

```python
def _load_map_for_module(
    self,
    module: BehaviorModule,
    composed: ComposedModules
) -> None:
    """Load map file and populate spatial graph in composed modules."""
    from modules.map_loader import load_map_file, create_movement_config

    map_file = module.config_values.get("map_file")
    if not map_file:
        return

    spatial_graph, metadata, geojson = load_map_file(map_file)
    movement_config = create_movement_config(module.config_values)

    composed.spatial_graph = spatial_graph
    composed.map_metadata = metadata
    composed.movement_config = movement_config
    composed.geojson = geojson
```

**Step 3: Commit**

```bash
git add src/modules/models.py src/modules/composer.py
git commit -m "feat(composer): pass GeoJSON through ComposedModules"
```

---

## Task 4: Update API to Include GeoJSON in Response

**Files:**
- Find and modify: API endpoint that returns run data with spatialGraph

**Step 1: Find the API endpoint**

```bash
grep -r "spatialGraph\|spatial_graph" src/api/ src/server/ --include="*.py"
```

**Step 2: Update the endpoint to include geojson**

Add `geojson` field alongside `spatialGraph` in the response. The exact code depends on the API structure found in step 1.

**Step 3: Commit**

```bash
git add src/api/  # or wherever the changes are
git commit -m "feat(api): include geojson in run data response"
```

---

## Task 5: Update Frontend Types and Store

**Files:**
- Modify: `dashboard/src/hooks/useSimulationState.ts`

**Step 1: Add coordinates to SpatialNode interface**

Update the SpatialNode interface (around line 32-38):

```typescript
interface SpatialNode {
  id: string;
  name: string;
  type: string;
  properties: Record<string, unknown>;
  conditions: string[];
  coordinates?: [number, number];  // [lng, lat] for point nodes
}
```

**Step 2: Add GeoJSON types**

Add after the SpatialGraph interface (around line 52):

```typescript
// GeoJSON types for map overlay
export interface GeoJSONGeometry {
  type: 'Polygon' | 'MultiPolygon';
  coordinates: number[][][] | number[][][][];
}

export interface GeoJSONFeature {
  type: 'Feature';
  id?: string | number;
  properties: Record<string, unknown>;
  geometry: GeoJSONGeometry;
}

export interface GeoJSONData {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
}
```

**Step 3: Add geojson to SimulationState**

In the SimulationState interface (around line 54-111), add:

```typescript
// GeoJSON data (for geographic map visualization)
geojson: GeoJSONData | null;
```

**Step 4: Update initial state**

In the create function (around line 113-126), add:

```typescript
geojson: null,
```

**Step 5: Update loadRunData action**

Update the loadRunData function parameter type (around line 96-107):

```typescript
loadRunData: (data: {
  runId: string;
  status: string;
  currentStep: number;
  maxSteps: number | null;
  agentNames: string[];
  spatialGraph: SpatialGraph | null;
  geojson: GeoJSONData | null;  // Add this
  messages: AgentMessage[];
  dangerSignals: DangerSignal[];
  globalVarsHistory: VariableHistory[];
  agentVarsHistory: Record<string, VariableHistory[]>;
}) => void;
```

And update the implementation (around line 256-268):

```typescript
loadRunData: (data) =>
  set({
    currentRunId: data.runId,
    status: (data.status as 'idle' | 'running' | 'completed' | 'failed') || 'completed',
    currentStep: data.currentStep,
    maxSteps: data.maxSteps,
    agentNames: data.agentNames,
    spatialGraph: data.spatialGraph,
    geojson: data.geojson,  // Add this
    messages: data.messages,
    dangerSignals: data.dangerSignals,
    globalVarsHistory: data.globalVarsHistory,
    agentVarsHistory: data.agentVarsHistory,
  }),
```

**Step 6: Update processEvent for simulation_started**

In the processEvent switch case for 'simulation_started' (around line 186-198), add geojson:

```typescript
case 'simulation_started':
  set({
    status: 'running',
    currentStep: 0,
    maxSteps: data.max_steps as number,
    agentNames: (data.agent_names as string[]) || [],
    spatialGraph: (data.spatial_graph as SpatialGraph) || null,
    geojson: (data.geojson as GeoJSONData) || null,  // Add this
    messages: [],
    dangerSignals: [],
    globalVarsHistory: [],
    agentVarsHistory: {},
  });
  break;
```

**Step 7: Update reset**

In the reset action (around line 270-284), add:

```typescript
geojson: null,
```

**Step 8: Commit**

```bash
git add dashboard/src/hooks/useSimulationState.ts
git commit -m "feat(dashboard): add GeoJSON types and state to store"
```

---

## Task 6: Install Leaflet Dependencies

**Files:**
- Modify: `dashboard/package.json`

**Step 1: Install dependencies**

```bash
cd dashboard && npm install leaflet react-leaflet && npm install -D @types/leaflet
```

**Step 2: Commit**

```bash
git add dashboard/package.json dashboard/package-lock.json
git commit -m "deps(dashboard): add leaflet and react-leaflet"
```

---

## Task 7: Create GeoMapVisualization Component

**Files:**
- Create: `dashboard/src/components/GeoMapVisualization.tsx`

**Step 1: Create the component file**

```typescript
/**
 * Geographic Map Visualization Component
 *
 * Renders a spatial graph on a real geographic map using Leaflet and GeoJSON.
 * Region nodes are rendered as GeoJSON polygons, point nodes as markers.
 * Used when the scenario includes GeoJSON data.
 */

import { useMemo, useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, Tooltip } from 'react-leaflet';
import { LatLngBoundsExpression, LatLngExpression } from 'leaflet';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import Badge from '@cloudscape-design/components/badge';
import SpaceBetween from '@cloudscape-design/components/space-between';
import { useSimulationStore, SpatialGraph, GeoJSONData, GeoJSONFeature } from '../hooks/useSimulationState';
import 'leaflet/dist/leaflet.css';

// Faction colors for region control
const factionColors: Record<string, string> = {
  contested: '#fbbf24',  // amber
  neutral: '#9ca3af',    // gray
};

// Generate faction colors dynamically based on agent names
function getFactionColor(agentName: string, index: number): string {
  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#a855f7', '#f97316', '#06b6d4'];
  return colors[index % colors.length];
}

// Compute centroid of a GeoJSON polygon
function computeCentroid(geometry: GeoJSONFeature['geometry']): [number, number] {
  let coords: number[][];

  if (geometry.type === 'Polygon') {
    coords = geometry.coordinates[0] as number[][];
  } else if (geometry.type === 'MultiPolygon') {
    // Use first polygon of multipolygon
    coords = (geometry.coordinates[0] as number[][][])[0];
  } else {
    return [0, 0];
  }

  let sumLng = 0;
  let sumLat = 0;
  for (const [lng, lat] of coords) {
    sumLng += lng;
    sumLat += lat;
  }

  return [sumLng / coords.length, sumLat / coords.length];
}

// Compute bounds of all GeoJSON features
function computeBounds(geojson: GeoJSONData): LatLngBoundsExpression {
  let minLat = 90, maxLat = -90, minLng = 180, maxLng = -180;

  for (const feature of geojson.features) {
    const processCoords = (coords: number[][]) => {
      for (const [lng, lat] of coords) {
        minLat = Math.min(minLat, lat);
        maxLat = Math.max(maxLat, lat);
        minLng = Math.min(minLng, lng);
        maxLng = Math.max(maxLng, lng);
      }
    };

    if (feature.geometry.type === 'Polygon') {
      processCoords(feature.geometry.coordinates[0] as number[][]);
    } else if (feature.geometry.type === 'MultiPolygon') {
      for (const polygon of feature.geometry.coordinates as number[][][][]) {
        processCoords(polygon[0]);
      }
    }
  }

  return [[minLat, minLng], [maxLat, maxLng]];
}

interface GeoMapVisualizationProps {
  onNodeClick?: (nodeId: string) => void;
}

export function GeoMapVisualization({ onNodeClick }: GeoMapVisualizationProps) {
  const spatialGraph = useSimulationStore((state) => state.spatialGraph);
  const geojson = useSimulationStore((state) => state.geojson);
  const agentVarsHistory = useSimulationStore((state) => state.agentVarsHistory);
  const agentNames = useSimulationStore((state) => state.agentNames);

  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Build agent color map
  const agentColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    agentNames.forEach((name, i) => {
      map[name] = getFactionColor(name, i);
    });
    return map;
  }, [agentNames]);

  // Get current agent locations
  const agentLocations = useMemo(() => {
    const locations: Record<string, string> = {};
    for (const [agentName, history] of Object.entries(agentVarsHistory)) {
      if (history.length > 0) {
        const latest = history[history.length - 1];
        const location = latest.values.location;
        if (location) {
          if (typeof location === 'string') {
            locations[agentName] = location;
          } else if (typeof location === 'object' && location !== null && 'value' in location) {
            locations[agentName] = (location as { value: string }).value;
          }
        }
      }
    }
    return locations;
  }, [agentVarsHistory]);

  // Get region control from controlled_regions variable
  const regionControl = useMemo(() => {
    const control: Record<string, string> = {};
    for (const [agentName, history] of Object.entries(agentVarsHistory)) {
      if (history.length > 0) {
        const latest = history[history.length - 1];
        const controlled = latest.values.controlled_regions;
        if (Array.isArray(controlled)) {
          for (const regionId of controlled) {
            const id = typeof regionId === 'string' ? regionId : String(regionId);
            if (control[id] && control[id] !== agentName) {
              control[id] = 'contested';
            } else if (!control[id]) {
              control[id] = agentName;
            }
          }
        }
      }
    }
    return control;
  }, [agentVarsHistory]);

  // Build agents at each node
  const agentsAtNode = useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const [agent, location] of Object.entries(agentLocations)) {
      if (!map[location]) map[location] = [];
      map[location].push(agent);
    }
    return map;
  }, [agentLocations]);

  // Build feature ID to centroid map
  const featureCentroids = useMemo(() => {
    if (!geojson) return {};
    const map: Record<string, [number, number]> = {};
    for (const feature of geojson.features) {
      const fid = String(feature.id ?? feature.properties?.id ?? '');
      if (fid) {
        map[fid] = computeCentroid(feature.geometry);
      }
    }
    return map;
  }, [geojson]);

  // Get position for any node (region centroid or point coordinates)
  const getNodePosition = (nodeId: string): [number, number] | null => {
    const node = spatialGraph?.nodes.find(n => n.id === nodeId);
    if (node?.coordinates) {
      // Leaflet uses [lat, lng], our data is [lng, lat]
      return [node.coordinates[1], node.coordinates[0]];
    }
    const centroid = featureCentroids[nodeId];
    if (centroid) {
      return [centroid[1], centroid[0]];  // [lat, lng]
    }
    return null;
  };

  // Get edges connected to hovered node
  const hoveredEdges = useMemo(() => {
    if (!hoveredNode || !spatialGraph) return [];
    return spatialGraph.edges.filter(
      e => e.from === hoveredNode || e.to === hoveredNode
    );
  }, [hoveredNode, spatialGraph]);

  // Compute map bounds
  const bounds = useMemo(() => {
    if (!geojson) return undefined;
    return computeBounds(geojson);
  }, [geojson]);

  if (!spatialGraph || !geojson) {
    return (
      <Container header={<Header variant="h2">Geographic Map</Header>} fitHeight>
        <Box textAlign="center" color="text-status-inactive" padding="xxl">
          <Box variant="h3" margin={{ bottom: 's' }}>No Geographic Data</Box>
          <Box variant="p">
            Run a scenario with <code>geojson</code> configured to see the geographic map
          </Box>
        </Box>
      </Container>
    );
  }

  // Style function for GeoJSON regions
  const getRegionStyle = (feature: GeoJSONFeature) => {
    const fid = String(feature.id ?? feature.properties?.id ?? '');
    const controller = regionControl[fid];

    let fillColor = factionColors.neutral;
    if (controller === 'contested') {
      fillColor = factionColors.contested;
    } else if (controller && agentColorMap[controller]) {
      fillColor = agentColorMap[controller];
    }

    return {
      fillColor,
      fillOpacity: 0.4,
      color: '#374151',
      weight: 1,
    };
  };

  // Handle region events
  const onEachRegion = (feature: GeoJSONFeature, layer: L.Layer) => {
    const fid = String(feature.id ?? feature.properties?.id ?? '');
    const node = spatialGraph.nodes.find(n => n.id === fid);
    const agents = agentsAtNode[fid] || [];

    layer.on({
      mouseover: () => setHoveredNode(fid),
      mouseout: () => setHoveredNode(null),
      click: () => onNodeClick?.(fid),
    });

    layer.bindTooltip(
      `<strong>${node?.name || fid}</strong><br/>` +
      `Type: ${node?.type || 'region'}<br/>` +
      (agents.length > 0 ? `Agents: ${agents.join(', ')}` : ''),
      { sticky: true }
    );
  };

  // Point nodes (non-region nodes with coordinates)
  const pointNodes = spatialGraph.nodes.filter(
    n => n.type !== 'region' && n.coordinates
  );

  return (
    <Container
      header={
        <Header variant="h2" counter={`${spatialGraph.nodes.length} locations`}>
          Geographic Map
        </Header>
      }
      fitHeight
    >
      <div style={{ height: '100%', minHeight: 400 }}>
        <MapContainer
          bounds={bounds}
          style={{ height: '100%', width: '100%', minHeight: 400 }}
          scrollWheelZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* GeoJSON regions */}
          <GeoJSON
            data={geojson}
            style={(feature) => getRegionStyle(feature as GeoJSONFeature)}
            onEachFeature={(feature, layer) => onEachRegion(feature as GeoJSONFeature, layer)}
          />

          {/* Point markers for non-region nodes */}
          {pointNodes.map(node => {
            const pos = getNodePosition(node.id);
            if (!pos) return null;
            const agents = agentsAtNode[node.id] || [];

            return (
              <CircleMarker
                key={node.id}
                center={pos as LatLngExpression}
                radius={agents.length > 0 ? 10 : 6}
                pathOptions={{
                  fillColor: agents.length > 0 ? '#ec7211' : '#0972d3',
                  fillOpacity: 0.9,
                  color: '#fff',
                  weight: 2,
                }}
                eventHandlers={{
                  mouseover: () => setHoveredNode(node.id),
                  mouseout: () => setHoveredNode(null),
                  click: () => onNodeClick?.(node.id),
                }}
              >
                <Tooltip>
                  <strong>{node.name}</strong><br/>
                  Type: {node.type}<br/>
                  {agents.length > 0 && `Agents: ${agents.join(', ')}`}
                </Tooltip>
              </CircleMarker>
            );
          })}

          {/* Edge lines on hover */}
          {hoveredEdges.map((edge, i) => {
            const fromPos = getNodePosition(edge.from);
            const toPos = getNodePosition(edge.to);
            if (!fromPos || !toPos) return null;

            return (
              <Polyline
                key={`edge-${i}`}
                positions={[fromPos as LatLngExpression, toPos as LatLngExpression]}
                pathOptions={{
                  color: '#0972d3',
                  weight: 3,
                  opacity: 0.8,
                  dashArray: edge.type === 'maritime' ? '5,5' : undefined,
                }}
              />
            );
          })}
        </MapContainer>

        {/* Legend */}
        <Box padding="xs">
          <SpaceBetween direction="horizontal" size="s">
            {Object.keys(regionControl).length > 0 && (
              <Badge color="blue">
                {Object.keys(regionControl).filter(k => regionControl[k] !== 'contested').length} controlled regions
              </Badge>
            )}
            <Badge color="grey">
              {Object.keys(agentLocations).length} agents tracked
            </Badge>
          </SpaceBetween>
        </Box>
      </div>
    </Container>
  );
}
```

**Step 2: Commit**

```bash
git add dashboard/src/components/GeoMapVisualization.tsx
git commit -m "feat(dashboard): add GeoMapVisualization component with Leaflet"
```

---

## Task 8: Update RunDetailPage for Conditional Rendering

**Files:**
- Modify: `dashboard/src/pages/RunDetailPage.tsx`

**Step 1: Import GeoMapVisualization**

Add import at top (around line 22):

```typescript
import { GeoMapVisualization } from '../components/GeoMapVisualization';
```

**Step 2: Get geojson from store**

Add to the state hooks (around line 55):

```typescript
const geojson = useSimulationStore((state) => state.geojson);
```

**Step 3: Update the Map tab content**

Replace the map tab content (around line 252-255):

```typescript
{
  id: 'map',
  label: spatialGraph ? `Map (${spatialGraph.nodes.length})` : 'Map',
  content: geojson ? (
    <GeoMapVisualization onNodeClick={handleNodeClick} />
  ) : (
    <MapVisualization onNodeClick={handleNodeClick} />
  ),
  disabled: !spatialGraph,
},
```

**Step 4: Update loadRunData call**

Update the loadRunData call to include geojson (around line 79-90):

```typescript
loadRunData({
  runId: data.runId,
  status: data.status,
  currentStep: data.currentStep,
  maxSteps: data.maxSteps,
  agentNames: data.agentNames || [],
  spatialGraph: data.spatialGraph,
  geojson: data.geojson || null,  // Add this
  messages: data.messages || [],
  dangerSignals: data.dangerSignals || [],
  globalVarsHistory: data.globalVarsHistory || [],
  agentVarsHistory: data.agentVarsHistory || {},
});
```

**Step 5: Commit**

```bash
git add dashboard/src/pages/RunDetailPage.tsx
git commit -m "feat(dashboard): conditionally render GeoMapVisualization"
```

---

## Task 9: Create Example GeoJSON Map

**Files:**
- Create: `src/modules/maps/europe_simple.geojson`
- Modify: `src/modules/maps/sample_conflict.yaml`

**Step 1: Create a simple example GeoJSON**

Create `src/modules/maps/europe_simple.geojson` with a few simplified country polygons (just for testing - real GeoJSON would come from Natural Earth or similar):

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "DEU",
      "properties": { "name": "Germany" },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[6, 47], [15, 47], [15, 55], [6, 55], [6, 47]]]
      }
    },
    {
      "type": "Feature",
      "id": "FRA",
      "properties": { "name": "France" },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-5, 42], [8, 42], [8, 51], [-5, 51], [-5, 42]]]
      }
    },
    {
      "type": "Feature",
      "id": "POL",
      "properties": { "name": "Poland" },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[14, 49], [24, 49], [24, 55], [14, 55], [14, 49]]]
      }
    }
  ]
}
```

**Step 2: Create example map YAML with GeoJSON reference**

Create `src/modules/maps/europe_geojson_example.yaml`:

```yaml
map:
  name: "Europe GeoJSON Example"
  description: "Example map using GeoJSON overlay"
  geojson: "europe_simple.geojson"

nodes:
  # Region nodes - match GeoJSON feature IDs
  - id: DEU
    name: Germany
    type: region
    properties:
      population: 83000000

  - id: FRA
    name: France
    type: region
    properties:
      population: 67000000

  - id: POL
    name: Poland
    type: region
    properties:
      population: 38000000

  # Point nodes with coordinates
  - id: berlin
    name: Berlin
    type: city
    coordinates: [13.405, 52.52]
    properties:
      capital: true

  - id: paris
    name: Paris
    type: city
    coordinates: [2.35, 48.85]
    properties:
      capital: true

edges:
  - from: DEU
    to: FRA
    type: land
    properties:
      distance_km: 500

  - from: DEU
    to: POL
    type: land
    properties:
      distance_km: 400

  - from: berlin
    to: DEU
    type: road
    properties:
      distance_km: 0

  - from: paris
    to: FRA
    type: road
    properties:
      distance_km: 0

initial_control:
  agent_west: [FRA]
  agent_east: [POL]
  contested: [DEU]
```

**Step 3: Commit**

```bash
git add src/modules/maps/europe_simple.geojson src/modules/maps/europe_geojson_example.yaml
git commit -m "feat(maps): add example GeoJSON map for testing"
```

---

## Task 10: Manual Testing

**Step 1: Build and run**

```bash
# Backend
cd /home/hendrik/coding/ai-safety/apart
python -m pytest tests/ -v --tb=short

# Frontend
cd dashboard
npm run build
```

**Step 2: Test the feature**

1. Create a scenario that uses the `europe_geojson_example.yaml` map
2. Run the simulation
3. Verify the geographic map renders with:
   - Region polygons colored by control
   - Point markers for cities
   - Edges shown on hover
   - Tooltips with node information

**Step 3: Final commit**

```bash
git add -A
git commit -m "test: verify GeoJSON map overlay functionality"
```
