# GeoJSON Map Overlay Design

**Date:** 2026-01-21
**Status:** Approved

## Overview

Add geographic map visualization to the territory module. The underlying data structure remains a graph, but the UI renders an actual map using GeoJSON polygons for regions. This enables realistic territorial simulations with real-world geography.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Map type | Real-world geography with actual country/region boundaries |
| Node types | Both regions (polygons) and points (cities/ports) |
| GeoJSON source | Bundled with map file, referenced via path |
| Rendering library | Leaflet (react-leaflet) |
| Edge display | Hidden by default, show on hover/selection |
| Control visualization | Region fill color by faction |
| Node-to-feature linking | ID matching (node.id = feature.id) |

## Data Model Changes

### Map YAML Additions

```yaml
map:
  name: "European Conflict"
  geojson: "maps/europe.geojson"  # Path relative to map file

nodes:
  # Region node - matches GeoJSON feature by ID
  - id: "DEU"              # Must match GeoJSON feature id
    name: "Germany"
    type: region           # "region" type = polygon from GeoJSON
    properties:
      population: 83000000

  # Point node - explicit coordinates
  - id: "berlin_hq"
    name: "Berlin Command"
    type: base
    coordinates: [13.405, 52.52]  # [lng, lat] for point nodes
    properties:
      capacity: 5000
```

### Node Type Behavior

- `type: region` → Rendered as GeoJSON polygon, ID must match a feature
- `type: city|port|base|etc` → Rendered as marker at `coordinates` (required for non-region nodes when using geographic map)

### Backend Changes

- `SpatialGraph` gets optional `geojson_path` field
- `Node` dataclass gets optional `coordinates: tuple[float, float]` field
- `map_loader.py` validates that region nodes have matching GeoJSON features
- GeoJSON file loaded and passed through API alongside graph data

## Backend & API Changes

### File Loading Pipeline (map_loader.py)

```python
def load_map_file(map_path: Path) -> tuple[SpatialGraph, dict, dict | None]:
    """Returns (graph, metadata, geojson_data)"""

    # Load map YAML
    map_data = yaml.safe_load(map_path.read_text())

    # Load GeoJSON if specified
    geojson_data = None
    if geojson_path := map_data.get("map", {}).get("geojson"):
        geojson_file = map_path.parent / geojson_path
        geojson_data = json.loads(geojson_file.read_text())

        # Validate region nodes have matching features
        feature_ids = {f["id"] or f["properties"].get("id")
                       for f in geojson_data["features"]}
        for node in map_data["nodes"]:
            if node["type"] == "region" and node["id"] not in feature_ids:
                raise ValueError(f"Region node '{node['id']}' has no matching GeoJSON feature")

    # Build graph (existing logic + coordinates field)
    graph = build_spatial_graph(map_data)

    return graph, metadata, geojson_data
```

### API Response Changes

The `/api/runs/{runId}` endpoint already returns `spatialGraph`. Add `geojson` alongside it:

```json
{
  "spatialGraph": { "nodes": [...], "edges": [...] },
  "geojson": { "type": "FeatureCollection", "features": [...] }
}
```

### Composer Updates

- `ComposedModules` gets `geojson: dict | None` field
- Passed through simulation and serialized to API

## Frontend Changes

### New TypeScript Types (useSimulationState.ts)

```typescript
// Extend SpatialNode with optional coordinates
interface SpatialNode {
  id: string;
  name: string;
  type: string;
  properties: Record<string, unknown>;
  conditions: string[];
  coordinates?: [number, number];  // [lng, lat] for point nodes
}

// GeoJSON types
interface GeoJSONFeature {
  type: "Feature";
  id?: string;
  properties: Record<string, unknown>;
  geometry: {
    type: "Polygon" | "MultiPolygon";
    coordinates: number[][][];
  };
}

interface GeoJSONData {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}
```

### Store Additions

```typescript
interface SimulationStore {
  // Existing
  spatialGraph: SpatialGraph | null;

  // New
  geojson: GeoJSONData | null;

  // Actions
  loadRunData: (data: RunData) => void;  // Updated to handle geojson
}
```

## Leaflet Map Component

### New Component: GeoMapVisualization.tsx

```typescript
import { MapContainer, TileLayer, GeoJSON, Marker, Tooltip } from 'react-leaflet';

interface Props {
  spatialGraph: SpatialGraph;
  geojson: GeoJSONData;
  agentLocations: Record<string, string>;  // agentId -> nodeId
  regionControl: Record<string, string>;   // nodeId -> agentId/faction
}
```

### Rendering Layers (bottom to top)

1. **Base tile layer** (optional) - OpenStreetMap or similar for context
2. **GeoJSON regions** - Polygons styled by control status:
   - Uncontrolled: neutral gray fill
   - Controlled: faction color fill (opacity ~0.5)
   - Contested: striped pattern or dual-color
3. **Point markers** - Cities/bases/ports rendered as Leaflet markers at coordinates
4. **Agent indicators** - Badge or count overlay on occupied nodes

### Interaction

- Hover region/marker → Tooltip with node name, type, agents present
- Hover region → Show edges to connected nodes (temporary lines)
- Click region/marker → Select node (callback to parent for details panel)

### Auto-fit Bounds

On load, calculate bounding box of all GeoJSON features and fit map view.

## Edge Display on Hover

When a node is hovered:

1. Find all edges connected to that node from `spatialGraph.edges`
2. For each connected node, determine its position:
   - Region nodes → centroid of GeoJSON polygon
   - Point nodes → use `coordinates` directly
3. Draw temporary polylines from hovered node to each neighbor
4. Style by edge type (land=green, maritime=teal, etc.)
5. Clear lines on mouse leave

```typescript
const [hoveredNode, setHoveredNode] = useState<string | null>(null);

// Compute centroids once from GeoJSON
const regionCentroids = useMemo(() => computeCentroids(geojson), [geojson]);

// Get position for any node
const getNodePosition = (nodeId: string): [number, number] => {
  const node = spatialGraph.nodes.find(n => n.id === nodeId);
  if (node?.coordinates) return node.coordinates;
  return regionCentroids[nodeId];
};
```

### Control Colors

```typescript
const factionColors: Record<string, string> = {
  faction_a: "#3b82f6",  // blue
  faction_b: "#ef4444",  // red
  contested: "#fbbf24",  // yellow/amber
  neutral: "#9ca3af",    // gray
};
```

## Component Integration

### Conditional Rendering in RunDetailPage.tsx

```typescript
// Use geographic map if geojson is available, otherwise fall back to force-directed
{geojson ? (
  <GeoMapVisualization
    spatialGraph={spatialGraph}
    geojson={geojson}
    agentLocations={currentAgentLocations}
    regionControl={currentRegionControl}
    onNodeClick={handleNodeClick}
  />
) : (
  <MapVisualization
    spatialGraph={spatialGraph}
    agentLocations={currentAgentLocations}
    onNodeClick={handleNodeClick}
  />
)}
```

### Why Keep Both

- `MapVisualization` (existing) - Works for abstract/non-geographic scenarios
- `GeoMapVisualization` (new) - Used when map has GeoJSON data

### Deriving regionControl from Agent Variables

```typescript
const currentRegionControl = useMemo(() => {
  const control: Record<string, string> = {};

  for (const [agentId, vars] of Object.entries(agentVarsHistory)) {
    const controlledRegions = vars.controlled_regions?.value || [];
    for (const regionId of controlledRegions) {
      if (control[regionId]) {
        control[regionId] = "contested";
      } else {
        control[regionId] = agentId;
      }
    }
  }

  return control;
}, [agentVarsHistory, currentStep]);
```

## Implementation Plan

### Files to Modify

| File | Changes |
|------|---------|
| `src/utils/spatial_graph.py` | Add `coordinates` field to `Node`, `geojson_path` to `SpatialGraph` |
| `src/modules/map_loader.py` | Load GeoJSON file, validate region nodes, return geojson data |
| `src/modules/composer.py` | Pass geojson through `ComposedModules` |
| `dashboard/src/hooks/useSimulationState.ts` | Add `geojson` to store, extend `SpatialNode` type with `coordinates` |
| `dashboard/src/pages/RunDetailPage.tsx` | Conditional render between map components |

### New Files

| File | Purpose |
|------|---------|
| `dashboard/src/components/GeoMapVisualization.tsx` | Leaflet-based geographic map component |
| `src/modules/maps/europe.geojson` | Example GeoJSON file for testing |

### New Dependencies

```bash
# Dashboard
npm install leaflet react-leaflet @types/leaflet
```

### Not Changing

- `MapVisualization.tsx` - Keep as-is for fallback
- Movement validation logic - Graph structure unchanged
- API endpoint structure - Just adding `geojson` field to response
