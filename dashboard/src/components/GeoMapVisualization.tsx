/**
 * Geographic Map Visualization Component
 *
 * Renders a spatial graph on a real geographic map using Leaflet and GeoJSON.
 * Region nodes are rendered as GeoJSON polygons, point nodes as markers.
 * Used when the scenario includes GeoJSON data.
 */

import { useMemo, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, Tooltip } from 'react-leaflet';
import type { LatLngBoundsExpression, LatLngExpression, Layer } from 'leaflet';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import Badge from '@cloudscape-design/components/badge';
import SpaceBetween from '@cloudscape-design/components/space-between';
import { useSimulationStore, GeoJSONData, GeoJSONFeature } from '../hooks/useSimulationState';
import 'leaflet/dist/leaflet.css';

// Faction colors for region control
const factionColors: Record<string, string> = {
  contested: '#fbbf24',  // amber
  neutral: '#9ca3af',    // gray
};

// Generate faction colors dynamically based on agent index
function getFactionColor(index: number): string {
  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#a855f7', '#f97316', '#06b6d4'];
  return colors[index % colors.length];
}

// Compute centroid of a GeoJSON polygon
function computeCentroid(geometry: GeoJSONFeature['geometry']): [number, number] {
  let coords: number[][];

  if (geometry.type === 'Polygon') {
    coords = geometry.coordinates[0] as number[][];
  } else if (geometry.type === 'MultiPolygon') {
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
      map[name] = getFactionColor(i);
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
  const onEachRegion = (feature: GeoJSONFeature, layer: Layer) => {
    const fid = String(feature.id ?? feature.properties?.id ?? '');
    const node = spatialGraph.nodes.find(n => n.id === fid);
    const agents = agentsAtNode[fid] || [];

    layer.on({
      mouseover: () => setHoveredNode(fid),
      mouseout: () => setHoveredNode(null),
      click: () => onNodeClick?.(fid),
    });

    if ('bindTooltip' in layer) {
      (layer as { bindTooltip: (content: string, options?: object) => void }).bindTooltip(
        `<strong>${node?.name || fid}</strong><br/>` +
        `Type: ${node?.type || 'region'}<br/>` +
        (agents.length > 0 ? `Agents: ${agents.join(', ')}` : ''),
        { sticky: true }
      );
    }
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
            data={geojson as GeoJSON.FeatureCollection}
            style={(feature) => getRegionStyle(feature as unknown as GeoJSONFeature)}
            onEachFeature={(feature, layer) => onEachRegion(feature as unknown as GeoJSONFeature, layer)}
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
                  <span>
                    <strong>{node.name}</strong><br/>
                    Type: {node.type}<br/>
                    {agents.length > 0 && `Agents: ${agents.join(', ')}`}
                  </span>
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
