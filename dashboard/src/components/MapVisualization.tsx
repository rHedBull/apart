/**
 * Map Visualization Component with Cloudscape styling
 *
 * Renders a spatial graph as an interactive SVG visualization showing:
 * - Nodes (locations) with agent positions
 * - Edges (connections) with types and properties
 * - Agent locations updated in real-time
 */

import { useMemo, useState, useEffect, useRef } from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import Badge from '@cloudscape-design/components/badge';
import SpaceBetween from '@cloudscape-design/components/space-between';
import { useSimulationStore, SpatialGraph } from '../hooks/useSimulationState';

// Node position after layout
interface LayoutNode {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  properties: Record<string, unknown>;
}

// Force-directed layout simulation
function computeLayout(graph: SpatialGraph, width: number, height: number): LayoutNode[] {
  const nodes = graph.nodes.map((n, i) => ({
    ...n,
    x: width / 2 + (Math.cos((i * 2 * Math.PI) / graph.nodes.length) * width * 0.35),
    y: height / 2 + (Math.sin((i * 2 * Math.PI) / graph.nodes.length) * height * 0.35),
  }));

  // Simple force-directed layout (few iterations)
  const iterations = 100;
  const repulsion = 5000;
  const attraction = 0.05;
  const dampening = 0.9;

  // Build edge lookup
  const edgeSet = new Set<string>();
  for (const edge of graph.edges) {
    edgeSet.add(`${edge.from}-${edge.to}`);
    if (!edge.directed) {
      edgeSet.add(`${edge.to}-${edge.from}`);
    }
  }

  const velocities = nodes.map(() => ({ vx: 0, vy: 0 }));

  for (let iter = 0; iter < iterations; iter++) {
    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = repulsion / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        velocities[i].vx -= fx;
        velocities[i].vy -= fy;
        velocities[j].vx += fx;
        velocities[j].vy += fy;
      }
    }

    // Attraction along edges
    for (const edge of graph.edges) {
      const iFrom = nodes.findIndex(n => n.id === edge.from);
      const iTo = nodes.findIndex(n => n.id === edge.to);
      if (iFrom >= 0 && iTo >= 0) {
        const dx = nodes[iTo].x - nodes[iFrom].x;
        const dy = nodes[iTo].y - nodes[iFrom].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = dist * attraction;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        velocities[iFrom].vx += fx;
        velocities[iFrom].vy += fy;
        velocities[iTo].vx -= fx;
        velocities[iTo].vy -= fy;
      }
    }

    // Apply velocities with dampening
    for (let i = 0; i < nodes.length; i++) {
      velocities[i].vx *= dampening;
      velocities[i].vy *= dampening;
      nodes[i].x += velocities[i].vx;
      nodes[i].y += velocities[i].vy;

      // Keep within bounds
      const padding = 60;
      nodes[i].x = Math.max(padding, Math.min(width - padding, nodes[i].x));
      nodes[i].y = Math.max(padding, Math.min(height - padding, nodes[i].y));
    }
  }

  return nodes;
}

// Color mapping for node types (Cloudscape-aligned)
const nodeColors: Record<string, string> = {
  nation: '#0972d3',      // blue
  port: '#1d8102',        // green
  city: '#9469d6',        // purple
  region: '#ec7211',      // orange
  base: '#d91515',        // red
  sea_zone: '#2ea597',    // teal
  waterway: '#067f68',    // dark teal
  location: '#5f6b7a',    // grey
};

// Color mapping for edge types
const edgeColors: Record<string, string> = {
  maritime: '#2ea597',    // teal
  land: '#1d8102',        // green
  air: '#9469d6',         // purple
  road: '#5f6b7a',        // grey
  highway: '#ec7211',     // orange
  connection: '#8d99a8',  // light grey
};

interface MapVisualizationProps {
  onNodeClick?: (nodeId: string) => void;
}

export function MapVisualization({ onNodeClick }: MapVisualizationProps) {
  const spatialGraph = useSimulationStore((state) => state.spatialGraph);
  const agentVarsHistory = useSimulationStore((state) => state.agentVarsHistory);

  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: Math.max(rect.height - 60, 300) });
      }
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Compute layout when graph changes
  const layoutNodes = useMemo(() => {
    if (!spatialGraph) return [];
    return computeLayout(spatialGraph, dimensions.width, dimensions.height);
  }, [spatialGraph, dimensions.width, dimensions.height]);

  // Get current agent locations from latest variable snapshot
  const agentLocations = useMemo(() => {
    const locations: Record<string, string> = {};
    for (const [agentName, history] of Object.entries(agentVarsHistory)) {
      if (history.length > 0) {
        const latest = history[history.length - 1];
        const location = latest.values.location;
        if (location) {
          // Handle both direct string and {value: string} format
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

  // Build a map of node positions
  const nodePositions = useMemo(() => {
    const map: Record<string, { x: number; y: number }> = {};
    for (const node of layoutNodes) {
      map[node.id] = { x: node.x, y: node.y };
    }
    return map;
  }, [layoutNodes]);

  // Get agents at each node
  const agentsAtNode = useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const [agent, location] of Object.entries(agentLocations)) {
      if (!map[location]) map[location] = [];
      map[location].push(agent);
    }
    return map;
  }, [agentLocations]);

  if (!spatialGraph) {
    return (
      <Container
        header={<Header variant="h2">Spatial Map</Header>}
        fitHeight
      >
        <Box textAlign="center" color="text-status-inactive" padding="xxl">
          <Box variant="h3" margin={{ bottom: 's' }}>No Spatial Graph</Box>
          <Box variant="p">
            Run a scenario with <code>spatial_model: "graph"</code> to see the map
          </Box>
        </Box>
      </Container>
    );
  }

  const { width, height } = dimensions;
  const blockedEdgeTypes = new Set(spatialGraph.blocked_edge_types);

  return (
    <Container
      header={
        <Header
          variant="h2"
          counter={`${layoutNodes.length} locations`}
        >
          Spatial Map
        </Header>
      }
      fitHeight
    >
      <div ref={containerRef} style={{ height: '100%', minHeight: 350 }}>
        <svg
          width={width}
          height={height}
          style={{ backgroundColor: '#fafafa' }}
        >
          {/* Render edges */}
          {spatialGraph.edges.map((edge, i) => {
            const from = nodePositions[edge.from];
            const to = nodePositions[edge.to];
            if (!from || !to) return null;

            const isBlocked = blockedEdgeTypes.has(edge.type);
            const isHovered = hoveredEdge === `${edge.from}-${edge.to}`;
            const color = edgeColors[edge.type] || edgeColors.connection;
            const distance = edge.properties.distance_km as number || 0;

            return (
              <g key={`edge-${i}`}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={isBlocked ? '#d91515' : color}
                  strokeWidth={isHovered ? 3 : 2}
                  strokeDasharray={isBlocked ? '5,5' : undefined}
                  opacity={isBlocked ? 0.5 : 0.8}
                  style={{ cursor: 'pointer', transition: 'stroke-width 0.2s' }}
                  onMouseEnter={() => setHoveredEdge(`${edge.from}-${edge.to}`)}
                  onMouseLeave={() => setHoveredEdge(null)}
                />
                {/* Edge label on hover */}
                {isHovered && (
                  <text
                    x={(from.x + to.x) / 2}
                    y={(from.y + to.y) / 2 - 8}
                    textAnchor="middle"
                    fill="#16191f"
                    fontSize="12"
                    fontWeight="500"
                    style={{ pointerEvents: 'none' }}
                  >
                    {edge.type} {distance > 0 ? `(${distance}km)` : ''}
                  </text>
                )}
              </g>
            );
          })}

          {/* Render nodes */}
          {layoutNodes.map((node) => {
            const agents = agentsAtNode[node.id] || [];
            const isHovered = hoveredNode === node.id;
            const hasAgents = agents.length > 0;
            const color = nodeColors[node.type] || nodeColors.location;
            const radius = hasAgents ? 24 : 18;

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onClick={() => onNodeClick?.(node.id)}
              >
                {/* Node circle */}
                <circle
                  r={isHovered ? radius + 4 : radius}
                  fill={color}
                  stroke={hasAgents ? '#ec7211' : '#fff'}
                  strokeWidth={hasAgents ? 3 : 2}
                  style={{ transition: 'r 0.2s' }}
                  opacity={0.9}
                />

                {/* Agent indicator */}
                {hasAgents && (
                  <circle
                    r={8}
                    cx={radius - 6}
                    cy={-radius + 6}
                    fill="#ec7211"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
                {hasAgents && (
                  <text
                    x={radius - 6}
                    y={-radius + 10}
                    textAnchor="middle"
                    fill="#16191f"
                    fontSize="11"
                    fontWeight="bold"
                    style={{ pointerEvents: 'none' }}
                  >
                    {agents.length}
                  </text>
                )}

                {/* Node type icon/letter */}
                <text
                  y={5}
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="14"
                  fontWeight="bold"
                  style={{ pointerEvents: 'none' }}
                >
                  {node.type[0].toUpperCase()}
                </text>

                {/* Node name label */}
                <text
                  y={radius + 14}
                  textAnchor="middle"
                  fill="#16191f"
                  fontSize="11"
                  fontWeight="500"
                  style={{ pointerEvents: 'none' }}
                >
                  {node.name}
                </text>

                {/* Tooltip on hover */}
                {isHovered && (
                  <foreignObject
                    x={-100}
                    y={-80}
                    width={200}
                    height={70}
                    style={{ pointerEvents: 'none' }}
                  >
                    <div style={{
                      backgroundColor: '#16191f',
                      color: '#fff',
                      fontSize: '12px',
                      borderRadius: '8px',
                      padding: '8px 12px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
                    }}>
                      <div style={{ fontWeight: 'bold' }}>{node.name}</div>
                      <div style={{ color: '#8d99a8' }}>Type: {node.type}</div>
                      {agents.length > 0 && (
                        <div style={{ color: '#ec7211' }}>
                          Agents: {agents.join(', ')}
                        </div>
                      )}
                    </div>
                  </foreignObject>
                )}
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <Box padding="xs">
          <SpaceBetween direction="horizontal" size="s">
            {blockedEdgeTypes.size > 0 && (
              <Badge color="red">
                Blocked: {Array.from(blockedEdgeTypes).join(', ')}
              </Badge>
            )}
            <Badge color="blue">
              {Object.keys(agentLocations).length} agents tracked
            </Badge>
          </SpaceBetween>
        </Box>
      </div>
    </Container>
  );
}
