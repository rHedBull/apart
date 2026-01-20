/**
 * Map Visualization Component
 *
 * Renders a spatial graph as an interactive SVG visualization showing:
 * - Nodes (locations) with agent positions
 * - Edges (connections) with types and properties
 * - Agent locations updated in real-time
 */

import { useMemo, useState, useEffect, useRef } from 'react';
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

// Color mapping for node types
const nodeColors: Record<string, string> = {
  nation: '#3b82f6',      // blue
  port: '#10b981',        // green
  city: '#8b5cf6',        // purple
  region: '#f59e0b',      // amber
  base: '#ef4444',        // red
  sea_zone: '#06b6d4',    // cyan
  waterway: '#0891b2',    // darker cyan
  location: '#6b7280',    // gray
};

// Color mapping for edge types
const edgeColors: Record<string, string> = {
  maritime: '#0ea5e9',    // sky blue
  land: '#84cc16',        // lime
  air: '#a855f7',         // purple
  road: '#78716c',        // stone
  highway: '#fbbf24',     // yellow
  connection: '#9ca3af',  // gray
};

export function MapVisualization() {
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
        setDimensions({ width: rect.width, height: rect.height });
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
      <div className="h-full bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4 flex items-center justify-center">
        <div className="text-slate-400 dark:text-slate-500 text-center">
          <div className="text-lg font-medium mb-2">No Spatial Graph</div>
          <div className="text-sm">
            Run a scenario with <code className="bg-slate-100 dark:bg-slate-700 px-1 rounded">spatial_model: "graph"</code> to see the map
          </div>
        </div>
      </div>
    );
  }

  const { width, height } = dimensions;
  const blockedEdgeTypes = new Set(spatialGraph.blocked_edge_types);

  return (
    <div
      ref={containerRef}
      className="h-full bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden"
    >
      {/* Header */}
      <div className="px-3 py-2 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <h3 className="font-semibold text-slate-700 dark:text-slate-200">
          Spatial Map
        </h3>
        <div className="text-xs text-slate-500 dark:text-slate-400">
          {layoutNodes.length} locations | {spatialGraph.edges.length} connections
        </div>
      </div>

      {/* SVG Map */}
      <svg
        width={width}
        height={height - 40}
        className="bg-slate-50 dark:bg-slate-900"
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
                stroke={isBlocked ? '#dc2626' : color}
                strokeWidth={isHovered ? 3 : 2}
                strokeDasharray={isBlocked ? '5,5' : undefined}
                opacity={isBlocked ? 0.5 : 0.8}
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHoveredEdge(`${edge.from}-${edge.to}`)}
                onMouseLeave={() => setHoveredEdge(null)}
              />
              {/* Edge label on hover */}
              {isHovered && (
                <text
                  x={(from.x + to.x) / 2}
                  y={(from.y + to.y) / 2 - 8}
                  textAnchor="middle"
                  className="fill-slate-600 dark:fill-slate-300 text-xs font-medium pointer-events-none"
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
              className="cursor-pointer"
              onMouseEnter={() => setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              {/* Node circle */}
              <circle
                r={isHovered ? radius + 4 : radius}
                fill={color}
                stroke={hasAgents ? '#fbbf24' : '#fff'}
                strokeWidth={hasAgents ? 3 : 2}
                className="transition-all"
                opacity={0.9}
              />

              {/* Agent indicator */}
              {hasAgents && (
                <circle
                  r={8}
                  cx={radius - 6}
                  cy={-radius + 6}
                  fill="#fbbf24"
                  stroke="#fff"
                  strokeWidth={2}
                />
              )}
              {hasAgents && (
                <text
                  x={radius - 6}
                  y={-radius + 10}
                  textAnchor="middle"
                  className="fill-slate-800 text-xs font-bold pointer-events-none"
                >
                  {agents.length}
                </text>
              )}

              {/* Node type icon/letter */}
              <text
                y={5}
                textAnchor="middle"
                className="fill-white text-sm font-bold pointer-events-none"
              >
                {node.type[0].toUpperCase()}
              </text>

              {/* Node name label */}
              <text
                y={radius + 14}
                textAnchor="middle"
                className="fill-slate-700 dark:fill-slate-300 text-xs font-medium pointer-events-none"
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
                  className="pointer-events-none"
                >
                  <div className="bg-slate-800 text-white text-xs rounded px-2 py-1 shadow-lg">
                    <div className="font-bold">{node.name}</div>
                    <div className="text-slate-300">Type: {node.type}</div>
                    {agents.length > 0 && (
                      <div className="text-yellow-400">
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
      <div className="absolute bottom-2 left-2 bg-white/90 dark:bg-slate-800/90 rounded px-2 py-1 text-xs flex gap-3">
        {blockedEdgeTypes.size > 0 && (
          <span className="text-red-500 flex items-center gap-1">
            <span className="w-4 h-0.5 bg-red-500" style={{ borderStyle: 'dashed' }}></span>
            Blocked: {Array.from(blockedEdgeTypes).join(', ')}
          </span>
        )}
        <span className="text-yellow-500 flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
          Agent present
        </span>
      </div>
    </div>
  );
}
