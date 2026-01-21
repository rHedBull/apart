/**
 * Zustand store for simulation state management
 */

import { create } from 'zustand';
import { SimulationEvent } from './useSimulationEvents';

interface AgentMessage {
  step: number;
  timestamp: string;
  agentName: string;
  direction: 'sent' | 'received';
  content: string;
}

interface DangerSignal {
  step: number;
  timestamp: string;
  category: string;
  agentName: string | null;
  metric: string;
  value: number;
  threshold: number | null;
}

interface VariableHistory {
  step: number;
  values: Record<string, unknown>;
}

// Spatial graph types
interface SpatialNode {
  id: string;
  name: string;
  type: string;
  properties: Record<string, unknown>;
  conditions: string[];
  coordinates?: [number, number];  // [lng, lat] for point nodes
}

interface SpatialEdge {
  from: string;
  to: string;
  type: string;
  directed: boolean;
  properties: Record<string, unknown>;
}

export interface SpatialGraph {
  nodes: SpatialNode[];
  edges: SpatialEdge[];
  blocked_edge_types: string[];
}

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

interface SimulationState {
  // Connection status
  connected: boolean;
  setConnected: (connected: boolean) => void;

  // Current simulation
  currentRunId: string | null;
  setCurrentRunId: (runId: string | null) => void;

  // Simulation status
  status: 'idle' | 'running' | 'completed' | 'failed';
  currentStep: number;
  maxSteps: number | null;
  agentNames: string[];

  // Spatial graph (for map visualization)
  spatialGraph: SpatialGraph | null;

  // GeoJSON data (for map overlay)
  geojson: GeoJSONData | null;

  // Messages
  messages: AgentMessage[];
  addMessage: (message: AgentMessage) => void;
  clearMessages: () => void;

  // Danger signals
  dangerSignals: DangerSignal[];
  addDangerSignal: (signal: DangerSignal) => void;
  clearDangerSignals: () => void;

  // Variable history (for charts)
  globalVarsHistory: VariableHistory[];
  agentVarsHistory: Record<string, VariableHistory[]>;
  addVariableSnapshot: (
    step: number,
    globalVars: Record<string, unknown>,
    agentVars: Record<string, Record<string, unknown>>
  ) => void;
  clearVariableHistory: () => void;

  // Process an incoming event
  processEvent: (event: SimulationEvent) => void;

  // Load full run data from API
  loadRunData: (data: {
    runId: string;
    status: string;
    currentStep: number;
    maxSteps: number | null;
    agentNames: string[];
    spatialGraph: SpatialGraph | null;
    geojson: GeoJSONData | null;
    messages: AgentMessage[];
    dangerSignals: DangerSignal[];
    globalVarsHistory: VariableHistory[];
    agentVarsHistory: Record<string, VariableHistory[]>;
  }) => void;

  // Reset all state
  reset: () => void;
}

export const useSimulationStore = create<SimulationState>((set, get) => ({
  // Initial state
  connected: false,
  currentRunId: null,
  status: 'idle',
  currentStep: 0,
  maxSteps: null,
  agentNames: [],
  spatialGraph: null,
  geojson: null,
  messages: [],
  dangerSignals: [],
  globalVarsHistory: [],
  agentVarsHistory: {},

  // Actions
  setConnected: (connected) => set({ connected }),
  setCurrentRunId: (runId) => set({ currentRunId: runId }),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages.slice(-99), message], // Keep last 100
    })),

  clearMessages: () => set({ messages: [] }),

  addDangerSignal: (signal) =>
    set((state) => ({
      dangerSignals: [...state.dangerSignals, signal],
    })),

  clearDangerSignals: () => set({ dangerSignals: [] }),

  addVariableSnapshot: (step, globalVars, agentVars) =>
    set((state) => {
      const newGlobalHistory = [
        ...state.globalVarsHistory,
        { step, values: globalVars },
      ];

      const newAgentHistory = { ...state.agentVarsHistory };
      for (const [agentName, vars] of Object.entries(agentVars)) {
        if (!newAgentHistory[agentName]) {
          newAgentHistory[agentName] = [];
        }
        newAgentHistory[agentName] = [
          ...newAgentHistory[agentName],
          { step, values: vars },
        ];
      }

      return {
        globalVarsHistory: newGlobalHistory,
        agentVarsHistory: newAgentHistory,
      };
    }),

  clearVariableHistory: () =>
    set({ globalVarsHistory: [], agentVarsHistory: {} }),

  processEvent: (event) => {
    const { event_type, timestamp, run_id, step, data } = event;

    // Auto-select this run if none selected
    if (get().currentRunId === null) {
      set({ currentRunId: run_id });
    }

    // Only process events for current run
    if (run_id !== get().currentRunId) {
      return;
    }

    switch (event_type) {
      case 'simulation_started':
        set({
          status: 'running',
          currentStep: 0,
          maxSteps: data.max_steps as number,
          agentNames: (data.agent_names as string[]) || [],
          spatialGraph: (data.spatial_graph as SpatialGraph) || null,
          geojson: (data.geojson as GeoJSONData) || null,
          messages: [],
          dangerSignals: [],
          globalVarsHistory: [],
          agentVarsHistory: {},
        });
        break;

      case 'step_started':
        set({ currentStep: step || 0 });
        break;

      case 'step_completed':
        if (data.global_vars && data.agent_vars) {
          get().addVariableSnapshot(
            step || 0,
            data.global_vars as Record<string, unknown>,
            data.agent_vars as Record<string, Record<string, unknown>>
          );
        }
        break;

      case 'agent_message_sent':
        get().addMessage({
          step: step || 0,
          timestamp,
          agentName: data.agent_name as string,
          direction: 'sent',
          content: data.message as string,
        });
        break;

      case 'agent_response_received':
        get().addMessage({
          step: step || 0,
          timestamp,
          agentName: data.agent_name as string,
          direction: 'received',
          content: data.response as string,
        });
        break;

      case 'danger_signal':
        get().addDangerSignal({
          step: step || 0,
          timestamp,
          category: data.category as string,
          agentName: (data.agent_name as string) || null,
          metric: data.metric as string,
          value: data.value as number,
          threshold: (data.threshold as number) || null,
        });
        break;

      case 'simulation_completed':
        set({ status: 'completed' });
        break;

      case 'simulation_failed':
        set({ status: 'failed' });
        break;
    }
  },

  loadRunData: (data) =>
    set({
      currentRunId: data.runId,
      status: (data.status as 'idle' | 'running' | 'completed' | 'failed') || 'completed',
      currentStep: data.currentStep,
      maxSteps: data.maxSteps,
      agentNames: data.agentNames,
      spatialGraph: data.spatialGraph,
      geojson: data.geojson,
      messages: data.messages,
      dangerSignals: data.dangerSignals,
      globalVarsHistory: data.globalVarsHistory,
      agentVarsHistory: data.agentVarsHistory,
    }),

  reset: () =>
    set({
      connected: false,
      currentRunId: null,
      status: 'idle',
      currentStep: 0,
      maxSteps: null,
      agentNames: [],
      spatialGraph: null,
      geojson: null,
      messages: [],
      dangerSignals: [],
      globalVarsHistory: [],
      agentVarsHistory: {},
    }),
}));
