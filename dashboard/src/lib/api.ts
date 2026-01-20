/**
 * API client for Apart Dashboard
 */

const API_BASE = '/api';

export interface SimulationSummary {
  run_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  scenario_name: string | null;
  started_at: string | null;
  completed_at: string | null;
  current_step: number;
  max_steps: number | null;
  agent_count: number;
}

export interface AgentInfo {
  name: string;
  llm_provider: string | null;
  llm_model: string | null;
}

export interface SimulationDetails extends SimulationSummary {
  agents: AgentInfo[];
  config_path: string | null;
  error_message: string | null;
}

export interface SimulationState {
  run_id: string;
  step: number;
  global_variables: Record<string, unknown>;
  agent_variables: Record<string, Record<string, unknown>>;
  messages: Array<{
    from: string;
    to: string;
    content: string;
  }>;
}

export interface DangerSignal {
  category: string;
  description: string;
  confidence: number;
  step: number;
  agent_name: string | null;
  timestamp: string;
}

export interface DangerSummary {
  run_id: string;
  total_signals: number;
  by_category: Record<string, number>;
  signals: DangerSignal[];
}

/**
 * Fetch all simulations
 */
export async function fetchSimulations(): Promise<SimulationSummary[]> {
  const response = await fetch(`${API_BASE}/simulations`);
  if (!response.ok) {
    throw new Error(`Failed to fetch simulations: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch details for a specific simulation
 */
export async function fetchSimulationDetails(runId: string): Promise<SimulationDetails> {
  const response = await fetch(`${API_BASE}/simulations/${runId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch simulation ${runId}: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch current state for a simulation
 */
export async function fetchSimulationState(runId: string): Promise<SimulationState> {
  const response = await fetch(`${API_BASE}/simulations/${runId}/state`);
  if (!response.ok) {
    throw new Error(`Failed to fetch state for ${runId}: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch danger signals for a simulation
 */
export async function fetchDangerSignals(runId: string): Promise<DangerSummary> {
  const response = await fetch(`${API_BASE}/simulations/${runId}/danger`);
  if (!response.ok) {
    throw new Error(`Failed to fetch danger signals for ${runId}: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Check API health
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
