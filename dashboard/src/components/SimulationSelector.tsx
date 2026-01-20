/**
 * SimulationSelector - Dropdown to select simulation run
 */

import { useEffect, useState } from 'react';
import { fetchSimulations, SimulationSummary } from '../lib/api';
import { useSimulationStore } from '../hooks/useSimulationState';

export function SimulationSelector() {
  const [simulations, setSimulations] = useState<SimulationSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const currentRunId = useSimulationStore((state) => state.currentRunId);
  const setCurrentRunId = useSimulationStore((state) => state.setCurrentRunId);
  const reset = useSimulationStore((state) => state.reset);

  // Fetch simulations on mount and periodically
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const data = await fetchSimulations();
        setSimulations(data);
      } catch (err) {
        console.error('Failed to fetch simulations:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5s

    return () => clearInterval(interval);
  }, []);

  const handleSelect = (runId: string) => {
    if (runId === '') {
      reset();
    } else {
      setCurrentRunId(runId);
    }
  };

  // Find running simulation for auto-select hint
  const runningSim = simulations.find((s) => s.status === 'running');

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-slate-500 dark:text-slate-400">
        Simulation:
      </label>
      <select
        value={currentRunId || ''}
        onChange={(e) => handleSelect(e.target.value)}
        className="px-3 py-1.5 text-sm bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded text-slate-700 dark:text-slate-200"
        disabled={loading}
      >
        <option value="">Select a run...</option>
        {simulations.map((sim) => (
          <option key={sim.run_id} value={sim.run_id}>
            {sim.run_id} - {sim.scenario_name || 'Unknown'} ({sim.status})
            {sim.status === 'running' && ` [Step ${sim.current_step}/${sim.max_steps}]`}
          </option>
        ))}
      </select>

      {/* Auto-select hint */}
      {!currentRunId && runningSim && (
        <button
          onClick={() => setCurrentRunId(runningSim.run_id)}
          className="px-2 py-1 text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded"
        >
          Join running: {runningSim.run_id}
        </button>
      )}
    </div>
  );
}
