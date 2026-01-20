/**
 * StatusBar - Shows connection and simulation status
 */

import { useSimulationStore } from '../hooks/useSimulationState';

export function StatusBar() {
  const connected = useSimulationStore((state) => state.connected);
  const status = useSimulationStore((state) => state.status);
  const currentStep = useSimulationStore((state) => state.currentStep);
  const maxSteps = useSimulationStore((state) => state.maxSteps);
  const currentRunId = useSimulationStore((state) => state.currentRunId);

  const statusColors: Record<string, string> = {
    idle: 'bg-slate-400',
    running: 'bg-green-500',
    completed: 'bg-blue-500',
    failed: 'bg-red-500',
  };

  const statusLabels: Record<string, string> = {
    idle: 'Idle',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
  };

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700">
      {/* Connection indicator */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${
            connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`}
        />
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {/* Run ID */}
      {currentRunId && (
        <div className="text-xs text-slate-500 dark:text-slate-400">
          Run: <span className="font-mono text-slate-700 dark:text-slate-300">{currentRunId}</span>
        </div>
      )}

      {/* Status badge */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${statusColors[status] || 'bg-slate-400'}`}
        />
        <span className="text-xs text-slate-600 dark:text-slate-300">
          {statusLabels[status] || 'Unknown'}
        </span>
      </div>

      {/* Progress */}
      {status === 'running' && maxSteps && (
        <div className="flex items-center gap-2 flex-1">
          <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${(currentStep / maxSteps) * 100}%` }}
            />
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400 min-w-[4rem] text-right">
            {currentStep} / {maxSteps}
          </span>
        </div>
      )}
    </div>
  );
}
