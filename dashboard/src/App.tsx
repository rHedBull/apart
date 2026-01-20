/**
 * Apart Dashboard - Main Application
 */

import { useCallback } from 'react';
import { useSimulationEvents, SimulationEvent } from './hooks/useSimulationEvents';
import { useSimulationStore } from './hooks/useSimulationState';
import { AgentPanel } from './components/AgentPanel';
import { VariableChart } from './components/VariableChart';
import { DangerPanel } from './components/DangerPanel';
import { SimulationSelector } from './components/SimulationSelector';
import { StatusBar } from './components/StatusBar';

function App() {
  const currentRunId = useSimulationStore((state) => state.currentRunId);
  const setConnected = useSimulationStore((state) => state.setConnected);
  const processEvent = useSimulationStore((state) => state.processEvent);

  // Event handlers
  const handleEvent = useCallback(
    (event: SimulationEvent) => {
      processEvent(event);
    },
    [processEvent]
  );

  const handleConnect = useCallback(() => {
    setConnected(true);
  }, [setConnected]);

  const handleDisconnect = useCallback(() => {
    setConnected(false);
  }, [setConnected]);

  // Connect to SSE
  useSimulationEvents({
    runId: currentRunId || undefined,
    includeHistory: true,
    onEvent: handleEvent,
    onConnect: handleConnect,
    onDisconnect: handleDisconnect,
  });

  return (
    <div className="h-screen flex flex-col bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-slate-800 dark:text-slate-100">
              Apart Dashboard
            </h1>
            <span className="text-xs text-slate-400 dark:text-slate-500">
              AI Safety Simulation Monitor
            </span>
          </div>
          <SimulationSelector />
        </div>
      </header>

      {/* Status bar */}
      <StatusBar />

      {/* Main content */}
      <main className="flex-1 p-4 min-h-0">
        <div className="h-full grid grid-cols-12 gap-4">
          {/* Left column: Agent messages */}
          <div className="col-span-5 h-full min-h-0">
            <AgentPanel />
          </div>

          {/* Right column: Charts and Danger */}
          <div className="col-span-7 h-full min-h-0 flex flex-col gap-4">
            {/* Variables chart */}
            <div className="flex-1 min-h-0">
              <VariableChart />
            </div>

            {/* Danger panel */}
            <div className="h-64">
              <DangerPanel />
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="px-4 py-2 bg-slate-100 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between text-xs text-slate-400 dark:text-slate-500">
          <span>Apart v0.1.0</span>
          <span>
            Backend: http://localhost:8000 | Dashboard: http://localhost:3000
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
