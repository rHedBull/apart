/**
 * Apart Dashboard - Main Application
 */

import { useCallback, useState } from 'react';
import { useSimulationEvents, SimulationEvent } from './hooks/useSimulationEvents';
import { useSimulationStore } from './hooks/useSimulationState';
import { AgentPanel } from './components/AgentPanel';
import { VariableChart } from './components/VariableChart';
import { DangerPanel } from './components/DangerPanel';
import { SimulationSelector } from './components/SimulationSelector';
import { StatusBar } from './components/StatusBar';
import { MapVisualization } from './components/MapVisualization';

function App() {
  const currentRunId = useSimulationStore((state) => state.currentRunId);
  const setConnected = useSimulationStore((state) => state.setConnected);
  const processEvent = useSimulationStore((state) => state.processEvent);
  const spatialGraph = useSimulationStore((state) => state.spatialGraph);

  // Tab state for right panel (chart vs map)
  const [rightPanelTab, setRightPanelTab] = useState<'chart' | 'map'>('chart');

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

          {/* Right column: Charts/Map and Danger */}
          <div className="col-span-7 h-full min-h-0 flex flex-col gap-4">
            {/* Tabs for chart/map */}
            <div className="flex-1 min-h-0 flex flex-col">
              {/* Tab headers */}
              <div className="flex border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 rounded-t-lg">
                <button
                  onClick={() => setRightPanelTab('chart')}
                  className={`px-4 py-2 text-sm font-medium transition-colors ${
                    rightPanelTab === 'chart'
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  📊 Variables
                </button>
                <button
                  onClick={() => setRightPanelTab('map')}
                  className={`px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2 ${
                    rightPanelTab === 'map'
                      ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  🗺️ Map
                  {spatialGraph && (
                    <span className="px-1.5 py-0.5 text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded">
                      {spatialGraph.nodes.length}
                    </span>
                  )}
                </button>
              </div>

              {/* Tab content */}
              <div className="flex-1 min-h-0">
                {rightPanelTab === 'chart' ? <VariableChart /> : <MapVisualization />}
              </div>
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
