/**
 * VariableChart - Line chart for variable history
 */

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useSimulationStore } from '../hooks/useSimulationState';

// Colors for different variables
const COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#84cc16', // lime
];

interface ChartData {
  step: number;
  [key: string]: number;
}

export function VariableChart() {
  const globalVarsHistory = useSimulationStore((state) => state.globalVarsHistory);
  const agentVarsHistory = useSimulationStore((state) => state.agentVarsHistory);
  const agentNames = useSimulationStore((state) => state.agentNames);

  const [showGlobal, setShowGlobal] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [selectedVars, setSelectedVars] = useState<Set<string>>(new Set());

  // Get available variables
  const availableVars: string[] = [];
  if (showGlobal && globalVarsHistory.length > 0) {
    const lastSnapshot = globalVarsHistory[globalVarsHistory.length - 1];
    Object.keys(lastSnapshot.values).forEach((key) => {
      if (typeof lastSnapshot.values[key] === 'number') {
        availableVars.push(key);
      }
    });
  } else if (selectedAgent && agentVarsHistory[selectedAgent]?.length > 0) {
    const lastSnapshot = agentVarsHistory[selectedAgent][agentVarsHistory[selectedAgent].length - 1];
    Object.keys(lastSnapshot.values).forEach((key) => {
      if (typeof lastSnapshot.values[key] === 'number') {
        availableVars.push(key);
      }
    });
  }

  // Build chart data
  const chartData: ChartData[] = [];
  if (showGlobal) {
    globalVarsHistory.forEach((snapshot) => {
      const point: ChartData = { step: snapshot.step };
      Object.entries(snapshot.values).forEach(([key, value]) => {
        if (typeof value === 'number' && (selectedVars.size === 0 || selectedVars.has(key))) {
          point[key] = value;
        }
      });
      chartData.push(point);
    });
  } else if (selectedAgent && agentVarsHistory[selectedAgent]) {
    agentVarsHistory[selectedAgent].forEach((snapshot) => {
      const point: ChartData = { step: snapshot.step };
      Object.entries(snapshot.values).forEach(([key, value]) => {
        if (typeof value === 'number' && (selectedVars.size === 0 || selectedVars.has(key))) {
          point[key] = value;
        }
      });
      chartData.push(point);
    });
  }

  // Get variables to show
  const varsToShow = selectedVars.size > 0
    ? Array.from(selectedVars)
    : availableVars.slice(0, 5); // Limit to 5 if none selected

  const toggleVar = (varName: string) => {
    const newSelected = new Set(selectedVars);
    if (newSelected.has(varName)) {
      newSelected.delete(varName);
    } else {
      newSelected.add(varName);
    }
    setSelectedVars(newSelected);
  };

  return (
    <div className="flex flex-col h-full min-h-0 bg-white dark:bg-slate-800 rounded-lg shadow-md">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
          Variables Over Time
        </h2>

        {/* Controls */}
        <div className="flex flex-wrap gap-2 mt-2">
          <button
            onClick={() => {
              setShowGlobal(true);
              setSelectedAgent(null);
            }}
            className={`px-3 py-1 text-sm rounded ${
              showGlobal
                ? 'bg-blue-500 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Global
          </button>
          {agentNames.map((name) => (
            <button
              key={name}
              onClick={() => {
                setShowGlobal(false);
                setSelectedAgent(name);
              }}
              className={`px-3 py-1 text-sm rounded ${
                !showGlobal && selectedAgent === name
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
              }`}
            >
              {name}
            </button>
          ))}
        </div>

        {/* Variable toggles */}
        {availableVars.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {availableVars.map((varName, idx) => (
              <button
                key={varName}
                onClick={() => toggleVar(varName)}
                className={`px-2 py-0.5 text-xs rounded border ${
                  selectedVars.size === 0 || selectedVars.has(varName)
                    ? 'border-transparent text-white'
                    : 'border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400 bg-transparent'
                }`}
                style={{
                  backgroundColor:
                    selectedVars.size === 0 || selectedVars.has(varName)
                      ? COLORS[idx % COLORS.length]
                      : undefined,
                }}
              >
                {varName}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0 p-4">
        {chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400 dark:text-slate-500">
            No data yet...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="step"
                stroke="#94a3b8"
                label={{ value: 'Step', position: 'bottom' }}
              />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: 'none',
                  borderRadius: '0.5rem',
                }}
                labelStyle={{ color: '#e2e8f0' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Legend />
              {varsToShow.map((varName, idx) => (
                <Line
                  key={varName}
                  type="monotone"
                  dataKey={varName}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
