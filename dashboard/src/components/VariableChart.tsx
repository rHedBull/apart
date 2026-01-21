/**
 * VariableChart - Line chart for variable history with Cloudscape styling
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
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Button from '@cloudscape-design/components/button';
import { useSimulationStore } from '../hooks/useSimulationState';

// Colors for different variables
const COLORS = [
  '#0972d3', // Cloudscape blue
  '#1d8102', // Cloudscape green
  '#d91515', // Cloudscape red
  '#5f6b7a', // Cloudscape grey
  '#9469d6', // purple
  '#ec7211', // orange
  '#2ea597', // teal
  '#c33193', // pink
];

interface ChartData {
  step: number;
  [key: string]: number;
}

interface VariableChartProps {
  onStepClick?: (step: number) => void;
}

export function VariableChart({ onStepClick }: VariableChartProps) {
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

  const handleChartClick = (data: { activePayload?: Array<{ payload: { step: number } }> }) => {
    if (data?.activePayload?.[0]?.payload?.step !== undefined && onStepClick) {
      onStepClick(data.activePayload[0].payload.step);
    }
  };

  return (
    <Container
      header={
        <Header variant="h2">
          Variables Over Time
        </Header>
      }
      fitHeight
    >
      <SpaceBetween size="s">
        {/* Source selector */}
        <SpaceBetween direction="horizontal" size="xs">
          <Button
            variant={showGlobal ? 'primary' : 'normal'}
            onClick={() => {
              setShowGlobal(true);
              setSelectedAgent(null);
            }}
          >
            Global
          </Button>
          {agentNames.map((name) => (
            <Button
              key={name}
              variant={!showGlobal && selectedAgent === name ? 'primary' : 'normal'}
              onClick={() => {
                setShowGlobal(false);
                setSelectedAgent(name);
              }}
            >
              {name}
            </Button>
          ))}
        </SpaceBetween>

        {/* Variable toggles */}
        {availableVars.length > 0 && (
          <SpaceBetween direction="horizontal" size="xs">
            {availableVars.map((varName, idx) => (
              <Button
                key={varName}
                variant={selectedVars.size === 0 || selectedVars.has(varName) ? 'primary' : 'normal'}
                onClick={() => toggleVar(varName)}
                iconSvg={
                  <svg viewBox="0 0 16 16" width={16} height={16}>
                    <circle cx="8" cy="8" r="6" fill={COLORS[idx % COLORS.length]} />
                  </svg>
                }
              >
                {varName}
              </Button>
            ))}
          </SpaceBetween>
        )}

        {/* Chart */}
        <Box padding="s">
          {chartData.length === 0 ? (
            <Box textAlign="center" color="text-status-inactive" padding="xxl">
              No data yet...
            </Box>
          ) : (
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <LineChart data={chartData} onClick={handleChartClick}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e9ebed" />
                  <XAxis
                    dataKey="step"
                    stroke="#5f6b7a"
                    label={{ value: 'Step', position: 'bottom', offset: -5 }}
                  />
                  <YAxis stroke="#5f6b7a" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#16191f',
                      border: 'none',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: '#d1d5db' }}
                    itemStyle={{ color: '#d1d5db' }}
                    cursor={{ stroke: '#0972d3', strokeDasharray: '5 5' }}
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
                      activeDot={{ r: 6, cursor: 'pointer' }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Box>
      </SpaceBetween>
    </Container>
  );
}
