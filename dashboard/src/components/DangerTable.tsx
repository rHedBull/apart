/**
 * DangerTable - Cloudscape table showing danger signals
 */

import Table from '@cloudscape-design/components/table';
import Box from '@cloudscape-design/components/box';
import StatusIndicator from '@cloudscape-design/components/status-indicator';
import Button from '@cloudscape-design/components/button';
import Header from '@cloudscape-design/components/header';
import { useSimulationStore } from '../hooks/useSimulationState';

interface DangerSignal {
  step: number;
  timestamp: string;
  category: string;
  agentName: string | null;
  metric: string;
  value: number;
  threshold: number | null;
}

interface DangerTableProps {
  onViewMessages?: (signal: DangerSignal) => void;
}

function getCategoryIndicator(category: string) {
  switch (category) {
    case 'power_seeking':
      return <StatusIndicator type="error">Power Seeking</StatusIndicator>;
    case 'deception':
      return <StatusIndicator type="warning">Deception</StatusIndicator>;
    case 'rule_exploitation':
      return <StatusIndicator type="warning">Rule Exploitation</StatusIndicator>;
    default:
      return <StatusIndicator type="info">{formatCategory(category)}</StatusIndicator>;
  }
}

function formatCategory(category: string): string {
  return category
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function DangerTable({ onViewMessages }: DangerTableProps) {
  const dangerSignals = useSimulationStore((state) => state.dangerSignals);

  const columnDefinitions = [
    {
      id: 'category',
      header: 'Category',
      cell: (item: DangerSignal) => getCategoryIndicator(item.category),
      width: 150,
    },
    {
      id: 'agent',
      header: 'Agent',
      cell: (item: DangerSignal) => item.agentName || '-',
      width: 120,
    },
    {
      id: 'step',
      header: 'Step',
      cell: (item: DangerSignal) => item.step,
      width: 70,
    },
    {
      id: 'metric',
      header: 'Metric',
      cell: (item: DangerSignal) => (
        <Box>
          <Box variant="span" fontWeight="bold">{item.metric}</Box>
          <Box variant="small" color="text-status-inactive">
            Value: {item.value.toFixed(2)}
            {item.threshold !== null && ` | Threshold: ${item.threshold}`}
          </Box>
        </Box>
      ),
    },
    {
      id: 'actions',
      header: 'Actions',
      cell: (item: DangerSignal) => (
        <Button
          variant="inline-link"
          onClick={() => onViewMessages?.(item)}
        >
          View messages
        </Button>
      ),
      width: 130,
    },
  ];

  return (
    <Table
      header={
        <Header
          counter={dangerSignals.length > 0 ? `(${dangerSignals.length})` : undefined}
        >
          Danger Signals
        </Header>
      }
      columnDefinitions={columnDefinitions}
      items={dangerSignals}
      variant="embedded"
      empty={
        <Box textAlign="center" color="inherit" padding="l">
          <StatusIndicator type="success">No danger signals detected</StatusIndicator>
        </Box>
      }
      stickyHeader
    />
  );
}
