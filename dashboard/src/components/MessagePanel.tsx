/**
 * MessagePanel - Content for SplitPanel showing agent messages
 */

import { useState } from 'react';
import Box from '@cloudscape-design/components/box';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Badge from '@cloudscape-design/components/badge';
import Select from '@cloudscape-design/components/select';
import Button from '@cloudscape-design/components/button';
import FormField from '@cloudscape-design/components/form-field';
import Spinner from '@cloudscape-design/components/spinner';
import { useSimulationStore } from '../hooks/useSimulationState';

interface MessagePanelProps {
  filterAgent?: string | null;
  filterStep?: number | null;
  onFilterChange?: (agent: string | null, step: number | null) => void;
}

export function MessagePanel({ filterAgent, filterStep, onFilterChange }: MessagePanelProps) {
  const messages = useSimulationStore((state) => state.messages);
  const agentNames = useSimulationStore((state) => state.agentNames);
  const maxSteps = useSimulationStore((state) => state.maxSteps);
  const pendingAgents = useSimulationStore((state) => state.pendingAgents);
  const status = useSimulationStore((state) => state.status);

  // Build agent options
  const agentOptions = [
    { value: '', label: 'All Agents' },
    ...agentNames.map((name) => ({ value: name, label: name })),
  ];

  // Build step options
  const stepOptions = [
    { value: '', label: 'All Steps' },
    ...Array.from({ length: maxSteps || 0 }, (_, i) => ({
      value: String(i + 1),
      label: `Step ${i + 1}`,
    })),
  ];

  const filteredMessages = messages.filter((msg) => {
    if (filterAgent && msg.agentName !== filterAgent) return false;
    if (filterStep !== null && filterStep !== undefined && msg.step !== filterStep) return false;
    return true;
  });

  const handleAgentChange = (value: string) => {
    onFilterChange?.(value || null, filterStep ?? null);
  };

  const handleStepChange = (value: string) => {
    onFilterChange?.(filterAgent ?? null, value ? parseInt(value, 10) : null);
  };

  const handleClearFilters = () => {
    onFilterChange?.(null, null);
  };

  // Build filter summary
  const getFilterSummary = () => {
    const parts: string[] = [];
    if (filterAgent) parts.push(filterAgent);
    if (filterStep !== null && filterStep !== undefined) parts.push(`Step ${filterStep}`);
    if (parts.length === 0) return `Showing all ${filteredMessages.length} messages`;
    return `Showing: ${parts.join(' at ')} · ${filteredMessages.length} messages`;
  };

  return (
    <SpaceBetween size="m">
      {/* Filter controls */}
      <SpaceBetween direction="horizontal" size="s">
        <FormField label="Agent">
          <Select
            selectedOption={agentOptions.find((o) => o.value === (filterAgent || '')) || agentOptions[0]}
            onChange={({ detail }) => handleAgentChange(detail.selectedOption.value || '')}
            options={agentOptions}
          />
        </FormField>
        <FormField label="Step">
          <Select
            selectedOption={stepOptions.find((o) => o.value === String(filterStep ?? '')) || stepOptions[0]}
            onChange={({ detail }) => handleStepChange(detail.selectedOption.value || '')}
            options={stepOptions}
          />
        </FormField>
        <Box margin={{ top: 'l' }}>
          <Button onClick={handleClearFilters} variant="link">
            Clear filters
          </Button>
        </Box>
      </SpaceBetween>

      {/* Filter summary */}
      <Box variant="small" color="text-status-inactive">
        {getFilterSummary()}
      </Box>

      {/* Thinking indicators for live runs */}
      {status === 'running' && pendingAgents.length > 0 && (
        <Box padding="s" color="text-status-info">
          <SpaceBetween direction="horizontal" size="xs">
            <Spinner size="normal" />
            <span>
              {pendingAgents.length === 1
                ? `${pendingAgents[0]} is thinking...`
                : `${pendingAgents.join(', ')} are thinking...`}
            </span>
          </SpaceBetween>
        </Box>
      )}

      {/* Messages */}
      {filteredMessages.length === 0 ? (
        <Box textAlign="center" color="text-status-inactive" padding="l">
          No messages
          {filterAgent && ` for ${filterAgent}`}
          {filterStep !== null && filterStep !== undefined && ` at step ${filterStep}`}
        </Box>
      ) : (
        filteredMessages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))
      )}
    </SpaceBetween>
  );
}

interface MessageBubbleProps {
  message: {
    step: number;
    timestamp: string;
    agentName: string;
    direction: 'sent' | 'received';
    content: string;
  };
}

const TRUNCATE_LENGTH = 150;

function MessageBubble({ message }: MessageBubbleProps) {
  const [expanded, setExpanded] = useState(false);
  const isSent = message.direction === 'sent';
  const isLong = message.content.length > TRUNCATE_LENGTH;

  const truncatedContent = isLong && !expanded
    ? message.content.slice(0, TRUNCATE_LENGTH) + '...'
    : message.content;

  return (
    <Box padding="s" variant="div">
      <SpaceBetween direction="horizontal" size="xs">
        <Badge color={isSent ? 'grey' : 'blue'}>
          {isSent ? `→ To ${message.agentName}` : `← ${message.agentName}`}
        </Badge>
        <Box variant="small" color="text-status-inactive">
          Step {message.step}
        </Box>
        {isLong && (
          <Button variant="inline-link" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show less' : 'Show full'}
          </Button>
        )}
      </SpaceBetween>
      <Box
        variant="p"
        margin={{ top: 'xs' }}
        color={isSent ? 'text-body-secondary' : 'text-status-info'}
      >
        <pre style={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          margin: 0,
          fontFamily: 'inherit',
          fontSize: 'inherit'
        }}>
          {truncatedContent}
        </pre>
      </Box>
    </Box>
  );
}
