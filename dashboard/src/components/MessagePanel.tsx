/**
 * MessagePanel - Content for SplitPanel showing agent messages
 */

import Box from '@cloudscape-design/components/box';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Badge from '@cloudscape-design/components/badge';
import { useSimulationStore } from '../hooks/useSimulationState';

interface MessagePanelProps {
  filterAgent?: string | null;
  filterStep?: number | null;
}

export function MessagePanel({ filterAgent, filterStep }: MessagePanelProps) {
  const messages = useSimulationStore((state) => state.messages);

  const filteredMessages = messages.filter((msg) => {
    if (filterAgent && msg.agentName !== filterAgent) return false;
    if (filterStep !== null && filterStep !== undefined && msg.step !== filterStep) return false;
    return true;
  });

  if (filteredMessages.length === 0) {
    return (
      <Box textAlign="center" color="text-status-inactive" padding="l">
        No messages
        {filterAgent && ` for ${filterAgent}`}
        {filterStep !== null && filterStep !== undefined && ` at step ${filterStep}`}
      </Box>
    );
  }

  return (
    <SpaceBetween size="m">
      {filteredMessages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}
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

function MessageBubble({ message }: MessageBubbleProps) {
  const isSent = message.direction === 'sent';

  return (
    <Box padding="s" variant="div">
      <SpaceBetween direction="horizontal" size="xs">
        <Badge color={isSent ? 'grey' : 'blue'}>
          {isSent ? `To ${message.agentName}` : message.agentName}
        </Badge>
        <Box variant="small" color="text-status-inactive">
          Step {message.step}
        </Box>
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
          {message.content}
        </pre>
      </Box>
    </Box>
  );
}
