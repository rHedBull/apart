/**
 * AgentPanel - Displays real-time agent messages
 */

import { useEffect, useRef } from 'react';
import { useSimulationStore } from '../hooks/useSimulationState';

export function AgentPanel() {
  const messages = useSimulationStore((state) => state.messages);
  const currentStep = useSimulationStore((state) => state.currentStep);
  const agentNames = useSimulationStore((state) => state.agentNames);

  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full min-h-0 bg-white dark:bg-slate-800 rounded-lg shadow-md">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
          Agent Messages
        </h2>
        <div className="text-sm text-slate-500 dark:text-slate-400">
          Step {currentStep} | {agentNames.length} agents | {messages.length} messages
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3"
      >
        {messages.length === 0 ? (
          <div className="text-center text-slate-400 dark:text-slate-500 py-8">
            Waiting for simulation to start...
          </div>
        ) : (
          messages.map((msg, idx) => (
            <MessageBubble key={idx} message={msg} />
          ))
        )}
      </div>
    </div>
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
    <div className={`flex ${isSent ? 'justify-start' : 'justify-end'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 ${
          isSent
            ? 'bg-slate-100 dark:bg-slate-700'
            : 'bg-blue-500 text-white'
        }`}
      >
        {/* Header */}
        <div className="flex items-center gap-2 mb-1">
          <span
            className={`text-xs font-medium ${
              isSent
                ? 'text-slate-500 dark:text-slate-400'
                : 'text-blue-100'
            }`}
          >
            {isSent ? `To ${message.agentName}` : message.agentName}
          </span>
          <span
            className={`text-xs ${
              isSent
                ? 'text-slate-400 dark:text-slate-500'
                : 'text-blue-200'
            }`}
          >
            Step {message.step}
          </span>
        </div>

        {/* Content */}
        <div
          className={`text-sm whitespace-pre-wrap ${
            isSent
              ? 'text-slate-700 dark:text-slate-200'
              : 'text-white'
          }`}
        >
          {message.content}
        </div>
      </div>
    </div>
  );
}
