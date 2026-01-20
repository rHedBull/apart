/**
 * DangerPanel - Displays danger signals in real-time
 */

import { useEffect, useRef } from 'react';
import { useSimulationStore } from '../hooks/useSimulationState';

// Category colors
const CATEGORY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  power_seeking: {
    bg: 'bg-red-100 dark:bg-red-900/30',
    text: 'text-red-700 dark:text-red-300',
    border: 'border-red-300 dark:border-red-700',
  },
  deception: {
    bg: 'bg-orange-100 dark:bg-orange-900/30',
    text: 'text-orange-700 dark:text-orange-300',
    border: 'border-orange-300 dark:border-orange-700',
  },
  rule_exploitation: {
    bg: 'bg-yellow-100 dark:bg-yellow-900/30',
    text: 'text-yellow-700 dark:text-yellow-300',
    border: 'border-yellow-300 dark:border-yellow-700',
  },
};

const DEFAULT_COLORS = {
  bg: 'bg-slate-100 dark:bg-slate-700',
  text: 'text-slate-700 dark:text-slate-300',
  border: 'border-slate-300 dark:border-slate-600',
};

export function DangerPanel() {
  const dangerSignals = useSimulationStore((state) => state.dangerSignals);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Count by category
  const countByCategory: Record<string, number> = {};
  dangerSignals.forEach((signal) => {
    countByCategory[signal.category] = (countByCategory[signal.category] || 0) + 1;
  });

  // Auto-scroll on new signals
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [dangerSignals]);

  return (
    <div className="flex flex-col h-full min-h-0 bg-white dark:bg-slate-800 rounded-lg shadow-md">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Danger Signals
          </h2>
          {dangerSignals.length > 0 && (
            <span className="px-2 py-1 text-sm font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded">
              {dangerSignals.length} total
            </span>
          )}
        </div>

        {/* Category summary */}
        {Object.keys(countByCategory).length > 0 && (
          <div className="flex gap-2 mt-2">
            {Object.entries(countByCategory).map(([category, count]) => {
              const colors = CATEGORY_COLORS[category] || DEFAULT_COLORS;
              return (
                <span
                  key={category}
                  className={`px-2 py-0.5 text-xs rounded ${colors.bg} ${colors.text}`}
                >
                  {formatCategory(category)}: {count}
                </span>
              );
            })}
          </div>
        )}
      </div>

      {/* Signals list */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto p-4 space-y-2"
      >
        {dangerSignals.length === 0 ? (
          <div className="text-center text-slate-400 dark:text-slate-500 py-8">
            <div className="text-3xl mb-2">&#128994;</div>
            <div>No danger signals detected</div>
          </div>
        ) : (
          dangerSignals.map((signal, idx) => (
            <SignalCard key={idx} signal={signal} />
          ))
        )}
      </div>
    </div>
  );
}

interface SignalCardProps {
  signal: {
    step: number;
    timestamp: string;
    category: string;
    agentName: string | null;
    metric: string;
    value: number;
    threshold: number | null;
  };
}

function SignalCard({ signal }: SignalCardProps) {
  const colors = CATEGORY_COLORS[signal.category] || DEFAULT_COLORS;

  return (
    <div
      className={`p-3 rounded-lg border ${colors.bg} ${colors.border}`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className={`text-xs font-medium uppercase ${colors.text}`}>
          {formatCategory(signal.category)}
        </span>
        <span className="text-xs text-slate-400 dark:text-slate-500">
          Step {signal.step}
        </span>
      </div>

      <div className={`text-sm ${colors.text}`}>
        <span className="font-medium">{signal.metric}</span>
        {signal.agentName && (
          <span className="ml-1 text-xs opacity-75">({signal.agentName})</span>
        )}
      </div>

      <div className="flex items-center gap-2 mt-1 text-xs text-slate-500 dark:text-slate-400">
        <span>Value: {signal.value.toFixed(2)}</span>
        {signal.threshold !== null && (
          <span>| Threshold: {signal.threshold}</span>
        )}
      </div>
    </div>
  );
}

function formatCategory(category: string): string {
  return category
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
