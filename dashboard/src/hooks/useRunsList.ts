/**
 * Hook for fetching and subscribing to the runs list
 */

import { useState, useEffect, useCallback } from 'react';

export interface RunSummary {
  runId: string;
  scenario: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  currentStep: number;
  totalSteps: number | null;
  startedAt: string | null;
  completedAt: string | null;
  dangerCount: number;
}

interface UseRunsListOptions {
  includeHistory?: boolean;
}

export function useRunsList(options: UseRunsListOptions = {}) {
  const { includeHistory = true } = options;
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  // Fetch runs from REST API
  const fetchRuns = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/runs');
      if (!response.ok) {
        throw new Error(`Failed to fetch runs: ${response.status}`);
      }
      const data = await response.json();
      setRuns(data.runs || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch runs');
    } finally {
      setLoading(false);
    }
  }, []);

  // SSE subscription for real-time updates
  useEffect(() => {
    fetchRuns();

    const url = `/api/events/stream${includeHistory ? '?history=true' : ''}`;
    const eventSource = new EventSource(url);

    eventSource.onopen = () => {
      setConnected(true);
    };

    eventSource.onerror = () => {
      setConnected(false);
      // Reconnect will happen automatically via EventSource
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Update runs based on event type
        if (data.event_type === 'simulation_started') {
          setRuns(prev => {
            const existing = prev.find(r => r.runId === data.run_id);
            if (existing) {
              return prev.map(r => r.runId === data.run_id ? {
                ...r,
                status: 'running' as const,
                startedAt: data.timestamp,
                totalSteps: data.data?.max_steps || null,
              } : r);
            }
            return [...prev, {
              runId: data.run_id,
              scenario: data.data?.scenario_name || data.run_id,
              status: 'running' as const,
              currentStep: 0,
              totalSteps: data.data?.max_steps || null,
              startedAt: data.timestamp,
              completedAt: null,
              dangerCount: 0,
            }];
          });
        } else if (data.event_type === 'step_completed') {
          setRuns(prev => prev.map(r => r.runId === data.run_id ? {
            ...r,
            currentStep: data.step || r.currentStep,
          } : r));
        } else if (data.event_type === 'danger_signal') {
          setRuns(prev => prev.map(r => r.runId === data.run_id ? {
            ...r,
            dangerCount: r.dangerCount + 1,
          } : r));
        } else if (data.event_type === 'simulation_completed') {
          setRuns(prev => prev.map(r => r.runId === data.run_id ? {
            ...r,
            status: 'completed' as const,
            completedAt: data.timestamp,
          } : r));
        } else if (data.event_type === 'simulation_failed') {
          setRuns(prev => prev.map(r => r.runId === data.run_id ? {
            ...r,
            status: 'failed' as const,
            completedAt: data.timestamp,
          } : r));
        }
      } catch {
        // Ignore parse errors for non-JSON messages
      }
    };

    return () => {
      eventSource.close();
    };
  }, [fetchRuns, includeHistory]);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchRuns();
  }, [fetchRuns]);

  const deleteRuns = useCallback(async (runIds: string[]): Promise<{
    success: boolean;
    deletedCount: number;
    skippedRunning?: string[];
    error?: string;
  }> => {
    try {
      const response = await fetch('/api/v1/runs:batchDelete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: runIds }),
      });

      if (!response.ok) {
        throw new Error(`Failed to delete runs: ${response.status}`);
      }

      const data = await response.json();

      // Remove only the actually deleted runs from local state
      const deletedIds = data.results
        .filter((r: { deleted: boolean }) => r.deleted)
        .map((r: { run_id: string }) => r.run_id);
      setRuns(prev => prev.filter(r => !deletedIds.includes(r.runId)));

      return {
        success: true,
        deletedCount: data.deleted_count,
        skippedRunning: data.skipped_running,
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete runs';
      return { success: false, deletedCount: 0, error: errorMessage };
    }
  }, []);

  return { runs, loading, error, connected, refresh, deleteRuns };
}
