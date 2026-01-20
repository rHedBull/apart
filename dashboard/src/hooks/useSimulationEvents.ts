/**
 * SSE hook for real-time simulation events
 */

import { useEffect, useCallback, useRef } from 'react';

export interface SimulationEvent {
  event_type: string;
  timestamp: string;
  run_id: string;
  step: number | null;
  data: Record<string, unknown>;
}

export type EventHandler = (event: SimulationEvent) => void;

interface UseSimulationEventsOptions {
  runId?: string;
  includeHistory?: boolean;
  onEvent?: EventHandler;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

/**
 * Hook to subscribe to simulation events via SSE
 */
export function useSimulationEvents(options: UseSimulationEventsOptions = {}) {
  const {
    runId,
    includeHistory = false,
    onEvent,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Build URL
    const baseUrl = runId
      ? `/api/events/stream/${runId}`
      : '/api/events/stream';
    const url = includeHistory ? `${baseUrl}?history=true` : baseUrl;

    // Create new EventSource
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    // Handle connection opened
    eventSource.onopen = () => {
      onConnect?.();
    };

    // Handle messages
    eventSource.onmessage = (e) => {
      try {
        const event: SimulationEvent = JSON.parse(e.data);
        onEvent?.(event);
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    };

    // Handle specific event types
    const eventTypes = [
      'connected',
      'simulation_started',
      'simulation_completed',
      'step_started',
      'step_completed',
      'agent_message_sent',
      'agent_response_received',
      'state_updated',
      'danger_signal',
    ];

    eventTypes.forEach((type) => {
      eventSource.addEventListener(type, (e: MessageEvent) => {
        try {
          const event: SimulationEvent = JSON.parse(e.data);
          onEvent?.(event);
        } catch (err) {
          console.error(`Failed to parse ${type} event:`, err);
        }
      });
    });

    // Handle errors
    eventSource.onerror = (e) => {
      onError?.(e);
      onDisconnect?.();

      // Attempt to reconnect after 5 seconds
      if (reconnectTimeoutRef.current === null) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          reconnectTimeoutRef.current = null;
          connect();
        }, 5000);
      }
    };

    return eventSource;
  }, [runId, includeHistory, onEvent, onConnect, onDisconnect, onError]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    const eventSource = connect();

    return () => {
      eventSource.close();
      if (reconnectTimeoutRef.current !== null) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [connect]);

  // Return methods to manually control connection
  return {
    reconnect: connect,
    disconnect: () => {
      eventSourceRef.current?.close();
      onDisconnect?.();
    },
  };
}
