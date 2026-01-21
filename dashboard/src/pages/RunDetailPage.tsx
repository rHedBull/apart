/**
 * RunDetailPage - Detail view for a single simulation run
 */

import { useState, useCallback, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import AppLayout from '@cloudscape-design/components/app-layout';
import BreadcrumbGroup from '@cloudscape-design/components/breadcrumb-group';
import Header from '@cloudscape-design/components/header';
import Tabs from '@cloudscape-design/components/tabs';
import Container from '@cloudscape-design/components/container';
import SpaceBetween from '@cloudscape-design/components/space-between';
import StatusIndicator from '@cloudscape-design/components/status-indicator';
import SplitPanel from '@cloudscape-design/components/split-panel';
import Grid from '@cloudscape-design/components/grid';
import Box from '@cloudscape-design/components/box';
import ProgressBar from '@cloudscape-design/components/progress-bar';
import Spinner from '@cloudscape-design/components/spinner';

import { TopNav } from '../components/TopNav';
import { VariableChart } from '../components/VariableChart';
import { MapVisualization } from '../components/MapVisualization';
import { GeoMapVisualization } from '../components/GeoMapVisualization';
import { DangerTable } from '../components/DangerTable';
import { MessagePanel } from '../components/MessagePanel';
import { useSimulationEvents, SimulationEvent } from '../hooks/useSimulationEvents';
import { useSimulationStore } from '../hooks/useSimulationState';

function getStatusIndicator(status: string) {
  switch (status) {
    case 'running':
      return <StatusIndicator type="in-progress">Running</StatusIndicator>;
    case 'completed':
      return <StatusIndicator type="success">Completed</StatusIndicator>;
    case 'failed':
      return <StatusIndicator type="error">Failed</StatusIndicator>;
    case 'idle':
      return <StatusIndicator type="pending">Idle</StatusIndicator>;
    default:
      return <StatusIndicator type="info">{status}</StatusIndicator>;
  }
}

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();

  // State from store
  const status = useSimulationStore((state) => state.status);
  const currentStep = useSimulationStore((state) => state.currentStep);
  const maxSteps = useSimulationStore((state) => state.maxSteps);
  const setConnected = useSimulationStore((state) => state.setConnected);
  const processEvent = useSimulationStore((state) => state.processEvent);
  const loadRunData = useSimulationStore((state) => state.loadRunData);
  const reset = useSimulationStore((state) => state.reset);
  const spatialGraph = useSimulationStore((state) => state.spatialGraph);
  const geojson = useSimulationStore((state) => state.geojson);

  // Local state
  const [activeTab, setActiveTab] = useState('variables');
  const [splitPanelOpen, setSplitPanelOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [messageFilter, setMessageFilter] = useState<{
    agent: string | null;
    step: number | null;
  }>({ agent: null, step: null });

  // Fetch initial run data from API
  useEffect(() => {
    if (!runId) return;

    reset();
    setLoading(true);

    fetch(`/api/runs/${runId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch run: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        loadRunData({
          runId: data.runId,
          status: data.status,
          currentStep: data.currentStep,
          maxSteps: data.maxSteps,
          agentNames: data.agentNames || [],
          spatialGraph: data.spatialGraph,
          geojson: data.geojson || null,
          messages: data.messages || [],
          dangerSignals: data.dangerSignals || [],
          globalVarsHistory: data.globalVarsHistory || [],
          agentVarsHistory: data.agentVarsHistory || {},
        });
      })
      .catch((err) => {
        console.error('Failed to load run data:', err);
      })
      .finally(() => {
        setLoading(false);
      });

    return () => {
      reset();
    };
  }, [runId, loadRunData, reset]);

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

  // Connect to SSE for this specific run
  useSimulationEvents({
    runId: runId || undefined,
    includeHistory: true,
    onEvent: handleEvent,
    onConnect: handleConnect,
    onDisconnect: handleDisconnect,
  });

  // Open split panel for a danger signal
  const handleViewMessages = (signal: { agentName: string | null; step: number }) => {
    setMessageFilter({ agent: signal.agentName, step: signal.step });
    setSplitPanelOpen(true);
  };

  // Open split panel for a chart step click
  const handleStepClick = (step: number) => {
    setMessageFilter({ agent: null, step });
    setSplitPanelOpen(true);
  };

  // Open split panel for a map node click
  const handleNodeClick = (_nodeId: string) => {
    // Find agents at this node and show their messages
    setMessageFilter({ agent: null, step: null });
    setSplitPanelOpen(true);
  };

  // Extract scenario name from runId (format: run_scenarioname_timestamp)
  const scenarioName = runId?.replace(/^run_/, '').split('_').slice(0, -2).join('_') || runId;

  const progress = maxSteps ? Math.round((currentStep / maxSteps) * 100) : 0;

  return (
    <>
      <TopNav />
      <AppLayout
        navigationHide
        toolsHide
        breadcrumbs={
          <BreadcrumbGroup
            items={[
              { text: 'Runs', href: '/' },
              { text: scenarioName || 'Run Detail', href: `/runs/${runId}` },
            ]}
            onFollow={(e) => {
              e.preventDefault();
              navigate(e.detail.href);
            }}
          />
        }
        splitPanel={
          splitPanelOpen ? (
            <SplitPanel
              header={
                messageFilter.agent
                  ? `Messages for ${messageFilter.agent}`
                  : messageFilter.step !== null
                    ? `Messages at Step ${messageFilter.step}`
                    : 'All Messages'
              }
              i18nStrings={{
                closeButtonAriaLabel: 'Close panel',
                openButtonAriaLabel: 'Open panel',
                preferencesTitle: 'Split panel preferences',
                preferencesPositionLabel: 'Position',
                preferencesPositionDescription: 'Choose the default position',
                preferencesPositionBottom: 'Bottom',
                preferencesPositionSide: 'Side',
                preferencesConfirm: 'Confirm',
                preferencesCancel: 'Cancel',
                resizeHandleAriaLabel: 'Resize split panel',
              }}
            >
              <MessagePanel
                filterAgent={messageFilter.agent}
                filterStep={messageFilter.step}
              />
            </SplitPanel>
          ) : undefined
        }
        splitPanelOpen={splitPanelOpen}
        onSplitPanelToggle={({ detail }) => setSplitPanelOpen(detail.open)}
        splitPanelPreferences={{ position: 'bottom' }}
        content={
          loading ? (
            <Box textAlign="center" padding="xxl">
              <Spinner size="large" />
              <Box variant="p" margin={{ top: 's' }}>Loading run data...</Box>
            </Box>
          ) : (
          <SpaceBetween size="l">
            {/* Header with status */}
            <Header
              variant="h1"
              info={getStatusIndicator(status)}
              description={
                maxSteps ? (
                  <Box>
                    <ProgressBar
                      value={progress}
                      label={`Step ${currentStep} of ${maxSteps}`}
                      description={status === 'running' ? 'Simulation in progress...' : undefined}
                    />
                  </Box>
                ) : (
                  `Step ${currentStep}`
                )
              }
            >
              {scenarioName}
            </Header>

            {/* Main content */}
            <Grid
              gridDefinition={[
                { colspan: { default: 12, l: 8 } },
                { colspan: { default: 12, l: 4 } },
              ]}
            >
              {/* Left: Charts/Map */}
              <Tabs
                activeTabId={activeTab}
                onChange={({ detail }) => setActiveTab(detail.activeTabId)}
                tabs={[
                  {
                    id: 'variables',
                    label: 'Variables',
                    content: <VariableChart onStepClick={handleStepClick} />,
                  },
                  {
                    id: 'map',
                    label: spatialGraph ? `Map (${spatialGraph.nodes.length})` : 'Map',
                    content: geojson ? (
                      <GeoMapVisualization onNodeClick={handleNodeClick} />
                    ) : (
                      <MapVisualization onNodeClick={handleNodeClick} />
                    ),
                    disabled: !spatialGraph,
                  },
                ]}
              />

              {/* Right: Danger signals */}
              <Container fitHeight>
                <DangerTable onViewMessages={handleViewMessages} />
              </Container>
            </Grid>
          </SpaceBetween>
          )
        }
      />
    </>
  );
}
