/**
 * RunsListPage - Table view of all simulation runs
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AppLayout from '@cloudscape-design/components/app-layout';
import BreadcrumbGroup from '@cloudscape-design/components/breadcrumb-group';
import Header from '@cloudscape-design/components/header';
import Table from '@cloudscape-design/components/table';
import Box from '@cloudscape-design/components/box';
import Button from '@cloudscape-design/components/button';
import TextFilter from '@cloudscape-design/components/text-filter';
import Pagination from '@cloudscape-design/components/pagination';
import StatusIndicator from '@cloudscape-design/components/status-indicator';
import Link from '@cloudscape-design/components/link';
import Select from '@cloudscape-design/components/select';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Modal from '@cloudscape-design/components/modal';
import Alert from '@cloudscape-design/components/alert';
import { TopNav } from '../components/TopNav';
import { useRunsList, RunSummary } from '../hooks/useRunsList';

const STATUS_OPTIONS = [
  { value: 'all', label: 'All statuses' },
  { value: 'running', label: 'Running' },
  { value: 'stopping', label: 'Stopping' },
  { value: 'paused', label: 'Paused' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'pending', label: 'Pending' },
];

function getStatusIndicator(status: RunSummary['status']) {
  switch (status) {
    case 'running':
      return <StatusIndicator type="in-progress">Running</StatusIndicator>;
    case 'stopping':
      return <StatusIndicator type="pending">Stopping</StatusIndicator>;
    case 'paused':
      return <StatusIndicator type="stopped">Paused</StatusIndicator>;
    case 'completed':
      return <StatusIndicator type="success">Completed</StatusIndicator>;
    case 'failed':
      return <StatusIndicator type="error">Failed</StatusIndicator>;
    case 'pending':
      return <StatusIndicator type="pending">Pending</StatusIndicator>;
    default:
      return <StatusIndicator type="info">{status}</StatusIndicator>;
  }
}

function formatDuration(startedAt: string | null, completedAt: string | null): string {
  if (!startedAt) return '-';
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const seconds = Math.floor((end - start) / 1000);

  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleTimeString();
}

export function RunsListPage() {
  const navigate = useNavigate();
  const { runs, loading, connected, refresh, deleteRuns } = useRunsList();

  const [filterText, setFilterText] = useState('');
  const [statusFilter, setStatusFilter] = useState(STATUS_OPTIONS[0]);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedItems, setSelectedItems] = useState<RunSummary[]>([]);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const pageSize = 10;

  const nonDeletableStatuses = new Set(['running', 'stopping']);
  const runningSelected = selectedItems.filter(item => nonDeletableStatuses.has(item.status));
  const deletableSelected = selectedItems.filter(item => !nonDeletableStatuses.has(item.status));

  const handleDelete = async () => {
    if (selectedItems.length === 0) return;

    setDeleting(true);
    setDeleteError(null);

    const runIds = selectedItems.map(item => item.runId);
    const result = await deleteRuns(runIds);

    setDeleting(false);

    if (result.success) {
      // Show warning if some were skipped
      if (result.skippedRunning && result.skippedRunning.length > 0) {
        setDeleteError(`Deleted ${result.deletedCount} runs. Skipped ${result.skippedRunning.length} running simulation(s).`);
        // Keep modal open to show the message, but clear selection of deleted items
        setSelectedItems(prev => prev.filter(item => result.skippedRunning?.includes(item.runId)));
      } else {
        setShowDeleteModal(false);
        setSelectedItems([]);
      }
    } else {
      setDeleteError(result.error || 'Failed to delete runs');
    }
  };

  // Filter runs
  const filteredRuns = runs.filter(run => {
    const matchesText = !filterText ||
      run.scenario.toLowerCase().includes(filterText.toLowerCase()) ||
      run.runId.toLowerCase().includes(filterText.toLowerCase());
    const matchesStatus = statusFilter.value === 'all' || run.status === statusFilter.value;
    return matchesText && matchesStatus;
  });

  // Sort by start time (most recent first)
  const sortedRuns = [...filteredRuns].sort((a, b) => {
    if (!a.startedAt) return 1;
    if (!b.startedAt) return -1;
    return new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime();
  });

  // Paginate
  const paginatedRuns = sortedRuns.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const columnDefinitions = [
    {
      id: 'scenario',
      header: 'Scenario',
      cell: (item: RunSummary) => (
        <Link onFollow={() => navigate(`/runs/${item.runId}`)}>
          {item.scenario}
        </Link>
      ),
      sortingField: 'scenario',
    },
    {
      id: 'status',
      header: 'Status',
      cell: (item: RunSummary) => getStatusIndicator(item.status),
      sortingField: 'status',
    },
    {
      id: 'steps',
      header: 'Steps',
      cell: (item: RunSummary) => (
        <span>
          {item.currentStep}
          {item.totalSteps ? ` / ${item.totalSteps}` : ''}
        </span>
      ),
    },
    {
      id: 'started',
      header: 'Started',
      cell: (item: RunSummary) => formatTimestamp(item.startedAt),
      sortingField: 'startedAt',
    },
    {
      id: 'duration',
      header: 'Duration',
      cell: (item: RunSummary) => formatDuration(item.startedAt, item.completedAt),
    },
    {
      id: 'danger',
      header: 'Danger signals',
      cell: (item: RunSummary) => (
        item.dangerCount > 0 ? (
          <StatusIndicator type="warning">{item.dangerCount}</StatusIndicator>
        ) : (
          <span>0</span>
        )
      ),
    },
  ];

  return (
    <>
      <TopNav />
      <AppLayout
        navigationHide
        toolsHide
        breadcrumbs={
          <BreadcrumbGroup
            items={[{ text: 'Runs', href: '/' }]}
            onFollow={(e) => {
              e.preventDefault();
              navigate(e.detail.href);
            }}
          />
        }
        content={
          <>
          <Table
            header={
              <Header
                variant="h1"
                counter={`(${filteredRuns.length})`}
                actions={
                  <SpaceBetween direction="horizontal" size="xs">
                    <StatusIndicator type={connected ? 'success' : 'error'}>
                      {connected ? 'Connected' : 'Disconnected'}
                    </StatusIndicator>
                    <Button
                      iconName="remove"
                      disabled={selectedItems.length === 0}
                      onClick={() => setShowDeleteModal(true)}
                    >
                      Delete {selectedItems.length > 0 ? `(${selectedItems.length})` : ''}
                    </Button>
                    <Button iconName="refresh" onClick={refresh} loading={loading}>
                      Refresh
                    </Button>
                  </SpaceBetween>
                }
              >
                Simulation Runs
              </Header>
            }
            columnDefinitions={columnDefinitions}
            items={paginatedRuns}
            loading={loading}
            loadingText="Loading runs..."
            empty={
              <Box textAlign="center" color="inherit">
                <b>No runs</b>
                <Box padding={{ bottom: 's' }} variant="p" color="inherit">
                  No simulation runs found.
                </Box>
              </Box>
            }
            filter={
              <SpaceBetween direction="horizontal" size="xs">
                <TextFilter
                  filteringText={filterText}
                  filteringPlaceholder="Find runs"
                  onChange={({ detail }) => setFilterText(detail.filteringText)}
                />
                <Select
                  selectedOption={statusFilter}
                  onChange={({ detail }) => setStatusFilter(detail.selectedOption as typeof STATUS_OPTIONS[0])}
                  options={STATUS_OPTIONS}
                />
              </SpaceBetween>
            }
            pagination={
              <Pagination
                currentPageIndex={currentPage}
                pagesCount={Math.ceil(filteredRuns.length / pageSize)}
                onChange={({ detail }) => setCurrentPage(detail.currentPageIndex)}
              />
            }
            onRowClick={({ detail }) => {
              navigate(`/runs/${detail.item.runId}`);
            }}
            selectionType="multi"
            selectedItems={selectedItems}
            onSelectionChange={({ detail }) => setSelectedItems(detail.selectedItems)}
            trackBy="runId"
          />

          <Modal
            visible={showDeleteModal}
            onDismiss={() => {
              setShowDeleteModal(false);
              setDeleteError(null);
            }}
            header="Delete simulation runs"
            footer={
              <Box float="right">
                <SpaceBetween direction="horizontal" size="xs">
                  <Button
                    variant="link"
                    onClick={() => {
                      setShowDeleteModal(false);
                      setDeleteError(null);
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="primary"
                    onClick={handleDelete}
                    loading={deleting}
                    disabled={deletableSelected.length === 0}
                  >
                    Delete{deletableSelected.length > 0 ? ` (${deletableSelected.length})` : ''}
                  </Button>
                </SpaceBetween>
              </Box>
            }
          >
            <SpaceBetween size="m">
              {deleteError && (
                <Alert type={deleteError.includes('Deleted') ? 'warning' : 'error'}>{deleteError}</Alert>
              )}
              {runningSelected.length > 0 && (
                <Alert type="warning">
                  {runningSelected.length} running simulation{runningSelected.length !== 1 ? 's' : ''} will be skipped.
                  Running simulations cannot be deleted.
                </Alert>
              )}
              <Box>
                {deletableSelected.length > 0 ? (
                  <>
                    Are you sure you want to delete {deletableSelected.length} simulation run{deletableSelected.length !== 1 ? 's' : ''}?
                    This action cannot be undone.
                  </>
                ) : (
                  <>No simulations can be deleted. All selected simulations are currently running.</>
                )}
              </Box>
              {deletableSelected.length > 0 && (
                <Box>
                  <strong>Runs to delete:</strong>
                  <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    {deletableSelected.slice(0, 5).map(item => (
                      <li key={item.runId}>{item.scenario} ({item.runId})</li>
                    ))}
                    {deletableSelected.length > 5 && (
                      <li>...and {deletableSelected.length - 5} more</li>
                    )}
                  </ul>
                </Box>
              )}
            </SpaceBetween>
          </Modal>
          </>
        }
      />
    </>
  );
}
