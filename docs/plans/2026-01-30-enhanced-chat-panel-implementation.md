# Enhanced Chat Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the RunDetailPage chat panel with filter controls, message truncation, step navigation, and real-time indicators.

**Architecture:** Enhance existing MessagePanel component with filter dropdowns and expand/collapse. Add StepNavigation component. Track "thinking" state for live runs via existing SSE events. Response times calculated from timestamps.

**Tech Stack:** React, TypeScript, Cloudscape Design System, Zustand

---

## Task 1: Add Filter Controls to MessagePanel

**Files:**
- Modify: `dashboard/src/components/MessagePanel.tsx`

**Step 1: Add imports for filter components**

Add at top of file after existing imports:

```typescript
import Select from '@cloudscape-design/components/select';
import Button from '@cloudscape-design/components/button';
import FormField from '@cloudscape-design/components/form-field';
```

**Step 2: Update MessagePanel to manage its own filter state**

Replace the entire MessagePanel component:

```typescript
interface MessagePanelProps {
  filterAgent?: string | null;
  filterStep?: number | null;
  onFilterChange?: (agent: string | null, step: number | null) => void;
}

export function MessagePanel({ filterAgent, filterStep, onFilterChange }: MessagePanelProps) {
  const messages = useSimulationStore((state) => state.messages);
  const agentNames = useSimulationStore((state) => state.agentNames);
  const maxSteps = useSimulationStore((state) => state.maxSteps);

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
            selectedOption={stepOptions.find((o) => o.value === String(filterStep || '')) || stepOptions[0]}
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
```

**Step 3: Verify it compiles**

Run: `cd dashboard && npm run build`
Expected: Build succeeds with no TypeScript errors

**Step 4: Commit**

```bash
git add dashboard/src/components/MessagePanel.tsx
git commit -m "feat(dashboard): add filter controls to MessagePanel"
```

---

## Task 2: Add Message Truncation with Expand Toggle

**Files:**
- Modify: `dashboard/src/components/MessagePanel.tsx`

**Step 1: Add useState import and ExpandableSection**

Update imports:

```typescript
import { useState } from 'react';
import ExpandableSection from '@cloudscape-design/components/expandable-section';
```

**Step 2: Update MessageBubble to truncate and expand**

Replace MessageBubble component:

```typescript
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
```

**Step 3: Verify it compiles**

Run: `cd dashboard && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add dashboard/src/components/MessagePanel.tsx
git commit -m "feat(dashboard): add message truncation with expand toggle"
```

---

## Task 3: Update RunDetailPage to Open Split Panel by Default

**Files:**
- Modify: `dashboard/src/pages/RunDetailPage.tsx`

**Step 1: Change default split panel state to true**

Find line 61:
```typescript
const [splitPanelOpen, setSplitPanelOpen] = useState(false);
```

Change to:
```typescript
const [splitPanelOpen, setSplitPanelOpen] = useState(true);
```

**Step 2: Update MessagePanel callback for filter changes**

Add handler after handleNodeClick:

```typescript
// Handle filter changes from MessagePanel
const handleFilterChange = (agent: string | null, step: number | null) => {
  setMessageFilter({ agent, step });
};
```

**Step 3: Pass onFilterChange to MessagePanel**

Update MessagePanel usage (around line 197):

```typescript
<MessagePanel
  filterAgent={messageFilter.agent}
  filterStep={messageFilter.step}
  onFilterChange={handleFilterChange}
/>
```

**Step 4: Verify it compiles**

Run: `cd dashboard && npm run build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add dashboard/src/pages/RunDetailPage.tsx
git commit -m "feat(dashboard): open chat panel by default, wire up filter callbacks"
```

---

## Task 4: Add Step Navigation Component

**Files:**
- Create: `dashboard/src/components/StepNavigation.tsx`
- Modify: `dashboard/src/pages/RunDetailPage.tsx`

**Step 1: Create StepNavigation component**

Create file `dashboard/src/components/StepNavigation.tsx`:

```typescript
/**
 * StepNavigation - Prev/Next controls for navigating simulation steps
 */

import Button from '@cloudscape-design/components/button';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Box from '@cloudscape-design/components/box';

interface StepNavigationProps {
  currentStep: number | null;
  maxSteps: number | null;
  onStepChange: (step: number | null) => void;
}

export function StepNavigation({ currentStep, maxSteps, onStepChange }: StepNavigationProps) {
  if (!maxSteps) return null;

  const canGoPrev = currentStep !== null && currentStep > 1;
  const canGoNext = currentStep !== null && currentStep < maxSteps;
  const isFiltered = currentStep !== null;

  const handlePrev = () => {
    if (currentStep !== null && currentStep > 1) {
      onStepChange(currentStep - 1);
    }
  };

  const handleNext = () => {
    if (currentStep !== null && currentStep < maxSteps) {
      onStepChange(currentStep + 1);
    } else if (currentStep === null) {
      onStepChange(1);
    }
  };

  const handleClear = () => {
    onStepChange(null);
  };

  return (
    <SpaceBetween direction="horizontal" size="xs" alignItems="center">
      <Button
        iconName="angle-left"
        variant="icon"
        disabled={!canGoPrev}
        onClick={handlePrev}
        ariaLabel="Previous step"
      />
      <Box variant="span" fontSize="body-s">
        {isFiltered ? `Step ${currentStep} of ${maxSteps}` : `All ${maxSteps} steps`}
      </Box>
      <Button
        iconName="angle-right"
        variant="icon"
        disabled={!canGoNext}
        onClick={handleNext}
        ariaLabel="Next step"
      />
      {isFiltered && (
        <Button variant="inline-link" onClick={handleClear}>
          Show all
        </Button>
      )}
    </SpaceBetween>
  );
}
```

**Step 2: Import and use in RunDetailPage**

Add import near other component imports:

```typescript
import { StepNavigation } from '../components/StepNavigation';
```

**Step 3: Add step navigation handler**

Add after handleFilterChange:

```typescript
// Handle step navigation
const handleStepNavigate = (step: number | null) => {
  setMessageFilter((prev) => ({ ...prev, step }));
  if (!splitPanelOpen) {
    setSplitPanelOpen(true);
  }
};
```

**Step 4: Add StepNavigation to the SplitPanel header area**

Update the SplitPanel header (around line 177-183):

```typescript
header={
  <SpaceBetween direction="horizontal" size="m" alignItems="center">
    <span>
      {messageFilter.agent
        ? `Messages for ${messageFilter.agent}`
        : messageFilter.step !== null
          ? `Messages at Step ${messageFilter.step}`
          : 'Agent Conversations'}
    </span>
    <StepNavigation
      currentStep={messageFilter.step}
      maxSteps={maxSteps}
      onStepChange={handleStepNavigate}
    />
  </SpaceBetween>
}
```

**Step 5: Add SpaceBetween import if needed**

Already imported in RunDetailPage.

**Step 6: Verify it compiles**

Run: `cd dashboard && npm run build`
Expected: Build succeeds

**Step 7: Commit**

```bash
git add dashboard/src/components/StepNavigation.tsx dashboard/src/pages/RunDetailPage.tsx
git commit -m "feat(dashboard): add step navigation for chat panel"
```

---

## Task 5: Add Real-time "Thinking" Indicator

**Files:**
- Modify: `dashboard/src/hooks/useSimulationState.ts`
- Modify: `dashboard/src/components/MessagePanel.tsx`

**Step 1: Add pendingAgents state to store**

In `useSimulationState.ts`, add to SimulationState interface (around line 86):

```typescript
// Pending agent responses (for "thinking" indicator)
pendingAgents: string[];
addPendingAgent: (name: string) => void;
removePendingAgent: (name: string) => void;
clearPendingAgents: () => void;
```

**Step 2: Add initial state and actions**

In the create() call, add to initial state (around line 149):

```typescript
pendingAgents: [],
```

Add actions after clearVariableHistory (around line 194):

```typescript
addPendingAgent: (name) =>
  set((state) => ({
    pendingAgents: state.pendingAgents.includes(name)
      ? state.pendingAgents
      : [...state.pendingAgents, name],
  })),

removePendingAgent: (name) =>
  set((state) => ({
    pendingAgents: state.pendingAgents.filter((n) => n !== name),
  })),

clearPendingAgents: () => set({ pendingAgents: [] }),
```

**Step 3: Update processEvent to track pending agents**

In processEvent, update the `agent_message_sent` case:

```typescript
case 'agent_message_sent':
  get().addPendingAgent(data.agent_name as string);
  get().addMessage({
    step: step || 0,
    timestamp,
    agentName: data.agent_name as string,
    direction: 'sent',
    content: data.message as string,
  });
  break;
```

Update the `agent_response_received` case:

```typescript
case 'agent_response_received':
  get().removePendingAgent(data.agent_name as string);
  get().addMessage({
    step: step || 0,
    timestamp,
    agentName: data.agent_name as string,
    direction: 'received',
    content: data.response as string,
  });
  break;
```

Update `simulation_started` case to clear pending:

```typescript
case 'simulation_started':
  set({
    status: 'running',
    currentStep: 0,
    maxSteps: data.max_steps as number,
    agentNames: (data.agent_names as string[]) || [],
    spatialGraph: (data.spatial_graph as SpatialGraph) || null,
    geojson: (data.geojson as GeoJSONData) || null,
    messages: [],
    dangerSignals: [],
    globalVarsHistory: [],
    agentVarsHistory: {},
    pendingAgents: [],
  });
  break;
```

Update reset to clear pendingAgents:

```typescript
reset: () =>
  set({
    connected: false,
    currentRunId: null,
    status: 'idle',
    currentStep: 0,
    maxSteps: null,
    agentNames: [],
    spatialGraph: null,
    geojson: null,
    messages: [],
    dangerSignals: [],
    globalVarsHistory: [],
    agentVarsHistory: {},
    pendingAgents: [],
  }),
```

**Step 4: Show thinking indicator in MessagePanel**

In MessagePanel, add to store selectors:

```typescript
const pendingAgents = useSimulationStore((state) => state.pendingAgents);
const status = useSimulationStore((state) => state.status);
```

Add Spinner import:

```typescript
import Spinner from '@cloudscape-design/components/spinner';
```

Add thinking indicators after filter summary, before messages:

```typescript
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
```

**Step 5: Verify it compiles**

Run: `cd dashboard && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add dashboard/src/hooks/useSimulationState.ts dashboard/src/components/MessagePanel.tsx
git commit -m "feat(dashboard): add real-time thinking indicator for agents"
```

---

## Task 6: Add Response Time Display

**Files:**
- Modify: `dashboard/src/hooks/useSimulationState.ts`
- Modify: `dashboard/src/components/MessagePanel.tsx`

**Step 1: Update AgentMessage interface to include responseTimeMs**

In `useSimulationState.ts`, update the interface:

```typescript
interface AgentMessage {
  step: number;
  timestamp: string;
  agentName: string;
  direction: 'sent' | 'received';
  content: string;
  responseTimeMs?: number;  // Only for received messages
}
```

**Step 2: Track sent timestamps for response time calculation**

Add to SimulationState interface:

```typescript
// Track sent timestamps for response time calculation
messageSentTimestamps: Record<string, string>;  // agentName -> timestamp
```

Add to initial state:

```typescript
messageSentTimestamps: {},
```

Update reset:

```typescript
messageSentTimestamps: {},
```

**Step 3: Update processEvent to calculate response time**

Update `agent_message_sent` case:

```typescript
case 'agent_message_sent':
  get().addPendingAgent(data.agent_name as string);
  set((state) => ({
    messageSentTimestamps: {
      ...state.messageSentTimestamps,
      [data.agent_name as string]: timestamp,
    },
  }));
  get().addMessage({
    step: step || 0,
    timestamp,
    agentName: data.agent_name as string,
    direction: 'sent',
    content: data.message as string,
  });
  break;
```

Update `agent_response_received` case:

```typescript
case 'agent_response_received': {
  const agentName = data.agent_name as string;
  const sentTimestamp = get().messageSentTimestamps[agentName];
  let responseTimeMs: number | undefined;

  if (sentTimestamp) {
    responseTimeMs = new Date(timestamp).getTime() - new Date(sentTimestamp).getTime();
  }

  get().removePendingAgent(agentName);
  get().addMessage({
    step: step || 0,
    timestamp,
    agentName,
    direction: 'received',
    content: data.response as string,
    responseTimeMs,
  });
  break;
}
```

**Step 4: Display response time in MessageBubble**

Update MessageBubbleProps interface:

```typescript
interface MessageBubbleProps {
  message: {
    step: number;
    timestamp: string;
    agentName: string;
    direction: 'sent' | 'received';
    content: string;
    responseTimeMs?: number;
  };
}
```

Update MessageBubble to show response time:

```typescript
function MessageBubble({ message }: MessageBubbleProps) {
  const [expanded, setExpanded] = useState(false);
  const isSent = message.direction === 'sent';
  const isLong = message.content.length > TRUNCATE_LENGTH;

  const truncatedContent = isLong && !expanded
    ? message.content.slice(0, TRUNCATE_LENGTH) + '...'
    : message.content;

  const formatResponseTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <Box padding="s" variant="div">
      <SpaceBetween direction="horizontal" size="xs">
        <Badge color={isSent ? 'grey' : 'blue'}>
          {isSent ? `→ To ${message.agentName}` : `← ${message.agentName}`}
        </Badge>
        <Box variant="small" color="text-status-inactive">
          Step {message.step}
        </Box>
        {message.responseTimeMs !== undefined && (
          <Box variant="small" color="text-status-inactive">
            {formatResponseTime(message.responseTimeMs)}
          </Box>
        )}
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
```

**Step 5: Verify it compiles**

Run: `cd dashboard && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add dashboard/src/hooks/useSimulationState.ts dashboard/src/components/MessagePanel.tsx
git commit -m "feat(dashboard): add response time display for agent messages"
```

---

## Task 7: Manual Testing

**Step 1: Start the dashboard dev server**

Run: `cd dashboard && npm run dev`

**Step 2: Start the backend server**

Run: `cd /home/hendrik/coding/ai-safety/apart && python -m src.server.app`

**Step 3: Run a simulation and verify features**

1. Navigate to http://localhost:5173
2. Start a simulation or view an existing run
3. Verify:
   - Chat panel opens by default
   - Agent and Step dropdowns work
   - Filter summary updates correctly
   - Messages truncate with "Show full" toggle
   - Step navigation (Prev/Next) works
   - During live runs: "thinking" indicator shows
   - Response times display on received messages

**Step 4: Final commit with any fixes**

```bash
git add -A
git commit -m "fix(dashboard): polish enhanced chat panel implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Filter controls (Agent + Step dropdowns) | MessagePanel.tsx |
| 2 | Message truncation with expand toggle | MessagePanel.tsx |
| 3 | Open split panel by default | RunDetailPage.tsx |
| 4 | Step navigation component | StepNavigation.tsx, RunDetailPage.tsx |
| 5 | Real-time "thinking" indicator | useSimulationState.ts, MessagePanel.tsx |
| 6 | Response time display | useSimulationState.ts, MessagePanel.tsx |
| 7 | Manual testing | - |
