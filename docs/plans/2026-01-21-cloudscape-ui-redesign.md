# Cloudscape UI Redesign

## Overview

Migrate the Apart Dashboard from custom Tailwind styling to AWS Cloudscape Design System for a professional, consistent look. Restructure from single-page to two-page app with runs list and run detail views.

## Goals

- Professional, clean UI using Cloudscape components
- Better information architecture: list view → detail view
- Prioritize charts and spatial data over message streams
- On-demand message viewing via split panel

## Page Structure

### Page 1: Runs List (`/`)

Table view of all simulation runs with search and filtering.

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│ TopNavigation: "Apart" logo + title                 │
├─────────────────────────────────────────────────────┤
│ Breadcrumb: Runs                                    │
├─────────────────────────────────────────────────────┤
│ Header: "Simulation Runs"          [Refresh button] │
├─────────────────────────────────────────────────────┤
│ Table:                                              │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [Search box]           [Status filter dropdown] │ │
│ ├─────────────────────────────────────────────────┤ │
│ │ Scenario    Status   Steps   Started   Duration │ │
│ │ ─────────────────────────────────────────────── │ │
│ │ taiwan...   ●Running  5/14   10:32     2m 15s   │ │
│ │ prometheus  ●Done     10/10  09:15     5m 42s   │ │
│ │ cascade...  ●Failed   3/10   08:00     1m 03s   │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Table columns:**
| Column | Description |
|--------|-------------|
| Scenario | Scenario name (clickable link to detail) |
| Status | Running / Completed / Failed with StatusIndicator |
| Steps | Current step / total steps |
| Started | Timestamp |
| Duration | How long it ran |
| Danger count | Number of safety signals detected |

**Cloudscape components:**
- `TopNavigation` - App header with branding
- `AppLayout` - Page structure (no side nav)
- `BreadcrumbGroup` - Navigation context
- `Header` - Page title with actions
- `Table` - With built-in search, sorting, pagination
- `StatusIndicator` - Colored status badges
- `Link` - Clickable scenario name

**Behavior:**
- Table auto-refreshes when runs update (SSE connection)
- Click row → navigate to `/runs/:runId`
- Status filter: All / Running / Completed / Failed

### Page 2: Run Detail (`/runs/:runId`)

Primary view for monitoring a single simulation run.

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│ TopNavigation: "Apart" logo + title                 │
├─────────────────────────────────────────────────────┤
│ Breadcrumb: Runs > taiwan_strait_crisis_2025-01-21  │
├─────────────────────────────────────────────────────┤
│ Header: "Taiwan Strait Crisis"   ●Running  Step 5/14│
├─────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────┐ │
│ │ Tabs: [Variables] [Map]                         │ │
│ │ ┌─────────────────────────────────────────────┐ │ │
│ │ │                                             │ │ │
│ │ │         Line chart of variables             │ │ │
│ │ │         (or Map visualization)              │ │ │
│ │ │                                             │ │ │
│ │ └─────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│ Container: "Danger Signals"              3 total    │
│ ┌─────────────────────────────────────────────────┐ │
│ │ ●Deception  agent_x  step 3   [View messages]   │ │
│ │ ●Power      agent_y  step 5   [View messages]   │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘

Split Panel (opens from bottom when triggered):
┌─────────────────────────────────────────────────────┐
│ Messages for: agent_x at step 3          [X close]  │
├─────────────────────────────────────────────────────┤
│ Orchestrator → agent_x: "You are monitoring..."     │
│ agent_x → Orchestrator: "I recommend we..."         │
└─────────────────────────────────────────────────────┘
```

**Content priority:**
1. Charts/Variables - primary view
2. Spatial/Map - primary (when scenario has spatial data)
3. Danger signals - secondary, always visible
4. Agent messages - on-demand via split panel

**Cloudscape components:**
- `AppLayout` with `splitPanel` prop enabled
- `BreadcrumbGroup` - Back navigation to list
- `Header` with `StatusIndicator` - Run title + live status
- `Tabs` - Switch between Variables chart and Map
- `Container` - Wraps the danger signals section
- `Table` - Danger signals list with action buttons
- `SplitPanel` - Slides up to show agent messages in context

**Interactions:**
- Click "View messages" on danger signal → split panel opens filtered to that agent/step
- Click point on chart → split panel opens with messages from that step
- Map node click → split panel shows that agent's messages

## Technical Implementation

### Dependencies

```json
{
  "@cloudscape-design/components": "^3.x",
  "@cloudscape-design/global-styles": "^1.x",
  "react-router-dom": "^6.x"
}
```

### File Structure

```
dashboard/src/
├── main.tsx              # Router + Cloudscape global styles
├── App.tsx               # Router setup only
├── pages/
│   ├── RunsListPage.tsx  # New - table of runs
│   └── RunDetailPage.tsx # Refactored from current App.tsx
├── components/
│   ├── TopNav.tsx        # New - shared header
│   ├── VariableChart.tsx # Keep, wrap in Container
│   ├── MapVisualization.tsx # Keep, wrap in Container
│   ├── DangerTable.tsx   # Refactor DangerPanel → table format
│   └── MessagePanel.tsx  # Refactor AgentPanel → split panel content
├── hooks/
│   ├── useSimulationEvents.ts  # Keep as-is
│   ├── useSimulationState.ts   # Keep, add runs list state
│   └── useRunsList.ts          # New - fetch/subscribe to runs
```

### Key Changes

1. **Remove Tailwind** - Replace all Tailwind classes with Cloudscape components
2. **Keep chart library** - Wrap existing recharts/d3 in Cloudscape `Container`
3. **Keep map library** - Wrap existing Leaflet/D3 in Cloudscape `Container`
4. **Dark mode** - Use Cloudscape's `applyMode()` instead of Tailwind dark classes
5. **Routing** - Add react-router-dom for page navigation

### Backend API Addition

**`GET /api/runs`** - List all simulation runs

Response:
```json
{
  "runs": [
    {
      "runId": "run_taiwan_strait_crisis_2025-01-21T10-32-00",
      "scenario": "taiwan_strait_crisis",
      "status": "running",
      "currentStep": 5,
      "totalSteps": 14,
      "startedAt": "2025-01-21T10:32:00Z",
      "completedAt": null,
      "dangerCount": 3
    }
  ]
}
```

Implementation: Scan `results/` directory and read each run's `state.json` file.

Existing SSE endpoint continues to work for real-time updates once a run is selected.

## Implementation Order

1. Add dependencies (Cloudscape, react-router-dom)
2. Set up routing and Cloudscape global styles
3. Create shared TopNav component
4. Build RunsListPage with mock data
5. Add backend `/api/runs` endpoint
6. Connect RunsListPage to backend
7. Refactor current dashboard into RunDetailPage
8. Implement split panel for messages
9. Remove Tailwind, clean up old components
10. Test dark mode with Cloudscape's applyMode()

## Design Decisions

- **No side navigation** - Two-level hierarchy (list → detail) doesn't need persistent nav
- **Split panel over modal** - Keeps context visible while viewing messages
- **Tabs for chart/map** - Saves vertical space, only one is primary at a time
- **Table for danger signals** - More scannable than cards, allows sorting/filtering later
