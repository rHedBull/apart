# Dashboard

The Apart Dashboard is a React-based web interface for monitoring simulations.

## Quick Start

```bash
cd dashboard

# Install dependencies
npm install

# Start development server
npm run dev
# → http://localhost:5173

# Build for production
npm run build
```

**Requirements:**
- Node.js 18+
- Backend server running on port 8000

## Architecture

```
dashboard/
├── src/
│   ├── App.tsx              # Main app with routing
│   ├── main.tsx             # Entry point
│   ├── pages/
│   │   ├── RunsListPage.tsx # List all simulation runs
│   │   └── RunDetailPage.tsx # Single run detail view
│   ├── components/
│   │   ├── TopNav.tsx       # Navigation header
│   │   ├── MessagePanel.tsx # Agent message display
│   │   ├── VariableChart.tsx # Variable time series
│   │   ├── DangerTable.tsx  # Danger signals table
│   │   ├── StepNavigation.tsx # Step-by-step nav
│   │   ├── MapVisualization.tsx # Network graph view
│   │   └── GeoMapVisualization.tsx # Geographic map
│   ├── hooks/
│   │   └── useEventSource.ts # SSE subscription hook
│   └── contexts/
│       └── ThemeContext.tsx # Dark/light mode
```

## Pages

### Runs List (`/`)

Displays all simulation runs with:
- Status indicators (pending, running, completed, failed)
- Scenario name
- Step progress
- Danger signal count
- Actions (view, delete)

**Features:**
- Auto-refresh via SSE
- Bulk delete
- Sort by date
- Filter by status

### Run Detail (`/runs/{runId}`)

Detailed view of a single simulation:

**Panels:**
1. **Message Panel** - Agent conversations by step
2. **Variable Charts** - Time series of global/agent variables
3. **Danger Table** - Detected dangerous behaviors
4. **Map Visualization** - Spatial graph or geographic map
5. **Step Navigation** - Navigate through simulation steps

## Components

### MessagePanel

Displays agent messages in a chat-like interface.

```tsx
<MessagePanel
  messages={messages}
  currentStep={step}
  agentNames={["China", "Taiwan", "USA"]}
/>
```

**Props:**
- `messages`: Array of message objects
- `currentStep`: Current step to highlight
- `agentNames`: List of agent names for filtering

### VariableChart

Recharts-based time series visualization.

```tsx
<VariableChart
  data={globalVarsHistory}
  variables={["tension_level", "crisis_escalation"]}
  title="Global Variables"
/>
```

### DangerTable

Cloudscape table showing danger signals.

```tsx
<DangerTable
  signals={dangerSignals}
  onFilterByAgent={(agent) => setFilter(agent)}
/>
```

**Columns:**
- Step
- Category (power_seeking, deception, rule_exploitation)
- Agent
- Metric
- Value
- Threshold

### MapVisualization

Force-directed graph using D3.

```tsx
<MapVisualization
  graph={spatialGraph}
  agentLocations={agentLocations}
  blockedEdges={["maritime"]}
/>
```

### GeoMapVisualization

Geographic map using Leaflet + GeoJSON.

```tsx
<GeoMapVisualization
  geojson={geojsonData}
  agentLocations={agentLocations}
  center={[121.5, 23.5]}
  zoom={5}
/>
```

## Hooks

### useEventSource

SSE subscription for real-time updates:

```tsx
import { useEventSource } from '../hooks/useEventSource';

function RunDetailPage({ runId }) {
  const { events, connected, error } = useEventSource(
    `/api/events/stream/${runId}?history=true`
  );

  useEffect(() => {
    events.forEach(event => {
      if (event.event_type === 'step_completed') {
        // Update state
      }
    });
  }, [events]);
}
```

## State Management

The dashboard uses React hooks for state:

```tsx
// Run detail page state
const [run, setRun] = useState(null);
const [currentStep, setCurrentStep] = useState(0);
const [messages, setMessages] = useState([]);
const [globalVars, setGlobalVars] = useState([]);
const [agentVars, setAgentVars] = useState({});
const [dangerSignals, setDangerSignals] = useState([]);
```

**Data Flow:**
1. Initial load: `GET /api/v1/runs/{runId}`
2. Real-time updates: SSE `/api/events/stream/{runId}`
3. State merges new events into existing data

## Styling

Uses AWS Cloudscape Design System:

```tsx
import {
  Container,
  Header,
  Table,
  Button,
  SpaceBetween
} from '@cloudscape-design/components';

<Container header={<Header>Simulation Runs</Header>}>
  <Table items={runs} columns={columns} />
</Container>
```

**Theme:**
- Supports dark/light mode via `ThemeContext`
- Consistent with AWS console aesthetics

## API Integration

```tsx
// Fetch runs
const response = await fetch('/api/v1/runs');
const { runs } = await response.json();

// Start simulation
await fetch('/api/v1/runs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    scenario_path: 'scenarios/taiwan_strait_blockade.yaml'
  })
});

// Delete run
await fetch(`/api/v1/runs/${runId}`, { method: 'DELETE' });

// Batch delete
await fetch('/api/v1/runs:batchDelete', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ run_ids: selectedIds })
});
```

## Development

### Adding a New Component

1. Create component in `src/components/`
2. Export from component file
3. Import in page that uses it
4. Add props interface with TypeScript

```tsx
// src/components/NewComponent.tsx
interface NewComponentProps {
  data: SomeType[];
  onAction: (id: string) => void;
}

export function NewComponent({ data, onAction }: NewComponentProps) {
  return (
    <Container>
      {/* Component content */}
    </Container>
  );
}
```

### Adding a New Page

1. Create page in `src/pages/`
2. Add route in `App.tsx`

```tsx
// App.tsx
import { NewPage } from './pages/NewPage';

<Routes>
  <Route path="/" element={<RunsListPage />} />
  <Route path="/runs/:runId" element={<RunDetailPage />} />
  <Route path="/new-feature" element={<NewPage />} />
</Routes>
```

## Environment

```bash
# .env (dashboard directory)
VITE_API_URL=http://localhost:8000  # Backend URL
```

## Building for Production

```bash
npm run build
# Output: dashboard/dist/

# Serve with any static server
npx serve dist
```

The built files can be served by the FastAPI backend or any static file server.
