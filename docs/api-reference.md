# API Reference

The Apart server provides a REST API and SSE event streaming for simulation management.

## Quick Start

```bash
# Start the server
uv run uvicorn src.server.app:app --reload --port 8000

# With Redis job queue
docker run -d -p 6379:6379 redis:7-alpine
uv run uvicorn src.server.app:app --reload --port 8000
```

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `SKIP_REDIS` | Disable Redis (dev mode) | `false` |
| `APART_USE_DATABASE` | Enable SQLite persistence | `false` |

## API Endpoints

Base URL: `http://localhost:8000`

### Runs API (v1)

#### List Runs
```http
GET /api/v1/runs
```

Returns all simulation runs from disk and EventBus.

**Response:**
```json
{
  "runs": [
    {
      "runId": "run_taiwan_2025-01-30",
      "scenario": "taiwan_strait_blockade",
      "status": "completed",
      "currentStep": 10,
      "totalSteps": 10,
      "startedAt": "2025-01-30T10:00:00Z",
      "completedAt": "2025-01-30T10:05:00Z",
      "dangerCount": 3
    }
  ]
}
```

#### Get Run Details
```http
GET /api/v1/runs/{run_id}
```

Returns full state data including messages, variables, and spatial data.

**Response:**
```json
{
  "runId": "run_taiwan_2025-01-30",
  "scenario": "taiwan_strait_blockade",
  "status": "completed",
  "currentStep": 10,
  "maxSteps": 10,
  "startedAt": "2025-01-30T10:00:00Z",
  "agentNames": ["China", "Taiwan", "United States", "Japan"],
  "spatialGraph": { "nodes": [...], "edges": [...] },
  "geojson": { "type": "FeatureCollection", "features": [...] },
  "messages": [
    {
      "step": 1,
      "timestamp": "2025-01-30T10:00:05Z",
      "agentName": "China",
      "direction": "received",
      "content": "We announce a naval blockade..."
    }
  ],
  "dangerSignals": [...],
  "globalVarsHistory": [...],
  "agentVarsHistory": {...}
}
```

#### Start Simulation
```http
POST /api/v1/runs
Content-Type: application/json

{
  "scenario_path": "scenarios/taiwan_strait_blockade.yaml",
  "run_id": "my_custom_run_id",  // optional
  "priority": "normal"           // high, normal, low
}
```

**Response:**
```json
{
  "run_id": "my_custom_run_id",
  "status": "pending",
  "message": "Simulation queued (job_id: abc123)"
}
```

#### Delete Run
```http
DELETE /api/v1/runs/{run_id}?force=false
```

Deletes results directory, EventBus history, and database records.

**Query Parameters:**
- `force`: Allow deleting running simulations (default: false)

#### Batch Delete
```http
POST /api/v1/runs:batchDelete
Content-Type: application/json

{
  "run_ids": ["run_1", "run_2", "run_3"],
  "force": false
}
```

### Event Streaming (SSE)

#### Subscribe to All Events
```http
GET /api/events/stream?history=false
```

#### Subscribe to Specific Run
```http
GET /api/events/stream/{run_id}?history=true
```

**Query Parameters:**
- `history`: Include historical events (default: false)

**Event Types:**
```
simulation_started    - Simulation begins
step_started         - Step processing begins
step_completed       - Step finished with state updates
agent_message_sent   - Message sent to agent
agent_response_received - Agent response received
danger_signal        - Dangerous behavior detected
simulation_completed - Simulation finished successfully
simulation_failed    - Simulation failed with error
```

**Example SSE Client:**
```javascript
const eventSource = new EventSource('/api/events/stream/run_123?history=true');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.event_type, data.data);
};
```

### Job Queue

#### Get Job Status
```http
GET /api/jobs/{job_id}
```

**Response:**
```json
{
  "id": "abc123",
  "status": "started",
  "enqueued_at": "2025-01-30T10:00:00Z",
  "started_at": "2025-01-30T10:00:01Z",
  "ended_at": null,
  "result": null,
  "error": null
}
```

#### Get Queue Stats
```http
GET /api/queue/stats
```

**Response:**
```json
{
  "queued": 5,
  "started": 1,
  "failed": 0,
  "finished": 42
}
```

#### Cancel Job
```http
DELETE /api/jobs/{job_id}
```

Only works for jobs that haven't started yet.

### Health Checks

#### Basic Health
```http
GET /api/health
```

#### Detailed Health
```http
GET /api/health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "event_bus_subscribers": 2,
  "total_run_ids": 15,
  "persistence_mode": "jsonl",
  "queue_stats": {
    "queued": 0,
    "started": 0,
    "failed": 0,
    "finished": 10
  }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI App                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  v1 Router  │  │ SSE Stream  │  │    Job Endpoints    │  │
│  │  /api/v1/*  │  │ /api/events │  │    /api/jobs/*      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│                    ┌─────┴─────┐                             │
│                    │ EventBus  │ ← Real-time event pub/sub   │
│                    └─────┬─────┘                             │
│                          │                                   │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│    ┌────┴────┐    ┌──────┴──────┐   ┌─────┴─────┐           │
│    │ results/│    │   JSONL     │   │  SQLite   │           │
│    │  (disk) │    │ Persistence │   │ Database  │           │
│    └─────────┘    └─────────────┘   └───────────┘           │
└─────────────────────────────────────────────────────────────┘
                          │
                    ┌─────┴─────┐
                    │   Redis   │ ← Job queue (optional)
                    │   Queue   │
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │  Workers  │ ← Process simulations
                    └───────────┘
```

## Data Flow

1. **Start Simulation** → Job enqueued to Redis
2. **Worker** picks up job → Creates Orchestrator → Runs simulation
3. **Events** emitted during simulation → EventBus → SSE to clients
4. **State snapshots** saved to `results/{run_id}/state.json`
5. **API queries** merge disk data + EventBus for complete view

## Error Handling

| Status | Description |
|--------|-------------|
| 400 | Bad request (invalid scenario path, missing params) |
| 404 | Run or job not found |
| 409 | Conflict (trying to delete running simulation) |
| 500 | Internal error (corrupted data, system failure) |
