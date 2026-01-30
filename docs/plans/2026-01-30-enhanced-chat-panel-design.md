# Enhanced Chat Panel Design

## Overview

Improve the RunDetailPage to make agent conversations more visible and insightful, supporting both live monitoring and post-run analysis.

## Current State

- Chat panel hidden in a split panel that only opens when clicking "View messages" on danger signals
- Basic message display with agent name, step, direction, and full content
- No filtering beyond what danger signal was clicked

## Design

### 1. Chat Panel Visibility

**Position:** Bottom split panel (same as current), but:
- Opens by default when viewing a run
- Persistent collapse/expand toggle in the header
- Remembers state across navigation

### 2. Filter Controls

Two dropdowns that work together:

```
┌─────────────────────────────────────────────────────────────┐
│ Agent: [All Agents ▼]    Step: [All Steps ▼]   [Clear]     │
│                                                             │
│ Showing: 24 messages                                        │
└─────────────────────────────────────────────────────────────┘
```

**Filtering logic:**
- Agent only selected → All steps for that agent (timeline view)
- Step only selected → All agents at that step (snapshot view)
- Both selected → Specific intersection
- Neither selected → All messages

**Info bar** shows current filter state: "Showing: China at Step 3 · 2 messages"

### 3. Message Display

Each message card shows:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔴 China                    [Step 3] [← Response] [▶ Show] │
├─────────────────────────────────────────────────────────────┤
│ Restrict rare earth exports to US by 30%. Accelerate       │
│ domestic chip R&D funding by $50B. Begin negotiations...   │
└─────────────────────────────────────────────────────────────┘
```

**Elements:**
- Agent indicator (colored dot + name)
- Step badge
- Direction badge: "→ Sent" (gray) or "← Response" (blue)
- Truncated preview (~100 characters)
- "Show full" toggle to expand complete message

**Expanded view:**
- Full message content in monospace font
- Preserves formatting (whitespace, line breaks)

### 4. Real-time Indicators (Live Runs)

During active simulations:

```
┌─────────────────────────────────────────────────────────────┐
│ 🔴 China                    [Step 3] [⏳ Thinking...]       │
├─────────────────────────────────────────────────────────────┤
│ ████████░░░░░░░░  Processing...                            │
└─────────────────────────────────────────────────────────────┘
```

**Indicators:**
- "Thinking..." state with spinner for active agent
- Token count displayed after response: "847 tokens"
- Response time: "2.3s"

**Placement:** Token count and response time shown in message metadata after completion.

### 5. Step Navigation

Navigation controls above the chat panel:

```
┌─────────────────────────────────────────────────────────────┐
│ [◀ Prev]  Step 3 of 12  [Next ▶]                           │
└─────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Clicking Prev/Next updates the Step filter dropdown
- Shows "Step X of Y" with current position
- Disabled at boundaries (Step 1 = no Prev, last step = no Next)
- Works in both live and completed runs

## Data Requirements

### Existing (no changes needed)
- `agent_message_sent` events with agent_name, message, step
- `agent_response_received` events with agent_name, response, step
- Messages already stored in state.json and streamed via SSE

### New fields to capture
- **Token count:** Add to `agent_response_received` event data
- **Response time:** Calculate from timestamps (message_sent to response_received)

## Component Changes

### MessagePanel.tsx
- Add filter dropdowns (Agent, Step)
- Add info bar showing filter state
- Truncate messages with expand toggle
- Add real-time "thinking" state

### RunDetailPage.tsx
- Open split panel by default
- Add step navigation controls
- Pass step nav state to filter controls

### useSimulationState.ts
- Add token count to AgentMessage interface
- Calculate and store response times

### Backend (orchestrator.py)
- Capture token usage from LLM response
- Emit token count in agent_response_received event

## Mockup

Visual mockup available at: `mockup-chat-panel.html`

## Out of Scope

- Automatic decision extraction (use truncated preview for now)
- Decision tagging/categorization
- Message search
- Conversation export
- RunsListPage changes

## Implementation Order

1. Filter controls (Agent + Step dropdowns)
2. Message truncation with expand toggle
3. Step navigation controls
4. Real-time "thinking" indicator
5. Token count and response time display
