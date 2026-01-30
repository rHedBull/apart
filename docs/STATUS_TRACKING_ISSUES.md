# Run Status Tracking - Issues Analysis

## Summary

Analysis of the run status tracking system revealed multiple architectural issues
beyond the stale cache bug that was fixed.

---

## Issue 1: FIXED ✅ - Stale EventBus Cache

**Status:** Fixed in commit `3f556d2`

**Problem:** EventBus only loaded events from persistence at initialization. Worker
processes writing to the shared JSONL file were invisible to the API server.

**Fix:** Added `_check_and_reload_if_stale()` that checks file modification time
and reloads when the file has been updated by external processes.

---

## Issue 2: FIXED ✅ - Multiple Conflicting Status Sources

**Status:** Fixed

**Problem:** There were THREE different sources of truth for run status:

1. **`_simulations` dict** (`routes/simulations.py:36`) - In-memory registry
2. **EventBus** (`event_bus.py`) - Event-based status derivation
3. **Database** (`database.py`) - SQLite persistence (when enabled)

**Fix:** Consolidated to EventBus as the single source of truth:
- Removed the `_simulations` dict entirely from `routes/simulations.py`
- Updated demo endpoint to use EventBus-only
- All API endpoints now derive status from EventBus events
- `/api/runs` and `/api/simulations` now return consistent status

---

## Issue 3: FIXED ✅ - Dead/Unused Route Code

**Status:** Fixed

**Problem:** `routes/simulations.py` contained dead code (routes not mounted,
`_simulations` dict not used by main app).

**Fix:**
- Removed all dead code from `routes/simulations.py`
- Updated `routes/events.py` demo to use EventBus directly
- Left placeholder file with documentation explaining the architecture

---

## Issue 4: FIXED ✅ - No Persistence for `_simulations` Registry

**Status:** Fixed (by Issue 2 fix)

**Problem:** The `_simulations` dict was purely in-memory.

**Fix:** Removed `_simulations` dict entirely. All status is now derived from
EventBus, which is persisted to JSONL file or SQLite database.

---

## Issue 5: N/A - Stop Simulation Doesn't Actually Stop

**Status:** Not applicable - endpoint was removed

**Problem:** `stop_simulation()` endpoint only marks status as STOPPED in the
registry. It doesn't actually signal the running simulation process to stop.

**Resolution:** The stop endpoint was removed along with dead routes cleanup.
No stop functionality currently exists. If needed in future, implement properly
with cancellation tokens or process signals.

---

## Issue 6: LOW - Database Mode Status Not Synced

**Status:** Partially addressed

**Problem:** When `APART_USE_DATABASE=1`, the database has a `status` column
that's set via `update_simulation_status()`, but this is NOT automatically
called when events are emitted.

**The flow should be:**
1. Worker emits `simulation_completed` event
2. Database status gets updated to 'completed'

**Current flow:**
1. Worker emits `simulation_completed` event
2. Database event is inserted
3. Database simulation status remains 'running'

**Location:** `database.py:186-209` has `update_simulation_status()` but it's
not called when events are emitted.

**Recommendation:** Add an event handler that updates database status when
completion/failure events are emitted.

---

## Issue 7: FIXED ✅ - Detail Page Shows "Idle" for Running Simulations

**Status:** Fixed in commit `f14dc68` on branch `fix/run-detail-status-idle`

**Problem:** The `/api/runs/{run_id}` endpoint required `state.json` on disk,
returning 404 for pending/running simulations that only had EventBus events.
The frontend caught the error silently and stayed at default "idle" status.

**Symptoms:**
- List page shows simulation as "running"
- Detail page shows same simulation as "idle"

**Fix:** The endpoint now checks EventBus first. If `state.json` doesn't exist
but events do, it builds the response from EventBus events. Only returns 404
when neither source has data.

---

## Issue 8: MEDIUM - Duplicate /api/runs and /api/simulations APIs

**Status:** Not fixed - needs consolidation

**Problem:** There are two parallel APIs with different data sources and behaviors:

| Endpoint | Data Source | Used By |
|----------|-------------|---------|
| `GET /api/runs` | Disk + EventBus | Dashboard |
| `GET /api/runs/{id}` | EventBus → Disk fallback | Dashboard |
| `DELETE /api/runs/{id}` | All sources | Dashboard |
| `GET /api/simulations` | EventBus only | Tests |
| `GET /api/simulations/{id}` | EventBus only | Tests |
| `POST /api/simulations` | Creates job | Tests, CLI |

**Issues:**
1. `/api/simulations` misses historical runs without EventBus data
2. `/api/simulations/{id}` returns 404 for historical runs (unlike `/api/runs/{id}`)
3. POST is on `/api/simulations` but DELETE is on `/api/runs`
4. Dashboard only uses `/api/runs`, tests mostly use `/api/simulations`

**Recommendation:** Consolidate to single `/api/runs` API:
1. Move `POST /api/simulations` → `POST /api/runs`
2. Deprecate/remove `/api/simulations` endpoints
3. Update all tests to use `/api/runs`
4. Single consistent data source (EventBus + disk fallback)

---

## Test Coverage Gaps

Current coverage for server module: **58%**

| File | Coverage | Notes |
|------|----------|-------|
| `routes/simulations.py` | 0% | Dead code, not mounted |
| `routes/events.py` | 0% | Dead code, not mounted |
| `job_queue.py` | 42% | Missing queue operation tests |
| `app.py` | 64% | Missing many endpoint tests |
| `event_bus.py` | 69% | Good after new tests |

---

## Recommended Priority

### Completed
- ✅ Issue 1 (stale cache)
- ✅ Issue 2 (conflicting status sources)
- ✅ Issue 3 (dead code)
- ✅ Issue 4 (persistence)
- ✅ Issue 7 (detail page idle bug)

### Outstanding
1. **Medium:** Issue 8 (API consolidation) - reduce confusion, single source of truth
2. **Low:** Issue 6 (database sync) - only matters if using database mode
3. **N/A:** Issue 5 (stop simulation) - endpoint removed, implement fresh if needed
