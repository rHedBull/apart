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

## Issue 5: LOW - Stop Simulation Doesn't Actually Stop

**Status:** Not fixed

**Problem:** `stop_simulation()` endpoint only marks status as STOPPED in the
registry. It doesn't actually signal the running simulation process to stop.

**Location:** `routes/simulations.py:330-355`

**Current behavior:**
```python
complete_simulation(run_id, SimulationStatus.STOPPED)
# But the actual simulation keeps running!
```

**Recommendation:** Implement proper cancellation via:
1. Cancellation tokens
2. Process signals
3. Shared state flags

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

1. **High:** Fix Issue 2 (conflicting status sources) - architectural
2. **Medium:** Fix Issue 3 (dead code) - code hygiene
3. **Medium:** Fix Issue 6 (database sync) - if using database mode
4. **Low:** Fix Issue 4 (persistence) - will be fixed by Issue 2
5. **Low:** Fix Issue 5 (stop simulation) - feature enhancement
