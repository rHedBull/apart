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

## Issue 2: CRITICAL - Multiple Conflicting Status Sources

**Status:** Not fixed

**Problem:** There are THREE different sources of truth for run status:

1. **`_simulations` dict** (`routes/simulations.py:36`) - In-memory registry
2. **EventBus** (`event_bus.py`) - Event-based status derivation
3. **Database** (`database.py`) - SQLite persistence (when enabled)

These can easily get out of sync:
- RQ workers use EventBus but not `_simulations`
- In-process async uses both
- `/api/runs` uses EventBus + results/ directory
- `/api/simulations` in app.py uses EventBus
- `/api/simulations` in routes/simulations.py uses `_simulations` (but not mounted!)

**Impact:** Depending on which endpoint you query, you may get different status.

**Recommendation:** Consolidate to a single source of truth. Options:
1. Use EventBus as the only source (current direction)
2. Use Database as the only source (more robust)
3. Remove `_simulations` dict entirely

---

## Issue 3: MEDIUM - Dead/Unused Route Code

**Status:** Not fixed

**Problem:** `routes/simulations.py` and `routes/events.py` define routers that
are NOT mounted in the FastAPI app. The code exists but most of it is never executed.

**Files affected:**
- `src/server/routes/simulations.py` - 0% test coverage
- `src/server/routes/events.py` - 0% test coverage

**Only used part:** `routes/events.py:demo_simulation()` imports from
`routes/simulations.py` for the demo endpoint.

**Recommendation:** Either:
1. Mount these routers properly and remove duplicates from app.py
2. Delete dead code and keep only what's used

---

## Issue 4: MEDIUM - No Persistence for `_simulations` Registry

**Status:** Not fixed

**Problem:** The `_simulations` dict is purely in-memory. On server restart,
all registered simulations are lost (even though the events may still be in
the EventBus persistence file).

**Impact:** After server restart, simulations appear to not exist even if
their events are persisted.

**Recommendation:** Remove `_simulations` entirely, derive all status from
EventBus or Database.

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
