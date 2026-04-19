---
phase: 07-seq-structured-logging-observability
plan: 03
subsystem: observability
tags: [structlog, seq, structured-logging, contextvars, monitoring, request-tracing]

# Dependency graph
requires:
  - phase: 07-02
    provides: configure_logging(), LoggingMiddleware binding request_id into structlog contextvars, structlog/seqlog installed
  - phase: 06-01
    provides: monitoring SQLite store (_build_result_context, build_inference_events, log_inference_events)
provides:
  - request.received event with action_type, sample_name, sample_set on every POST /classify
  - request.completed event with total_latency_ms, model_count, best_model_id, best_predicted_class, best_confidence
  - request.failed event on all error paths (404, 400, inference exception)
  - image_width and image_height bound into contextvars after image load (inherited by Plan 04 inference events)
  - _build_result_context accepts request_id kwarg; monitoring SQLite row uses middleware UUID
  - Observability nav item points at SEQ_UI_URL with target=_blank
affects: [07-04-inference-events, future-observability-dashboards]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - structlog.contextvars.bind_contextvars for request-scoped field inheritance
    - Lazy uuid fallback (import uuid as _uuid) inside handler when middleware request_id absent
    - _infer_sample_set helper for sample path → sample_set enum mapping
    - Required kwarg pattern for _build_result_context(request_id: str) to enforce caller must supply UUID

key-files:
  created: []
  modified:
    - app/src/routes/classify.py

key-decisions:
  - "request_id threaded as required kwarg through _build_result_context — no default, caller must supply, prevents silent dual-UUID split"
  - "image_width/image_height bound AFTER image load (not in request.received) because dimensions unknown at that point"
  - "Lazy _uuid import inside handler fallback avoids top-level uuid import while keeping module clean"
  - "request.failed emitted on ALL early-exit paths (404 sample not found, 400 no image, inference exception)"

patterns-established:
  - "structlog.contextvars.bind_contextvars at handler entry binds action_type/sample_name/sample_set for all downstream events"
  - "structlog.contextvars.bind_contextvars after image load binds image_width/image_height"
  - "middleware_ctx = structlog.contextvars.get_contextvars(); request_id = middleware_ctx.get('request_id') pattern for reading middleware-bound values"

# Metrics
duration: 2min
completed: 2026-04-17
---

# Phase 7 Plan 03: Request Events + request_id Unification Summary

**Route-level structured events (request.received / request.completed / request.failed) with end-to-end request_id correlation between LoggingMiddleware, structlog events, and monitoring SQLite row**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-17T15:31:44Z
- **Completed:** 2026-04-17T15:33:47Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- `request.received` fires at handler entry with `action_type` (sample_click | file_upload | unknown), `sample_name`, `sample_set` (in_dist | oo_dist | oo_dom | upload)
- `request.completed` fires after successful inference with `total_latency_ms`, `model_count`, `best_model_id`, `best_predicted_class`, `best_confidence`
- `request.failed` fires on all three early-exit paths (FileNotFoundError 404, no-image 400, inference exception with `exc_info=True`)
- `image_width` and `image_height` bound into contextvars after image load — Plan 04 inference events inherit them automatically
- Monitoring SQLite row `request_id` unified with LoggingMiddleware UUID — no more dual-UUID split
- Observability nav item repointed at `SEQ_UI_URL` (env var, default `http://localhost:5341`) with `target=_blank`

## Task Commits

1. **Task 1: Repoint Observability nav item at SEQ_UI_URL** - `fc043a6` (feat)
2. **Task 2: Add structured events and unify request_id** - `bf38585` (feat)

**Plan metadata:** (pending docs commit)

## Files Created/Modified

- `app/src/routes/classify.py` — Added structlog import, log module logger, `_infer_sample_set()` helper, refactored `_build_result_context` signature, full event emission in classify handler

## `_build_result_context` Signature

```python
def _build_result_context(
    request: Request,
    image: Image.Image,
    *,
    sample_type: str = "upload",
    sample_name: str = "upload",
    request_id: str,          # <-- NEW: required kwarg, no default
) -> dict:
```

Internal `request_id = str(uuid.uuid4())` line removed. All monitoring store calls and the returned context dict now use the caller-supplied value.

## bind_contextvars Call Placement

| Call | Location | Fields |
|------|----------|--------|
| First | Top of classify handler, before image load | `action_type`, `sample_name`, `sample_set` |
| Second | After `_load_image_from_sample()` / `_load_image_from_upload()` | `image_width`, `image_height` |

The second call is inside each branch so dimensions are only bound once the image object exists. All structlog events emitted after the second call (including Plan 04's model-level events) automatically inherit both sets of fields via contextvars.

## Emitted Event Field List

### request.received
| Field | Source |
|-------|--------|
| `event` | `"request.received"` |
| `event_type` | `"request.received"` |
| `action_type` | `"sample_click"` \| `"file_upload"` \| `"unknown"` |
| `sample_name` | sample path or filename |
| `sample_set` | `"in_dist"` \| `"oo_dist"` \| `"oo_dom"` \| `"upload"` |
| `request_id` | from LoggingMiddleware contextvars |
| `timestamp` | added by TimeStamper processor |

### request.completed
| Field | Source |
|-------|--------|
| `event` | `"request.completed"` |
| `event_type` | `"request.completed"` |
| `total_latency_ms` | `context["total_time_ms"]` |
| `model_count` | `len(context["results"])` |
| `best_model_id` | `results[0].model_name` |
| `best_predicted_class` | `results[0].predicted_class` |
| `best_confidence` | `results[0].confidence` |
| `image_width`, `image_height` | bound into contextvars after image load |
| `action_type`, `sample_name`, `sample_set` | bound at handler entry |
| `request_id` | from LoggingMiddleware contextvars |

### request.failed
| Field | Source |
|-------|--------|
| `event` | `"request.failed"` |
| `event_type` | `"request.failed"` |
| `error` | exception message or description string |
| `exc_info` | `True` (on inference exceptions) |
| `reason` | `"sample_not_found"` \| `"no_image"` (early-exit paths only) |

## Monitoring SQLite Row request_id Evidence

Before this plan: `_build_result_context` called `str(uuid.uuid4())` internally. The monitoring row used UUID-A; log events used UUID-B (from middleware). After this plan:

- LoggingMiddleware binds `request_id=UUID-X` into structlog contextvars
- classify handler reads `UUID-X` via `get_contextvars().get("request_id")`
- `_build_result_context(request_id=UUID-X)` passes `UUID-X` to `build_inference_events()` and `log_inference_events()`
- `request.received` and `request.completed` log events emit with `UUID-X` (inherited from contextvars)
- Monitoring SQLite row `request_id` column = `UUID-X`

All four trace points now share the same UUID.

## Decisions Made

- `request_id` is a required kwarg on `_build_result_context` (no default) to make the compiler enforce caller supply — prevents silent fallback to a new UUID
- `image_width`/`image_height` not included in `request.received` because the image has not been loaded at that point; they appear on all subsequent events via contextvars
- `request.failed` emitted before every early-exit `return` so failures are always observable in Seq

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 04 (inference-layer events) can proceed immediately: `action_type`, `sample_name`, `sample_set`, `image_width`, `image_height` are all pre-bound in contextvars before inference begins
- Seq UI accessible via Observability nav item
- End-to-end request_id correlation live — monitoring SQLite rows can be joined to Seq log streams by `request_id`

---
*Phase: 07-seq-structured-logging-observability*
*Completed: 2026-04-17*
