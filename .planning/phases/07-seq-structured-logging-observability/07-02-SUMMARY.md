---
phase: 07-seq-structured-logging-observability
plan: "02"
subsystem: infra
tags: [structlog, seqlog, logging, middleware, contextvars, fastapi, uvicorn]

requires:
  - phase: 07-01
    provides: Seq compose service, SEQ_SERVER_URL/SEQ_API_KEY/ENVIRONMENT config vars, structlog + seqlog 0.4.3 installed

provides:
  - configure_logging() entry point wiring structlog processor chain, stdout JSONRenderer, optional Seq handler
  - LoggingMiddleware generating per-request UUID request_id and binding structlog contextvars
  - main.py wired: configure_logging() in lifespan + LoggingMiddleware outermost in middleware stack

affects:
  - 07-03 (log emission — can now call structlog.get_logger() anywhere and get structured JSON with request_id)

tech-stack:
  added: []
  patterns:
    - "Shared processor chain: merge_contextvars → add_log_level → add_logger_name → TimeStamper(iso/utc) → StackInfoRenderer → UnicodeDecoder"
    - "ProcessorFormatter bridge: foreign_pre_chain=shared_processors, processors=[remove_processors_meta, JSONRenderer()]"
    - "Non-fatal Seq attachment: try/except wraps SeqLogHandler construction, failure logs warning and continues"
    - "Per-request contextvars: clear_contextvars() first in dispatch(), then bind environment+request_id+method+path"
    - "Uvicorn log silencing: handlers.clear() + propagate=True on uvicorn, uvicorn.access, uvicorn.error"

key-files:
  created:
    - app/src/logging_config.py
    - app/src/middleware/__init__.py
    - app/src/middleware/logging.py
  modified:
    - app/src/main.py

key-decisions:
  - "SeqLogHandler constructor path used: seqlog.SeqLogHandler(server_url, api_key, batch_size=10, auto_flush_timeout=2) — confirmed api_key is a valid param in 0.4.3"
  - "LoggingMiddleware registered before CORSMiddleware via add_middleware() so it is outermost in Starlette execution order"
  - "configure_logging() called as FIRST line of lifespan before init_db and model loading — ensures all startup logs are structured"
  - "environment contextvar pre-bound in configure_logging() and re-bound in LoggingMiddleware.dispatch() after clear_contextvars()"

patterns-established:
  - "All structured logging goes through configure_logging() — single entry point, no other logging.basicConfig calls"
  - "Every log call during a request carries request_id, method, path, environment automatically via contextvars"

duration: 2min
completed: 2026-04-17
---

# Phase 7 Plan 02: Structured Logging Backbone Summary

**structlog JSON pipeline with per-request UUID contextvars: configure_logging() processor chain, LoggingMiddleware, and main.py wired for structured stdout + optional Seq ingestion**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-17T15:28:14Z
- **Completed:** 2026-04-17T15:29:25Z
- **Tasks:** 3
- **Files modified:** 4 (3 created, 1 modified)

## Accomplishments
- `configure_logging()` builds shared processor chain with JSONRenderer on stdout and conditionally attaches SeqLogHandler (non-fatal on failure)
- `LoggingMiddleware` generates UUID `request_id` per request, binds `environment`, `method`, `path` into structlog contextvars — every downstream log call inherits these automatically
- `main.py` wired: `configure_logging()` is first call in lifespan startup, `LoggingMiddleware` registered as outermost middleware (before CORSMiddleware), `logging.basicConfig` removed

## Task Commits

1. **Task 1: Create logging_config.configure_logging()** - `8c5e645` (feat)
2. **Task 2: Create middleware package with LoggingMiddleware** - `b5fd8a9` (feat)
3. **Task 3: Wire configure_logging() and LoggingMiddleware into main.py** - `369bddd` (feat)

## Files Created/Modified
- `app/src/logging_config.py` — `configure_logging()`: shared processor chain, ProcessorFormatter/JSONRenderer, optional SeqLogHandler, uvicorn handler silencing, environment contextvar pre-bind
- `app/src/middleware/__init__.py` — package marker
- `app/src/middleware/logging.py` — `LoggingMiddleware(BaseHTTPMiddleware)`: clear then bind contextvars per request
- `app/src/main.py` — removed `logging.basicConfig`, added imports, `configure_logging()` in lifespan, `LoggingMiddleware` registered before `CORSMiddleware`

## Decisions Made

- **SeqLogHandler path confirmed:** `seqlog.SeqLogHandler(server_url, api_key=..., batch_size=10, auto_flush_timeout=2)` works in 0.4.3. The `api_key` parameter exists in the constructor. No fallback to `seqlog.log_to_seq()` was needed.
- **Processor chain composition:** `merge_contextvars → add_log_level → add_logger_name → TimeStamper(fmt="iso", utc=True) → StackInfoRenderer → UnicodeDecoder`. `ProcessorFormatter` uses `[remove_processors_meta, JSONRenderer()]` as its terminal processors with `foreign_pre_chain=shared_processors` for stdlib log bridging.
- **Middleware registration order:** `app.user_middleware` list shows `['CORSMiddleware', 'LoggingMiddleware']` — Starlette stores in reverse registration order. `LoggingMiddleware` was registered first, making it the outermost (first to execute). Confirmed at runtime.
- **Re-binding environment after clear:** `LoggingMiddleware.dispatch()` calls `clear_contextvars()` first (clears ALL contextvars including the environment bound in `configure_logging()`), then re-binds `environment` along with request-scoped values.
- **Uvicorn silencing:** Three-logger pattern (`uvicorn`, `uvicorn.access`, `uvicorn.error`) — `handlers.clear()` + `propagate=True` — no additional tweaks needed. Uvicorn log lines route through root logger and render as JSON via ProcessorFormatter bridge.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required beyond what was set up in Plan 01.

## Next Phase Readiness

- Logging backbone live: any `structlog.get_logger().info(...)` call now emits JSON with `request_id`, `method`, `path`, `environment`, `level`, `timestamp`
- Plan 03 (log emission) can immediately start adding structured log calls to inference, classification, and monitoring routes
- Seq ingestion will activate automatically when `SEQ_SERVER_URL` is set in the running environment

---
*Phase: 07-seq-structured-logging-observability*
*Completed: 2026-04-17*
