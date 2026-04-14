---
plan: 06-01
phase: 06-model-performance-monitoring
status: complete
completed: 2026-04-14
commit: a6327fb
---

# Plan 06-01: Structured Inference Logging + Dashboard Route

## What Was Built

Added a SQLite-backed inference event store that captures one row per model prediction on every classify request. The classify route now generates a `request_id`, builds `InferenceEvent` objects (one per model result including per-class probabilities, timing breakdowns, and feature-availability flags), and persists them via `log_inference_events`. A `/model-performance` route and sidebar link provide the entrypoint to the Evidently dashboard when `EVIDENTLY_DASHBOARD_URL` is configured, or a clear fallback message when it is not.

## Deliverables

- `app/src/monitoring/__init__.py`: Package marker
- `app/src/monitoring/schema.py`: `InferenceEvent` dataclass with per-class probability column naming, `build_inference_events` builder
- `app/src/monitoring/store.py`: `init_db`, `log_inference_events`, `query_events` over stdlib `sqlite3` — no ORM
- `app/src/routes/monitoring.py`: `GET /model-performance` — redirect to Evidently or fallback HTML
- `app/src/routes/classify.py`: Request-id generation, monitoring event capture after every inference pipeline run
- `app/src/templates/base.html`: Sidebar "Model Performance" link at bottom of left panel
- `app/src/config.py`: `MONITORING_DB_PATH` and `EVIDENTLY_DASHBOARD_URL` config entries
- `.env.example`: Two new env vars documented with defaults and explanation
- `app/src/main.py`: `init_db` called during lifespan startup; `monitoring_router` registered
- `app/tests/test_monitoring.py`: 11 tests covering store CRUD, schema builder, route redirect/fallback
- `app/tests/test_routes.py`: Sidebar link assertion added

## Key Decisions

- `sqlite3` stdlib only — no ORM or additional dependency; sufficient for batch export to Evidently
- One row per model per request — enables per-model drift tracking independently, not just overall pipeline
- Monitoring failure is non-fatal (wrapped in try/except with warning log) — inference must not break if DB write fails
- `EVIDENTLY_DASHBOARD_URL` env var drives redirect target; empty string triggers fallback HTML — no hardcoded URLs
- `init_db` uses `CREATE TABLE IF NOT EXISTS` — re-running on existing DB is idempotent, no data loss
- `sample_type` column distinguishes "sample" (preloaded RVL-CDIP images) from "upload" (user file) for drift segmentation

## Issues Encountered

None — plan executed exactly as written.

## Test Results

```
21 passed, 7 deselected, 152 warnings in 1.46s
(non-slow tests: store unit, schema builder, route redirect/fallback, sidebar link)
```
