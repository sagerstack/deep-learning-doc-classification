---
phase: 06-model-performance-monitoring
verified: 2026-04-14T00:00:00Z
status: gaps_found
score: 5/6 must-haves verified
gaps:
  - truth: "Labeled monitoring can be enabled later without changing the logging schema by adding target labels to the same event store"
    status: failed
    reason: "The target column is not pre-declared in the SQLite DDL. The docs describe backfilling via UPDATE statements and list target in the schema table, but ALTER TABLE would be required first. The must-have claims no schema change is needed."
    artifacts:
      - path: "app/src/monitoring/store.py"
        issue: "_CREATE_TABLE_SQL does not include a target TEXT column. The batch job (run_evidently.py) checks 'target' in model_df.columns dynamically, but this column will never appear unless added via ALTER TABLE first."
    missing:
      - "Add 'target TEXT' column to _CREATE_TABLE_SQL DDL in store.py so it is pre-declared on first init_db() call"
      - "Update _column_list() and _event_to_row() in store.py to include target=None at insert time (preserving existing rows via DEFAULT NULL)"
      - "Update InferenceEvent dataclass in schema.py to include target: Optional[str] = None if schema is to be set at event-creation time"
---

# Phase 6: Model Performance Monitoring Verification Report

**Phase Goal:** Add lightweight model performance monitoring for the FastAPI demo using Evidently, with one structured inference log row per model prediction, periodic per-model monitoring reports, and a sidebar route to the monitoring dashboard
**Verified:** 2026-04-14
**Status:** gaps_found — 1 gap
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every /classify request persists one durable monitoring row per model prediction with request_id, model_id, confidence, probabilities, latencies, and input metadata | VERIFIED | classify.py lines 115-136 build InferenceEvents via build_inference_events() and call log_inference_events(events, MONITORING_DB_PATH); all required fields present in schema.py InferenceEvent dataclass and store.py DDL |
| 2 | The web app sidebar includes a "Model Performance" entry that routes to /model-performance, which redirects to the configured Evidently dashboard URL | VERIFIED | base.html lines 209-213 have the sidebar link; routes/monitoring.py implements RedirectResponse to EVIDENTLY_DASHBOARD_URL with fallback HTML; router registered in main.py line 83 |
| 3 | A monitoring batch job reads logged events, groups by model_id, and generates one Evidently report per model per batch window | VERIFIED | run_evidently.py build_reports_for_window() iterates current_df.groupby("model_id") and calls _run_model_reports() per model; DataSummaryPreset + DataDriftPreset used for unlabeled; ClassificationPreset for labeled |
| 4 | Unlabeled monitoring captures class distribution drift, confidence shifts, latency drift, and OCR/text-density availability rates per model | VERIFIED | _METADATA_NUMERICAL_COLS includes confidence, total_time_ms, feature_time_ms, graph_time_ms, model_time_ms; _METADATA_CATEGORICAL_COLS includes predicted_label, ocr_available, text_density_available; DataDriftPreset applied when reference available |
| 5 | Labeled monitoring can be enabled later without changing the logging schema by adding target labels to the same event store | FAILED | target column not pre-declared in _CREATE_TABLE_SQL DDL (store.py). Must-have requires no schema change; the actual DDL requires ALTER TABLE before any backfill UPDATE can succeed |
| 6 | Monitoring setup is documented and runnable locally through a script or scheduled job outside the FastAPI request path | VERIFIED | docs/monitoring.md covers step-by-step local setup; bootstrap_reference.py and run_evidently.py are fully implemented with --offline mode; Docker Compose monitoring profile documented |

**Score:** 5/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/src/monitoring/schema.py` | InferenceEvent dataclass + build_inference_events() | VERIFIED | 109 lines, fully substantive, exports InferenceEvent and build_inference_events; imported in classify.py and store.py |
| `app/src/monitoring/store.py` | SQLite persistence with log_inference_events() | VERIFIED | 212 lines, fully substantive; init_db, log_inference_events, query_events, fetch_events_as_dataframe all implemented; imported in classify.py and main.py |
| `app/src/routes/classify.py` | Event logging integrated into classification pipeline | VERIFIED | build_inference_events + log_inference_events called in _build_result_context() lines 115-136; wrapped in try/except to be non-fatal |
| `app/src/routes/monitoring.py` | /model-performance route with redirect | VERIFIED | 41 lines; RedirectResponse when EVIDENTLY_DASHBOARD_URL set, fallback HTML otherwise; registered in main.py |
| `app/src/templates/base.html` | Sidebar "Model Performance" entry | VERIFIED | Lines 209-213; anchor href="/model-performance" with Material icon and label |
| `scripts/monitoring/run_evidently.py` | Batch job with build_reports_for_window() | VERIFIED | 437 lines; full Evidently SDK integration with groupby model_id; offline + cloud modes; CLI entrypoint |
| `scripts/monitoring/bootstrap_reference.py` | Reference dataset bootstrap | VERIFIED | 252 lines; real-traffic export and synthetic generation modes; full CLI |
| `docs/monitoring.md` | Operator documentation | VERIFIED | 200 lines; covers event schema, local usage steps, cloud publishing, labeled backfill strategy, Docker Compose profile |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `classify.py` | `monitoring/store.py` | `log_inference_events(events, MONITORING_DB_PATH)` | WIRED | Called in _build_result_context(), line 134 |
| `classify.py` | `monitoring/schema.py` | `build_inference_events(...)` | WIRED | Called lines 119-133; all pipeline fields mapped |
| `main.py` | `monitoring/store.py` | `init_db(MONITORING_DB_PATH)` | WIRED | Called in lifespan startup, line 41 |
| `main.py` | `routes/monitoring.py` | `app.include_router(monitoring_router)` | WIRED | Line 83 |
| `routes/monitoring.py` | `config.py` | `EVIDENTLY_DASHBOARD_URL` | WIRED | Imported and used in conditional redirect |
| `run_evidently.py` | `monitoring/store.py` | `fetch_events_as_dataframe()` | WIRED | Called in build_reports_for_window() line 98 |
| `run_evidently.py` | `monitoring/schema.py` | `PROB_COLUMN_NAMES, RVL_CDIP_LABELS` | WIRED | Imported at top of file, used in _build_evidently_dataset() |

### Requirements Coverage

All 6 must-haves checked. 5 satisfied, 1 blocked.

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| Per-request inference row persistence | SATISFIED | — |
| Sidebar model-performance route | SATISFIED | — |
| Per-model batch reports | SATISFIED | — |
| Unlabeled drift monitoring signals | SATISFIED | — |
| Future labeled monitoring without schema changes | BLOCKED | target column absent from DDL |
| Local runnable documentation | SATISFIED | — |

### Anti-Patterns Found

None. No TODO/FIXME blockers, no empty handlers, no placeholder implementations in any monitoring artifact.

### Human Verification Required

None needed for automated gap assessment. The single gap (missing target column in DDL) is structurally verifiable.

### Gaps Summary

One gap found. The `target` column is documented in `docs/monitoring.md` as a supported schema field for labeled backfill, and the batch job (`run_evidently.py`) correctly handles it when present via `"target" in model_df.columns`. However, the column is not pre-declared in `_CREATE_TABLE_SQL` in `store.py`. An operator following the documented backfill procedure (`UPDATE inference_events SET target = ...`) would receive a SQLite error: `no such column: target`.

The fix is straightforward: add `target TEXT` to the DDL in `store.py`. Since `CREATE TABLE IF NOT EXISTS` is used, existing databases would still need an `ALTER TABLE` migration, but new deployments would get the column automatically. Alternatively, `init_db()` can be extended to check for and add the column via `ALTER TABLE IF NOT EXISTS` (SQLite does not support IF NOT EXISTS on ALTER TABLE, but a `PRAGMA table_info` check suffices).

---

_Verified: 2026-04-14_
_Verifier: Claude (gsd-verifier)_
