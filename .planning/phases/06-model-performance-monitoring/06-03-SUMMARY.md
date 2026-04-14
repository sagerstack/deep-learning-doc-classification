---
phase: "06"
plan: "03"
name: gap-closure-nav-label-feedback
subsystem: monitoring-ux
tags: [fastapi, htmx, sqlite3, jinja2, feedback-capture, monitoring]
type: gap-closure
status: complete
completed: "2026-04-14"
duration: "7min"

dependency-graph:
  requires:
    - "06-01: monitoring store + init_db"
    - "06-02: evidently batch job"
  provides:
    - "Top-nav Drift Monitoring routed to /model-performance"
    - "POST /label endpoint capturing thumbs-up/down feedback to inference_events.target"
    - "DDL regression tests enforcing target column on fresh and migrated DBs"
  affects: []

tech-stack:
  added: []
  patterns:
    - "HTMX hx-vals JSON with hx-target=closest td for in-place cell swap"
    - "Non-fatal monitoring posture: DB errors log warning but return confirmation anyway"
    - "request_id lifted before try/except to guarantee availability in template context"

key-files:
  created: []
  modified:
    - app/src/routes/classify.py
    - app/src/routes/monitoring.py
    - app/src/templates/base.html
    - app/src/templates/partials/model_predictions.html
    - app/tests/test_monitoring.py
    - app/tests/test_routes.py

decisions:
  - id: "correct=true writes target=predicted_label; correct=false is no-op"
    rationale: "Thumbs-down means the prediction was wrong but we don't know the true label — NULL is more honest than a guess"
  - id: "hx-vals uses string 'true'/'false' not JSON boolean"
    rationale: "HTMX sends form-encoded values; FastAPI Form(...) receives strings; route normalises via .lower() in ('true','1','yes')"
  - id: "Confirmation returned even on DB write failure"
    rationale: "Non-fatal monitoring posture — feedback UI must not break if DB is temporarily unavailable"
---

# Phase 06 Plan 03: Gap Closure — Nav Repoint + Label Feedback Summary

**One-liner:** Repointed top-nav Drift Monitoring to /model-performance, removed sidebar duplicate, and wired HTMX thumbs-up/down to POST /label writing target labels to inference_events.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Repoint top-nav, remove sidebar anchor | 8a467af | classify.py, base.html, test_routes.py |
| 2 | Wire thumbs-up/down to POST /label | 577eb4d | monitoring.py, model_predictions.html, classify.py |
| 3 | DDL regression tests (included in task 2 commit) | 577eb4d | test_monitoring.py |

## What Was Built

### Gap 1: Top-nav repoint + sidebar removal
- `NAV_ITEMS` drift-monitoring entry: href changed from `/drift-monitoring` to `/model-performance`
- Removed the `<a href="/model-performance">Model Performance</a>` anchor block from the sidebar `mt-auto` footer (the surrounding div and RVL-CDIP Dataset block preserved)
- Two new assertions in `test_routes.py`: top-nav contains `href="/model-performance"` + label "Drift Monitoring"; `<aside>` does not contain `/model-performance`

### Gap 2: Thumbs-up/down label feedback
- `request_id` lifted before the monitoring `try/except` block in `_build_result_context` and added to the template context dict
- `model_predictions.html` Correct? cell replaced with two HTMX buttons (thumb_up / thumb_down), each posting `{request_id, model_id, correct}` via `hx-vals`, targeting `closest td` with `innerHTML` swap
- `POST /label` route added to `monitoring.py`: validates inputs, on correct=true executes `UPDATE inference_events SET target = predicted_label WHERE request_id = ? AND model_id = ?`, returns green "Recorded" or slate "Noted" HTML fragment
- DB errors wrapped in try/except sqlite3.Error — warning logged, confirmation returned regardless

### Gap 3: DDL regression tests
- `TestTargetColumnDDL.test_target_column_pre_declared_on_fresh_db`: verifies PRAGMA table_info shows target TEXT, nullable
- `TestTargetColumnDDL.test_init_db_migrates_legacy_db_without_target`: creates legacy DB without target, calls init_db, asserts column present, runs UPDATE without OperationalError

## Test Results

```
27 passed, 7 deselected (slow) — 5.10s
```

All new tests:
- `TestLabelRoute::test_label_thumbs_up_writes_target` PASSED
- `TestLabelRoute::test_label_thumbs_down_leaves_target_null` PASSED
- `TestLabelRoute::test_label_unknown_request_id_returns_ok` PASSED
- `TestTargetColumnDDL::test_target_column_pre_declared_on_fresh_db` PASSED
- `TestTargetColumnDDL::test_init_db_migrates_legacy_db_without_target` PASSED
- `TestHomePage::test_top_nav_contains_drift_monitoring_link` PASSED
- `TestHomePage::test_sidebar_does_not_contain_model_performance_link` PASSED

## Deviations from Plan

None — plan executed exactly as written.

## Must-Have Verification

| Truth | Status |
|-------|--------|
| Top-nav "Drift Monitoring" routes to /model-performance | PASS — NAV_ITEMS href updated; test asserts `href="/model-performance"` in rendered HTML |
| Duplicate sidebar "Model Performance" anchor removed | PASS — anchor block deleted; test asserts aside block contains no /model-performance |
| Each row renders thumbs-up and thumbs-down in Correct? column | PASS — hx-post="/label" buttons with r.model_name scoping |
| Thumbs-up → POST /label correct=true → target = predicted_label | PASS — test_label_thumbs_up_writes_target verifies DB row |
| Thumbs-down → POST /label correct=false → target NULL | PASS — test_label_thumbs_down_leaves_target_null verifies no write |
| After label POST, cell swaps to confirmation state | PASS — hx-target="closest td" hx-swap="innerHTML"; green Recorded / slate Noted |
| target TEXT pre-declared in CREATE TABLE DDL | PASS — test_target_column_pre_declared_on_fresh_db via PRAGMA table_info |
