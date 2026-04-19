---
status: complete
phase: 06-model-performance-monitoring
source: 06-01-SUMMARY.md, 06-02-SUMMARY.md
started: 2026-04-14T00:00:00Z
updated: 2026-04-14T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Sidebar "Model Performance" link
expected: Open the app in the browser. The left sidebar should have a "Model Performance" link at the bottom of the nav panel. It should be visible without scrolling or expanding anything.
result: issue
reported: "why sidebar? there is already a title bar on top. Drift Monitoring should route to this new page"
severity: major

### 2. /model-performance without config shows fallback
expected: With EVIDENTLY_DASHBOARD_URL not set (or empty), navigating to http://localhost:8000/model-performance should show an HTML page clearly stating the dashboard URL is not configured — not a crash or blank page.
result: pass

### 3. /model-performance with URL redirects correctly
expected: With EVIDENTLY_DASHBOARD_URL=https://example.com set in the environment, navigating to /model-performance should redirect the browser to https://example.com (or wherever the URL is set).
result: pass

### 4. Classify request writes monitoring rows
expected: After running a classify on any sample image, the SQLite DB at monitoring/data/inference_events.sqlite3 (or the path in MONITORING_DB_PATH) should contain one row per model that ran. You can check with: `poetry run python3 -c "from app.src.monitoring.store import query_events; from pathlib import Path; rows = query_events(Path('monitoring/data/inference_events.sqlite3')); print(len(rows), 'rows')"` — should print a positive number.
result: issue
reported: "when i load a document, it keeps spinning, not done feature extraction yet"
severity: blocker

### 5. Reference dataset bootstrap runs
expected: Running `poetry run python scripts/monitoring/bootstrap_reference.py --synthetic` should complete without errors and produce monitoring/reference/reference_dataset.parquet. The output should mention how many rows and models were written.
result: pass

### 6. Batch monitoring job generates per-model reports
expected: Running `poetry run python scripts/monitoring/run_evidently.py --window-hours 24 --offline` should produce separate HTML and JSON files under monitoring/output/ (or EVIDENTLY_OFFLINE_OUTPUT_DIR), one set per model_id found in the DB. If the DB is empty, the script should exit cleanly with a message rather than crashing.
result: pass

### 7. Monitoring docs exist and are usable
expected: `docs/monitoring.md` exists and contains example commands for at least: bootstrapping the reference dataset, running the batch job offline, and enabling labeled monitoring via target column backfill.
result: pass

## Summary

total: 7
passed: 5
issues: 2
pending: 0
skipped: 0

## Gaps

- truth: "Nav link to /model-performance is visible in the top title bar with label 'Drift Monitoring'"
  status: failed
  reason: "User reported: why sidebar? there is already a title bar on top. Drift Monitoring should route to this new page"
  severity: major
  test: 1
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "The 'Correct?' column in the Model Predictions table captures thumbs-up/down feedback and writes target labels to the monitoring store"
  status: failed
  reason: "Column is a dead placeholder — both branches render a dash. Thumbs-up should write target=predicted_class, thumbs-down should skip. Needs: UI buttons + POST /label route + DB write to target column (already exists in schema)."
  severity: major
  test: enhancement
  root_cause: "Column was scaffolded but never wired up"
  artifacts:
    - path: "app/src/templates/partials/model_predictions.html"
      issue: "Correct? column renders dash unconditionally"
  missing:
    - "Thumbs-up/down buttons with HTMX POST /label (request_id, model_id, correct: bool)"
    - "POST /label route writing target column to inference_events SQLite"
  debug_session: ""
