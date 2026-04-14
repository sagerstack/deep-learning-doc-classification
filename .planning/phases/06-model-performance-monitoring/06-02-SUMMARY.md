---
plan: 06-02
phase: 06-model-performance-monitoring
status: complete
completed: 2026-04-14
commit: 2277af2
---

# Plan 06-02: Evidently Batch Job + Operator Docs

## What Was Built

A batch monitoring pipeline that reads logged inference events from SQLite,
groups them per model_id, and produces separate Evidently reports per model
for a configurable time window. The system runs fully offline without cloud
credentials, and includes a reference dataset bootstrap script and operator
documentation covering the end-to-end monitoring workflow.

## Deliverables

- `scripts/monitoring/run_evidently.py`: Batch job with `build_reports_for_window()` — groups events by model_id, runs DataSummaryPreset + DataDriftPreset (when reference present), writes per-model HTML+JSON artifacts offline or publishes to Evidently Cloud
- `scripts/monitoring/bootstrap_reference.py`: Reference dataset generator with `bootstrap_reference_dataset()` — real-traffic export (--window-hours) or synthetic balanced mode (--synthetic) covering all 16 labels and 6 models
- `monitoring/reference/README.md`: Reference dataset documentation — what it is, how generated, when to refresh
- `docs/monitoring.md`: Operator guide covering event schema, multiple-model strategy, offline/cloud usage, labeled backfill strategy, and exact example commands
- `app/src/monitoring/store.py`: Added `fetch_events_as_dataframe()` helper for batch job use (pandas import kept separate from serving path via TYPE_CHECKING)
- `docker-compose.yml`: Optional `monitoring` profile service for container-based job invocation
- `pyproject.toml` + `.env.example`: Added evidently/pandas dependencies and Evidently Cloud env vars

## Key Decisions

- `DataDriftPreset` requires reference dataset: when reference absent, fall back to `DataSummaryPreset` only — avoids hard failure on first run before bootstrap
- `MulticlassClassification` requires both target and prediction columns: unlabeled report omits the classification block entirely; labeled report adds it only when `target` column is present and non-null
- Timestamp column converted to `pd.to_datetime(utc=True)` before passing to Evidently: Evidently's drift visualization code internally calls `.dt.to_period()` which requires a proper datetime type
- `fetch_events_as_dataframe()` uses `TYPE_CHECKING` guard for the pandas import: keeps the serving path free of pandas overhead at import time
- One report per model per window (not a combined report): enables per-model drift tracking and independent alerting

## Issues Encountered

Three bugs fixed inline during execution:

1. [Rule 1 - Bug] `MulticlassClassification` raised ValueError when `target` was absent — required both target and prediction. Fixed by conditionally including the classification block only when labels are present.
2. [Rule 1 - Bug] `DataDriftPreset` raised `ValueError: Reference dataset should be present` when no reference file existed. Fixed by only appending `DataDriftPreset` to the metrics list when a reference dataset is available.
3. [Rule 1 - Bug] `AttributeError: Can only use .dt accessor with datetimelike values` — Evidently's drift visualization attempted `.dt.to_period()` on a string timestamp column. Fixed by converting the timestamp column to `pd.datetime` with UTC timezone before wrapping in an Evidently Dataset.

## Test Results

```
# Bootstrap synthetic reference (600 rows, 6 models, 16 labels)
2026-04-14 09:40:46  INFO  Synthetic reference dataset written to monitoring/reference/reference_dataset.parquet
2026-04-14 09:40:46  INFO    Rows: 600  Models: 6  Labels: 16

# Full dry-run with reference dataset — 2 models, separate artifacts
2026-04-14 09:41:23  INFO  Loaded 60 events across 2 model(s)
2026-04-14 09:41:23  INFO  Loaded reference dataset: 60 rows from reference_dataset.parquet
2026-04-14 09:41:23  INFO  Processing model_id=cnn_baseline  rows=30
2026-04-14 09:41:30  INFO    [offline] wrote cnn_baseline__unlabeled__20260414T014123.html
2026-04-14 09:41:30  INFO    [offline] wrote cnn_baseline__unlabeled__20260414T014123.json
2026-04-14 09:41:30  INFO  Processing model_id=graphsage_fusion  rows=30
2026-04-14 09:41:37  INFO    [offline] wrote graphsage_fusion__unlabeled__20260414T014130.html
2026-04-14 09:41:37  INFO    [offline] wrote graphsage_fusion__unlabeled__20260414T014130.json
2026-04-14 09:41:37  INFO  Reports generated for 2 model(s):
2026-04-14 09:41:37  INFO    cnn_baseline → 2 artifact(s)
2026-04-14 09:41:37  INFO    graphsage_fusion → 2 artifact(s)
EXIT=0
```
