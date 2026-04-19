# Phase 6: Model Performance Monitoring - Research

**Researched:** 2026-04-13
**Domain:** Evidently-based monitoring for a FastAPI multi-model document classification app
**Confidence:** HIGH

## Summary

The correct way to integrate Evidently into this project is as a batch monitoring layer, not as part of the synchronous `/classify` request path. Evidently's platform and library are built around evaluating structured datasets in batches and publishing reports to a shared project/dashboard. For this app, each classification request produces multiple model outputs, so the event schema must persist one row per `(request_id, model_id)` pair rather than one row per request.

The recommended architecture is:

1. FastAPI serves predictions normally.
2. The classify route writes one structured monitoring event per model prediction to a durable local store.
3. A separate batch job reads recent events, groups them by `model_id`, builds one Evidently report per model per batch window, and publishes them to a shared Evidently project using tags/metadata.
4. The app sidebar exposes a "Model Performance" entry that routes to a local `/model-performance` endpoint, which redirects to the configured Evidently dashboard URL.

**Primary recommendation:** Use SQLite for the initial event store, one Evidently project for the deployment environment, and one report run per model per batch with tags such as `env`, `model_id`, `model_version`, and `batch_window`. This keeps the runtime path simple and fits Evidently's batch-first model.

## Key Findings

### How Evidently fits this app
- Evidently expects structured tabular datasets rather than raw image tensors. The monitoring job should therefore consume derived serving data: predicted label, confidence, class probabilities, latencies, input metadata, and pipeline flags.
- Evidently can organize multiple models in a single project through tags and metadata. This is the cleanest pattern for the demo app because all models are deployed together and evaluated from the same inference traffic.
- Evidently also supports several classification outputs inside one dataset definition, but for production monitoring the cleaner operational pattern is one report per `model_id` per batch.

### Multi-model serving pattern
- Each request already produces outputs for several models. Monitoring must not flatten these into a single row, because drift, confidence, and latency are model-specific.
- The event store should therefore contain:
  - shared request fields: `request_id`, `timestamp`, `sample_type`, `image_width`, `image_height`
  - per-model fields: `model_id`, `model_version`, `predicted_label`, `confidence`, probabilities, per-model latency
  - pipeline fields: `feature_time_ms`, `graph_time_ms`, `ocr_available`, `text_density_available`
- The monitoring job should group by `model_id` and produce separate Evidently runs with tags. This preserves clean dashboards and avoids having to compare heterogeneous prediction columns in one report.

### Recommended stack
| Component | Choice | Why |
|-----------|--------|-----|
| Event store | SQLite | Built-in, durable, easy to query from app and batch job |
| Batch processing | pandas | Natural input format for Evidently |
| Monitoring library | evidently | Purpose-built for drift/quality reports and dashboard publishing |
| Dashboard access | env-configured URL + redirect route | Keeps deployment-specific URL out of templates and code |
| Scheduling | cron / launchd / manual script / separate container | Avoids putting batch work inside FastAPI request handling |

### Metrics to monitor first
**Unlabeled monitoring**
- predicted class distribution by model
- confidence distribution by model
- p50/p95 latency by model
- request volume by sample source (`upload` vs `sample`)
- OCR fallback rate
- text-density availability rate
- malformed input / request failure rate

**Labeled monitoring (later)**
- accuracy
- macro F1
- per-class precision/recall/F1
- confusion trends
- low-confidence error slices

## Architecture Pattern

### Pattern 1: One row per model prediction
**What:** Persist one monitoring event for each model prediction generated inside a request.
**When to use:** Always, because the app serves multiple models side-by-side.
**Why:** This makes it possible to monitor drift, quality, and latency independently per model.

### Pattern 2: One Evidently report per model per batch window
**What:** The monitoring job loads recent events, filters or groups by `model_id`, and publishes a separate report run for each model.
**When to use:** For both unlabeled and labeled monitoring.
**Why:** Operationally simpler than packing all model outputs into one report, and aligns with dashboard filtering by tags.

### Pattern 3: Dashboard route as redirect
**What:** Add a local `/model-performance` route that redirects to `EVIDENTLY_DASHBOARD_URL`.
**When to use:** Always in the app UI.
**Why:** The sidebar stays stable even if the actual dashboard URL changes between local, staging, and demo environments.

## Anti-Patterns to Avoid

- Running Evidently inside the live `/classify` request path. It will add latency and couples UI responsiveness to monitoring.
- Logging one row per request with all model outputs embedded in a single JSON blob. It makes per-model analysis awkward and fragile.
- Hardcoding the dashboard URL in the template. Use configuration and a route indirection instead.
- Requiring synchronous ground-truth labels. The schema should accept delayed labels later.
- Treating raw image bytes as the primary monitoring surface. Monitor structured serving outputs and metadata instead.

## Recommended File Layout

```text
app/src/
  monitoring/
    __init__.py
    schema.py            # event schema / helpers
    store.py             # SQLite initialization + inserts + window queries
  routes/
    monitoring.py        # GET /model-performance redirect route
scripts/monitoring/
  run_evidently.py       # batch job for unlabeled and labeled monitoring
  bootstrap_reference.py # create reference dataset from seeded events or samples
monitoring/
  reference/
    baseline_events.parquet
  outputs/               # optional local HTML/JSON report artifacts for dev
```

## Implementation Notes

- Store class probabilities in explicit columns such as `proba_letter`, `proba_form`, etc. This makes Evidently dataset definitions straightforward.
- Persist a `request_id` so all model rows from one classify action can be correlated later.
- Include `model_version` in the event store even if it starts as a checkpoint filename. It will matter once new weights are compared.
- Keep the monitoring job callable manually first. Add scheduling only after the batch logic is stable.
