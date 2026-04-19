# Phase 7: Seq Structured Logging & Observability - Context

**Gathered:** 2026-04-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Instrument the existing FastAPI app with structured logs via structlog → Seq for every user action, API call, and model interaction. Seq runs as a Docker Compose service alongside the app. The existing Observability nav item in the title bar links to the Seq UI. No new inference logic — logging wraps what already exists.

</domain>

<decisions>
## Implementation Decisions

### Event granularity per model
- One structured event per model per request, emitted after inference completes (not start/end pair)
- Request-level events: `request.received`, `request.completed`, `request.failed` — all three
- Model failures get a dedicated `model.inference.failed` event (separate from `request.failed`)
- Graph construction gets its own `graph.built` event — one per request (shared across models), not repeated per model

### User action capture
- Server-side from route params — infer action type from /classify request (no frontend changes, no dedicated endpoint)
- User action is captured at request entry (before inference), as fields on `request.received`
- Action fields on every user action event: `action_type` (sample_click | file_upload), `sample_name` / filename, image dimensions (width × height), `sample_set` (in_dist | oo_dist)
- `sample_set` is a required field — distinguishes in-distribution from OO-distribution samples in Seq

### Standard log field schema
- Every event carries: `timestamp` (ISO 8601 UTC), `event_type` (dot-separated e.g. `request.received`), `level` (info/warning/error), `environment` (local/prod), `request_id`
- Model identity field: `model_id` slug only (e.g. `fusion_graphsage`, `cnn_baseline`) — not human label. Enables clean Seq grouping and filtering.
- Latency fields as structured numeric values (not embedded in message strings): `model_latency_ms`, `graph_latency_ms`, `total_latency_ms`, `feature_extraction_ms`
- Inference result fields on `model.inference`: `predicted_class`, `confidence` (float 0–1), `top3_classes` (list with scores), `model_latency_ms`

### Claude's Discretion
- Log ingestion method (stdout via Docker log driver vs direct HTTP POST to Seq API)
- structlog processor chain configuration
- How `request_id` is generated and propagated (middleware vs context var)
- Seq Docker image version and port configuration details

</decisions>

<specifics>
## Specific Ideas

- model_id must be a first-class structured field on every model-related event — not embedded in the message string — so Seq can group and filter by model without text parsing
- All latency values must be numeric fields, not strings, to enable Seq charts and alerting
- Logs must be complete enough to troubleshoot any inference failure without reading source code

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-seq-structured-logging-observability*
*Context gathered: 2026-04-17*
