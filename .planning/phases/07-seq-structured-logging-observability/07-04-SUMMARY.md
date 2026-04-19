---
phase: 07-seq-structured-logging-observability
plan: 04
subsystem: observability
tags: [structlog, seqlog, inference, logging, monitoring]

# Dependency graph
requires:
  - phase: 07-02
    provides: structlog configure_logging() + LoggingMiddleware with request_id contextvars
provides:
  - graph.built event with graph_latency_ms, feature_extraction_ms, num_nodes, knn_edges, grid_edges
  - model.inference event per model with model_id (slug), predicted_class, confidence, top3_classes, model_latency_ms
  - model.inference.failed event on per-model exception with model_id and error string
  - Per-model exception handling in GNN loop (non-fatal, pipeline continues)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Module-level log = structlog.get_logger() for structured events in service modules
    - _emit_model_inference helper encapsulates top3 computation + log.info call
    - try/except around per-model inference with model.inference.failed on exception
    - model_id as first-class kwarg on all inference events (never embedded in message)

key-files:
  created: []
  modified:
    - app/src/services/inference.py

key-decisions:
  - "model_id uses InferenceResult.model_name (slug), not display_name"
  - "model_id is a keyword argument on log.info(), never embedded in message string"
  - "OCR-unavailable placeholder branches do not emit model.inference — skipped via continue"
  - "import logging retained — only the GAT except branch warning was replaced"
  - "GNN loop wrapped in try/except so a single broken model does not abort the pipeline"

patterns-established:
  - "_emit_model_inference(r): builds top3 list from probabilities, calls log.info with float coercions"
  - "Per-model try/except: model.inference.failed on error, exc_info=True for stack trace in Seq"

# Metrics
duration: 1min
completed: 2026-04-17
---

# Phase 7 Plan 04: Inference-Layer Structured Events Summary

**Structlog graph.built + per-model model.inference/model.inference.failed events wired into inference pipeline with non-fatal per-model exception handling**

## Performance

- **Duration:** 1 min
- **Started:** 2026-04-17T15:35:16Z
- **Completed:** 2026-04-17T15:36:29Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Emits exactly one `graph.built` event per request after graph construction with graph_latency_ms, feature_extraction_ms, num_nodes, knn_edges, grid_edges
- Emits one `model.inference` event per successful model inference (CNN baseline, all GNN models, Multimodal GAT) with model_id as first-class slug field
- Emits `model.inference.failed` on per-model exceptions (GNN loop + Multimodal GAT), pipeline continues without aborting

## Task Commits

1. **Task 1: Add structlog logger and emit graph.built event** - `5bd637a` (feat)
2. **Task 2: Emit model.inference events for all models** - `56d29e5` (feat)

**Plan metadata:** (included in this docs commit)

## Files Created/Modified
- `app/src/services/inference.py` - Added `log = structlog.get_logger()`, `_emit_model_inference` helper, graph.built + per-model inference events, try/except wrapping for GNN loop

## Decisions Made
- `model_id` uses `InferenceResult.model_name` (slug), not `display_name` — keeps it machine-parseable in Seq queries
- `model_id` is always a keyword argument, never embedded in the message string — satisfies SC-4
- OCR-unavailable placeholder branches skipped via `continue` so no false model.inference events with zero confidence
- `import logging` retained at line 3 — only the GAT `except` branch `logger.warning` was replaced; all other `logger.*` paths untouched
- GNN loop wrapped in `try/except` at the `_run_single_model` call site rather than inside it — gives access to `spec.name` for `model_id` in the failure event

## Helper Signature

```python
def _emit_model_inference(r: "InferenceResult") -> None:
    # top3: sorted(enumerate(r.probabilities), key=score, reverse=True)[:3]
    # mapped to [{"class": RVL_CDIP_LABELS[i], "score": float(p)} ...]
    log.info(
        "model.inference",
        event_type="model.inference",
        model_id=r.model_name,          # slug, not display_name
        predicted_class=r.predicted_class,
        confidence=float(r.confidence),
        top3_classes=top3,
        model_latency_ms=float(r.inference_time_ms),
    )
```

Float coercions prevent torch/numpy tensor types from leaking into JSON serialization.

## Per-Model Error Handling Scope

| Scope | Event emitted | Pipeline behavior |
|---|---|---|
| GNN loop (_run_single_model) | model.inference.failed with model_id=spec.name | Continue to next model |
| Multimodal GAT | model.inference.failed with model_id="multimodal_gat" | Continue (sort + return) |

## Confirmations

- `import logging` is present at line 3 of inference.py — verified with `grep -n "^import logging"`
- `PipelineResult.total_time_ms` is unchanged — still computed at the bottom as `(time.perf_counter() - pipeline_start) * 1000`
- `import app.src.services.inference` exits 0

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

Phase 7 plans 01-04 complete. All structured events are live:
- request.received / request.completed / request.failed (07-03)
- graph.built (this plan)
- model.inference / model.inference.failed per model (this plan)

All events inherit request_id from LoggingMiddleware contextvars. End-to-end correlation is functional.

---
*Phase: 07-seq-structured-logging-observability*
*Completed: 2026-04-17*
