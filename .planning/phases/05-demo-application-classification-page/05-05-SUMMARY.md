---
phase: 05
plan: 05
subsystem: demo-application
tags: [fastapi, htmx, jinja2, inference-pipeline, visualization]
dependency-graph:
  requires: ["05-03", "05-04"]
  provides: ["full-request-response-cycle", "htmx-partial-rendering", "model-lifespan-loading"]
  affects: ["05-06"]
tech-stack:
  added: [jinja2-fragments]
  patterns: [htmx-partial-blocks, alpine-accordion, lifespan-model-loading]
key-files:
  created:
    - app/src/templates/partials/cnn_features.html
    - app/src/templates/partials/graph_construction.html
    - app/src/templates/partials/model_predictions.html
    - app/src/templates/partials/detailed_analysis.html
  modified:
    - app/src/main.py
    - app/src/routes/classify.py
    - app/src/templates/base.html
decisions:
  - jinja2-fragments Jinja2Blocks for block_name partial rendering instead of custom logic
  - Template split into 4 include partials for maintainability
  - Auto-submit form on file selection for seamless UX
  - CPU fallback for device detection (MPS -> CUDA -> CPU)
metrics:
  duration: 5min
  completed: 2026-04-07
---

# Phase 5 Plan 5: Classification Page Wiring Summary

Full request-response cycle wired: upload image -> lifespan-loaded models -> inference pipeline -> visualization generation -> HTMX partial rendering with all 4 result sections.

## What Was Done

### Task 1: App Lifespan + POST /classify Route Handler
- Updated `main.py` lifespan to detect device (MPS/CUDA/CPU) and load all models at startup via `load_all_models()`
- Stores registry and device in `app.state` for route access
- `classify.py` POST handler accepts file upload or sample name
- Runs full inference pipeline, generates all visualizations (heatmap, text density, graph SVGs, probability bars, 16-class charts, node importance)
- Builds structured context dict and renders via `Jinja2Blocks` with `block_name="results_section"` for HTMX partials

### Task 2: Complete Template with All 4 Sections
- Split into 4 partial templates for readability: cnn_features, graph_construction, model_predictions, detailed_analysis
- CNN Features: activation heatmap (base64 img) + text density map (CSS grid)
- Graph Construction: grid vs kNN inline SVGs + stats bar (nodes, edges, degree)
- Model Predictions: table with type badges, confidence, sorted by confidence + top-3 probability bars
- Detailed Analysis: Alpine.js accordion with per-model 16-class chart, inference time, node importance map
- HTMX loading indicator with spinner during inference
- Auto-submit on file selection (`onchange="this.form.requestSubmit()"`)

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

1. Used `jinja2-fragments` `Jinja2Blocks` class which supports `block_name` kwarg for partial rendering
2. Split result sections into `{% include %}` partials rather than inline for maintainability
3. Added `onchange` auto-submit on file input for seamless upload UX
4. Device detection in main.py rather than importing src.config.Config (avoids loading dotenv/numpy in web app)

## Verification Results

- GET / returns 10,859 chars with upload placeholder
- POST /classify with test image returns 237,866 chars with all 4 sections
- Base64 images, inline SVGs, and Alpine.js accordion all render correctly
- Full pipeline: feature extraction -> graph construction -> 6 model inference -> visualization -> template rendering

## Next Phase Readiness

Plan 05-06 (final polish/deployment) can proceed. The classification page is fully functional end-to-end.
