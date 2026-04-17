# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** The GraphSAGE model must demonstrate whether graph-based spatial reasoning improves document classification over CNN-only baselines — with clear, reproducible evidence.
**Current focus:** Phase 7 in progress — Seq structured logging observability

## Current Position

Phase: 7 of 7 (Seq Structured Logging + Observability)
Plan: 3 of 3 (completed: 01, 02, 03)
Status: In progress
Last activity: 2026-04-17 — Completed 07-03-PLAN.md (request events + request_id unification)

Progress: [█████████░] 96%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 6min
- Total execution time: 0.9 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-notebook-foundation-data-features | 1 | 12min | 12min |
| 02.1-hybrid-fusion-positional-encoding | 2 | 6min | 3min |
| 02.2-text-aware-hybrid-gnn | 1 | 7min | 7min |
| 05-demo-application-classification-page | 5 | 24min | 5min |
| 07-seq-structured-logging-observability | 1 | 2min | 2min |

**Recent Trend:**
- Last 5 plans: 7min, 4min, 8min, 1min, 6min
- Trend: Stable ~5min

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Subset-first development: Iterate on architecture locally (5-10% data) before full training on cluster
- CNN baseline from reference notebook: Avoid duplicating work, use existing baseline for comparison
- Graph construction as experimentation: Finding the right approach is part of the contribution
- Incremental notebook building: Each phase adds sections to a single Jupyter notebook that grows phase by phase
- **[01-01]** Dataset source: chainyo/rvl-cdip instead of aharley/rvl_cdip (modern parquet format compatible with datasets v4.x)
- **[01-01]** Device detection pattern: MPS -> CUDA -> CPU (supports local Mac dev and cluster training)
- **[01-01]** Sample mode with streaming + shuffle for random sampling without full download
- **[02.1-01]** Single cache format for all notebooks (hybrid notebooks use global_feat, plain GraphSAGE ignores it)
- **[02.1-01]** Normalized 2D coordinates (2 dims) over sinusoidal PE (64+ dims) to minimize overfitting
- **[02.1-01]** Store global_feat as [1, 2048] for correct PyG batching to [batch_size, 2048]
- **[02.1-01]** Direct concatenation fusion without CNN projection (parameter-efficient on 2560 samples)
- **[02.2-01]** Raw DBNet probability map (return_model_output=True) over bounding-box reconstruction for text density
- **[02.2-01]** MPS falls back to CPU for doctr inference (doctr lacks MPS support)
- **[02.2-01]** TextAwareGraphSAGE as independent class (not subclass of HybridGraphSAGE) for independent evolution
- **[05-01]** exp26 text_gate is plain nn.Linear with sigmoid in forward (not Sequential) per checkpoint keys
- **[05-01]** exp27 attn_key uses bias=False per checkpoint lacking attn_key.bias
- **[05-01]** torch_geometric.utils.scatter used instead of torch_scatter (not installed, same API)
- **[05-02]** Starlette 1.0 TemplateResponse uses keyword args (request=, name=) not positional
- **[05-02]** Tailwind CDN with inline config for MD3 color tokens rather than build step
- **[05-02]** HTMX form with hx-post for classify, targeting #results div
- **[05-03]** needs_pe flag on ModelSpec to distinguish raw CNN (2048) vs PE-augmented (2050) node features
- **[05-03]** Feature extractor built from fine-tuned checkpoint, not pretrained ImageNet weights
- **[05-03]** OCR-dependent models return placeholder result when tesseract unavailable
- **[05-04]** matplotlib.use('Agg') at module top before any pyplot imports for server-safe rendering
- **[05-04]** CSS grid with opacity mapping for text density/node importance (no matplotlib overhead)
- **[05-04]** Inline SVG for graph topology (embeddable via Jinja2 safe filter)
- **[05-05]** jinja2-fragments Jinja2Blocks for block_name HTMX partial rendering
- **[05-05]** Template split into 4 include partials for maintainability
- **[05-05]** Device detection in main.py lifespan (MPS -> CUDA -> CPU) independent of src.config.Config
- **[06-01]** sqlite3 stdlib only for monitoring store — no ORM dependency
- **[06-01]** One row per model per request — enables per-model drift tracking independently
- **[06-01]** Monitoring failure is non-fatal (try/except + warning log) — inference must not break on DB write failure
- **[06-01]** EVIDENTLY_DASHBOARD_URL env var drives redirect; empty string triggers fallback HTML page
- **[06-01]** sample_type column distinguishes "sample" vs "upload" for drift segmentation
- **[06-02]** DataDriftPreset omitted when reference absent; DataSummaryPreset always runs
- **[06-02]** Timestamp column must be converted to pd.datetime(utc=True) before passing to Evidently drift metrics
- **[06-02]** Unlabeled report omits MulticlassClassification block; labeled report adds it only when target column present
- **[06-03]** correct=true writes target=predicted_label; correct=false is no-op (NULL more honest than guessing true label)
- **[06-03]** hx-vals uses string "true"/"false" not JSON boolean; route normalises via .lower() check
- **[06-03]** request_id lifted before monitoring try/except to guarantee availability in template context
- **[07-01]** SEQ_SERVER_URL default is empty string (disables ingestion non-fatally) — same pattern as EVIDENTLY_DASHBOARD_URL
- **[07-01]** SEQ_SERVER_URL=http://seq:80 inside Docker compose network; http://localhost:5341 from host (different URLs for same service)
- **[07-01]** ACCEPT_EULA: Y mandatory in datalust/seq environment block — container exits without it
- **[07-02]** SeqLogHandler(server_url, api_key, batch_size=10, auto_flush_timeout=2) — api_key param confirmed valid in seqlog 0.4.3; no fallback to log_to_seq() needed
- **[07-02]** Processor chain: merge_contextvars → add_log_level → add_logger_name → TimeStamper(iso/utc) → StackInfoRenderer → UnicodeDecoder; terminal: remove_processors_meta + JSONRenderer
- **[07-02]** LoggingMiddleware registered before CORSMiddleware — outermost in Starlette stack (confirmed via app.user_middleware)
- **[07-02]** clear_contextvars() in dispatch() clears ALL contextvars including environment; must re-bind environment after clearing
- **[07-03]** request_id is required kwarg on _build_result_context (no default) — enforces caller supply, prevents silent dual-UUID split
- **[07-03]** image_width/image_height bound AFTER image load (not in request.received) — dimensions unknown at handler entry
- **[07-03]** request.failed emitted on ALL early-exit paths so failures are always observable in Seq
- **[07-03]** Observability nav item uses SEQ_UI_URL env var (default http://localhost:5341) with target=_blank

### Pending Todos

None - poetry.lock pending from Phase 6 resolved in 07-01 (SSL did not block this run).

### Blockers/Concerns

- SSL certificate verification fails in current environment (corporate/self-signed cert). Workaround in place (ssl bypass + curl --insecure for model weights). Does not affect runtime.

## Session Continuity

Last session: 2026-04-17T15:33:47Z
Stopped at: Completed 07-03-PLAN.md (request events + request_id unification)
Resume file: None

**Milestone status:** Phase 7 plans 01-03 complete. Route-level structured events live: request.received/completed/failed emitted per classify request, end-to-end request_id correlation between middleware + log events + monitoring SQLite row. Plan 04 (inference-layer events) can proceed immediately — contextvars already pre-loaded with action_type, sample_name, sample_set, image_width, image_height.
