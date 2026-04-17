# Phase 7: Seq Structured Logging & Observability - Research

**Researched:** 2026-04-17
**Domain:** Structured logging (structlog + Seq), FastAPI middleware, Docker Compose service integration
**Confidence:** HIGH

---

## Summary

This phase instruments the existing FastAPI app with structured logs routed to Seq. The standard approach is:

1. **structlog** as the Python logging library — produces structured event dicts, renders to JSON stdout
2. **Seq** (datalust/seq Docker image) as the log sink — receives JSON via HTTP POST to `/ingest/clef`
3. **Direct HTTP POST** from Python to Seq's CLEF endpoint — preferred over Docker log drivers for this stack

The direct HTTP POST approach (structlog + a lightweight logging.Handler that POSTs to Seq's `/ingest/clef`) is preferred over stdout + GELF Docker log driver for two reasons: it works without Docker network configuration changes, and it avoids GELF encoding complexity. The seqlog library (v0.4.3, active maintenance, datalust-recommended) wraps this pattern as a stdlib `logging.Handler`, making it compatible with the existing `logging.getLogger(__name__)` pattern throughout the codebase.

**Primary recommendation:** Use structlog for structured event construction + seqlog's `SeqLogHandler` attached to Python's stdlib logging system. structlog's `ProcessorFormatter` bridges structlog events into stdlib logging, so seqlog's handler picks them up automatically. request_id is generated in FastAPI middleware and bound via `structlog.contextvars.bind_contextvars()`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| structlog | 25.x (latest) | Structured event construction, processor chain, contextvars | Industry standard for Python structured logging; async-safe contextvars; composable processor pipeline |
| seqlog | 0.4.3 | stdlib logging.Handler that POSTs to Seq's `/ingest/clef` | Datalust's recommended Python→Seq path; env var config (SEQ_SERVER_URL); active maintenance |
| datalust/seq | latest | Seq Docker image | Official Seq container; ACCEPT_EULA=Y required; free for local |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dotenv | already present | Load SEQ_SERVER_URL from .env.local | Already in stack |
| httpx (dev dep) | already present | Tests only | Not used at runtime |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| seqlog handler | Direct requests.post() to /ingest/clef | seqlog already handles batching, retry, async flush — no reason to hand-roll |
| seqlog handler | structlog-to-seq (gjedlicska) | 2 stars, 0 forks, no releases published — avoid |
| direct HTTP | stdout + Docker GELF log driver | GELF requires Docker networking changes and a GELF input enabled in Seq — more moving parts |
| direct HTTP | stdout + Fluent Bit sidecar | Adds another Docker service and config complexity |

**Installation:**
```bash
poetry add structlog seqlog
```

---

## Architecture Patterns

### Recommended Project Structure

New files added to the existing structure:

```
app/src/
├── logging_config.py        # structlog configure() + seqlog handler setup
├── middleware/
│   └── logging.py           # FastAPI middleware: generate request_id, bind contextvars
├── main.py                  # wire middleware + call configure_logging() in lifespan
├── routes/classify.py       # emit structured log events (request.received, model.inference, etc.)
└── services/inference.py    # emit graph.built event (graph construction timing)
```

No new top-level packages — logging_config and middleware stay inside `app/src/`.

### Pattern 1: structlog Processor Chain (JSON stdout + seqlog bridge)

**What:** Configure structlog with a shared processor chain that outputs JSON. seqlog's handler is attached to Python's root stdlib logger. structlog's `ProcessorFormatter` routes structlog events through the chain before handing off to stdlib logging, so seqlog sees them.

**When to use:** Always — this is the single configuration point.

```python
# Source: structlog docs (stdlib integration) + ouassim.tech FastAPI pattern
import logging
import sys
import structlog
import seqlog
from app.src.config import LOG_LEVEL, SEQ_SERVER_URL

def configure_logging() -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    # stdout handler (Docker captures this too)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Seq handler — non-fatal if Seq URL not configured
    if SEQ_SERVER_URL:
        try:
            seq_handler = seqlog.StructuredRootLogger(
                server_url=SEQ_SERVER_URL,
                level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                batch_size=10,
                auto_flush_timeout=2,
            )
            root_logger.addHandler(seq_handler)
        except Exception as exc:
            logging.warning("Seq logging setup failed (non-fatal): %s", exc)
```

**Note on seqlog API:** `seqlog.log_to_seq()` is the simpler entry point but configures the root logger internally. For more control, use `seqlog.SeqLogHandler` directly. Verify the exact API against seqlog 0.4.3 docs at setup time.

### Pattern 2: Request ID Middleware with contextvars

**What:** Starlette-compatible middleware that generates a UUID request_id per request and binds it (plus other request metadata) to structlog's contextvars. Every subsequent `structlog.get_logger().info(...)` call in that request's async context automatically includes request_id.

**When to use:** Must be registered before the classify route handler in main.py lifespan.

```python
# Source: angelospanag.me FastAPI+structlog pattern
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=str(uuid.uuid4()),
            method=request.method,
            path=request.url.path,
        )
        response = await call_next(request)
        return response
```

**Critical:** `clear_contextvars()` must be called at request start — Python's contextvars persist across requests in the same thread/coroutine if not cleared.

**FastAPI async note:** contextvars are async-safe in Python 3.7+. Structlog's `merge_contextvars` processor reads from the current async context, so values bound in middleware propagate correctly through `await call_next()`.

### Pattern 3: Structured Event Emission

**What:** Each significant operation emits a single structlog event with all relevant fields as keyword arguments (not embedded in message strings).

**When to use:** In routes/classify.py and services/inference.py at the points defined by the event schema.

```python
# Source: structlog docs + decisions from CONTEXT.md
import structlog

log = structlog.get_logger()

# request.received (at top of classify handler, before inference)
log.info(
    "request.received",
    event_type="request.received",
    action_type="sample_click",       # or "file_upload"
    sample_name="in-dist/invoice.jpg",
    sample_set="in_dist",
    image_width=1200,
    image_height=900,
)

# model.inference (once per model, after inference completes)
log.info(
    "model.inference",
    event_type="model.inference",
    model_id="fusion_graphsage",       # first-class field, never in message string
    predicted_class="invoice",
    confidence=0.94,
    top3_classes=[
        {"class": "invoice", "score": 0.94},
        {"class": "form", "score": 0.04},
        {"class": "budget", "score": 0.01},
    ],
    model_latency_ms=45.2,
)

# graph.built (once per request)
log.info(
    "graph.built",
    event_type="graph.built",
    graph_latency_ms=12.3,
    feature_extraction_ms=38.1,
    num_nodes=49,
    knn_edges=392,
)

# model.inference.failed
log.error(
    "model.inference.failed",
    event_type="model.inference.failed",
    model_id="multimodal_gat",
    error=str(exc),
    exc_info=True,
)

# request.completed
log.info(
    "request.completed",
    event_type="request.completed",
    total_latency_ms=pipeline.total_time_ms,
    model_count=len(pipeline.results),
    best_model_id="fusion_graphsage",
    best_predicted_class="invoice",
    best_confidence=0.94,
)

# request.failed
log.error(
    "request.failed",
    event_type="request.failed",
    error=str(exc),
    exc_info=True,
)
```

**Field naming:** `event_type` is a top-level field on every event (dot-separated). `model_id` must always be a keyword argument (not embedded in the message). All latency fields must be numeric floats (not strings).

### Anti-Patterns to Avoid

- **Embedding model_id in message string:** `log.info(f"fusion_graphsage inference done")` — breaks Seq filtering. Always use `model_id="fusion_graphsage"` as a keyword arg.
- **Embedding latency in message string:** `log.info(f"took {ms}ms")` — breaks Seq charts/alerting. Use `model_latency_ms=ms`.
- **Calling `structlog.configure()` multiple times:** Configure once in `configure_logging()`, called from lifespan, never again.
- **Forgetting `clear_contextvars()` in middleware:** Without clearing, request_id from a previous request leaks into the next request on the same coroutine.
- **Using `logging.basicConfig()` after `configure_logging()`:** basicConfig is a no-op if handlers already exist, but calling it before configure_logging() pre-empts structlog's handler setup.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP batching to Seq | Custom requests.post() with retry | seqlog.SeqLogHandler | Already handles batching, async flush, timeout, retry |
| CLEF format encoding | Custom JSON serializer with @t @mt fields | seqlog handler handles encoding | CLEF has specific field conventions (@t, @mt, @l) — easy to get wrong |
| request_id generation/propagation | Thread-local or FastAPI Depends() | structlog.contextvars + middleware | contextvars are async-safe; Depends() doesn't propagate to middleware level |

**Key insight:** seqlog is the official Datalust-recommended Python→Seq path. It directly POSTs to `/ingest/clef`. Batching, flush timeout, and error suppression are already handled.

---

## Common Pitfalls

### Pitfall 1: Port Confusion (5341 vs 80 inside container)

**What goes wrong:** Container exposes port 80 internally for both UI and ingestion. Port 5341 is the *host-side* mapping in the canonical Docker example `docker run -p 5341:80`. The container does not listen on 5341 — the host maps 5341 to container's 80.

**Why it happens:** The Seq docs show `-p 5341:80` which maps host:5341 → container:80. Some docs also mention a separate 5341 ingestion-only mode, but that requires a different port binding.

**How to avoid:** In docker-compose.yml, use `ports: ["5341:80"]`. `SEQ_SERVER_URL` in .env.local should be `http://seq:80` (inside Docker network, using service name) or `http://localhost:5341` (from host). From within the Docker `app` service, use `http://seq:80`.

**Warning signs:** Connection refused on 5341 from within Docker network — check that the service name `seq` resolves correctly and use port 80 inside the network.

### Pitfall 2: structlog contextvars Not Propagating in Hybrid Sync/Async Code

**What goes wrong:** Context bound in middleware doesn't appear in logs from synchronous code called via `await loop.run_in_executor()` or thread pool. `_build_result_context()` in classify.py is currently a sync function called directly (not via executor), so this is not an issue in the current codebase. But if any sync code runs in a thread pool, it won't see contextvars.

**Why it happens:** Python's contextvars are copied to child coroutines but not to threads spawned by run_in_executor.

**How to avoid:** Keep logging calls in the async request handler scope. The current codebase's `_build_result_context()` is a synchronous function called directly from the async handler — contextvars are accessible because it runs in the same thread.

### Pitfall 3: seqlog Blocking on Seq Startup

**What goes wrong:** App fails to start if Seq isn't ready yet when configure_logging() runs.

**Why it happens:** seqlog attempts to connect or flush on initialization.

**How to avoid:** Wrap seqlog handler setup in try/except (same non-fatal pattern as monitoring store). Set `SEQ_SERVER_URL` to empty string in .env.local to disable Seq without changing code.

### Pitfall 4: Uvicorn Access Logs Not Structured

**What goes wrong:** Uvicorn's built-in access logs emit unformatted text alongside structured events, making Seq logs noisy.

**Why it happens:** Uvicorn's loggers (`uvicorn`, `uvicorn.access`, `uvicorn.error`) use their own formatters by default.

**How to avoid:** Disable uvicorn access logs by setting uvicorn loggers to use the structlog ProcessorFormatter, or silence `uvicorn.access` logger. The nymous/gist approach configures uvicorn loggers to propagate to root after structlog setup.

```python
logging.getLogger("uvicorn").handlers.clear()
logging.getLogger("uvicorn.access").handlers.clear()
logging.getLogger("uvicorn").propagate = True
logging.getLogger("uvicorn.access").propagate = True
```

### Pitfall 5: ACCEPT_EULA Required

**What goes wrong:** Seq container immediately exits if `ACCEPT_EULA=Y` is not set.

**How to avoid:** Always set `ACCEPT_EULA=Y` in the seq service environment in docker-compose.yml.

---

## Code Examples

### Seq Docker Compose Service

```yaml
# Source: datalust.co/docs/getting-started-with-docker
seq:
  image: datalust/seq:latest
  environment:
    ACCEPT_EULA: "Y"
  ports:
    - "5341:80"
  volumes:
    - seq-data:/data
  restart: unless-stopped

volumes:
  seq-data:
```

**Inside Docker network:** app service reaches Seq at `http://seq:80`
**From host browser:** UI is at `http://localhost:5341`

### config.py additions

```python
# No hardcoded values — all from env
SEQ_SERVER_URL = os.environ.get("SEQ_SERVER_URL", "")   # empty = disabled
SEQ_API_KEY = os.environ.get("SEQ_API_KEY", "")          # blank = no auth (local)
ENVIRONMENT = os.environ.get("ENVIRONMENT", "local")
```

### .env.local additions

```bash
SEQ_SERVER_URL=http://seq:80
SEQ_API_KEY=
ENVIRONMENT=local
```

### Observability nav item linking to Seq UI

The existing `NAV_ITEMS` in classify.py hardcodes `/observability` as the Observability href. This needs to point to `http://localhost:5341` (the host-facing Seq UI). Pattern from EVIDENTLY_DASHBOARD_URL: read `SEQ_UI_URL` from config and pass it to template context.

```python
# In config.py
SEQ_UI_URL = os.environ.get("SEQ_UI_URL", "http://localhost:5341")

# In classify.py NAV_ITEMS — change to dynamic lookup
# OR pass SEQ_UI_URL into template context so base.html can render the correct href
```

**Implementation choice:** Either update NAV_ITEMS to be a function that reads from config, or pass seq_ui_url as a template context variable. The EVIDENTLY_DASHBOARD_URL pattern uses a target="_blank" link in the nav, which matches what's needed here.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| print()/logging.basicConfig() | structlog with processor chain | Industry shift 2020-2022 | Queryable structured fields vs grep |
| GELF Docker log driver → Seq | Direct HTTP POST via seqlog handler | Still both valid; direct HTTP simpler for single-service | No Docker network config changes needed |
| request_id via thread-local | structlog.contextvars | Python 3.7+ (2018) | Async-safe context propagation |

---

## Open Questions

1. **seqlog.log_to_seq() vs SeqLogHandler directly**
   - What we know: seqlog 0.4.3 provides both `log_to_seq()` (simple) and `SeqLogHandler` (manual). The `log_to_seq()` call configures the root logger.
   - What's unclear: Whether `log_to_seq()` clears existing handlers (it might conflict with structlog's stdout handler already attached to root).
   - Recommendation: Use `SeqLogHandler` directly and `addHandler()` to avoid handler-clearing side effects. Verify by checking seqlog source at implementation time.

2. **seqlog and structlog ProcessorFormatter compatibility**
   - What we know: seqlog's handler receives stdlib LogRecord objects. structlog's ProcessorFormatter converts structlog events to LogRecord before passing to handlers. So seqlog should see structlog events.
   - What's unclear: Whether seqlog re-encodes the JSON string from ProcessorFormatter into CLEF correctly, or produces double-encoded JSON.
   - Recommendation: Test with a single request in dev first. If double-encoding occurs, use a CLEF-aware structlog renderer as the final processor instead of JSONRenderer, and send directly to seqlog's endpoint without ProcessorFormatter intermediary.

3. **Seq free tier limits**
   - What we know: Seq is free for single-user local use.
   - What's unclear: Event volume limits in the free tier.
   - Recommendation: For a local demo app with low request volume, this is a non-issue.

---

## Sources

### Primary (HIGH confidence)
- datalust.co/docs/getting-started-with-docker — Seq Docker image, ACCEPT_EULA, port 5341:80 mapping
- datalust.co/docs/posting-raw-events — CLEF format, /ingest/clef endpoint, @t @mt @l @x fields
- datalust.co/docs/using-python — seqlog as recommended Python library
- hub.docker.com/r/datalust/seq — datalust/seq:latest image, /data volume, 271.8 MB

### Secondary (MEDIUM confidence)
- structlog.org/en/stable/contextvars.html (attempted fetch, returned 403) + WebSearch summary — merge_contextvars, bind_contextvars, clear_contextvars, async behavior
- angelospanag.me/blog/structured-logging-using-structlog-and-fastapi — middleware pattern with contextvars, processor chain
- ouassim.tech/notes/setting-up-structured-logging-in-fastapi-with-structlog — ProcessorFormatter bridge with shared_processors

### Tertiary (LOW confidence)
- johal.in/structlog-contextvars-python-async-logging-2026 — contextvars async behavior (single blog post, unverified)
- github.com/gjedlicska/structlog-to-seq — structlog-to-seq library (2 stars, no releases — do not use)

---

## Metadata

**Confidence breakdown:**
- Standard stack (structlog + seqlog + datalust/seq): HIGH — Datalust official docs confirm seqlog recommendation; structlog is well-established
- Architecture (middleware + contextvars pattern): HIGH — Multiple corroborating sources, confirmed against structlog official docs summary
- Pitfalls (port confusion, ACCEPT_EULA, contextvars clearing): HIGH — Directly from official docs and known async Python behavior
- seqlog/ProcessorFormatter compatibility: MEDIUM — Logical from both APIs but not verified with working code; Open Question 2 covers this

**Research date:** 2026-04-17
**Valid until:** 2026-05-17 (seqlog and structlog are stable; Seq Docker image version pinning not critical)
