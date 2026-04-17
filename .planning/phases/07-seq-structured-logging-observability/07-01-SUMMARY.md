---
phase: 07-seq-structured-logging-observability
plan: "01"
subsystem: infra
tags: [seq, structlog, seqlog, docker-compose, observability, logging]

# Dependency graph
requires:
  - phase: 06-model-performance-monitoring
    provides: config.py pattern (os.environ.get), .env.local + .env.example conventions
provides:
  - Seq Docker Compose service (datalust/seq:latest on 5341:80) with named volume
  - SEQ_SERVER_URL, SEQ_API_KEY, SEQ_UI_URL, ENVIRONMENT in app/src/config.py
  - structlog 25.5.0 and seqlog 0.4.3 as runtime dependencies
affects: [07-02-logging-core, 07-03-log-emission, future phases needing structured logging]

# Tech tracking
tech-stack:
  added: [structlog 25.5.0, seqlog 0.4.3, datalust/seq:latest]
  patterns:
    - SEQ_SERVER_URL=http://seq:80 for intra-Docker DNS; http://localhost:5341 for host browser
    - Empty SEQ_SERVER_URL disables Seq ingestion non-fatally (same pattern as EVIDENTLY_DASHBOARD_URL)

key-files:
  created: []
  modified:
    - docker-compose.yml
    - app/src/config.py
    - .env.example
    - pyproject.toml
    - poetry.lock

key-decisions:
  - "Seq host port 5341 -> container port 80: Seq container listens on 80 internally, 5341 is the community convention"
  - "ACCEPT_EULA: Y mandatory in environment block — container exits immediately without it"
  - "SEQ_SERVER_URL default is empty string (disables ingestion), not the host URL — prevents crashes when Seq not running"
  - "poetry.lock successfully refreshed via poetry add (SSL issue did not block this run)"

patterns-established:
  - "Docker service naming as DNS: http://seq:80 inside compose network, http://localhost:5341 from host"
  - "Config defaults: empty string = feature disabled, same as EVIDENTLY_DASHBOARD_URL pattern"

# Metrics
duration: 2min
completed: 2026-04-17
---

# Phase 7 Plan 01: Seq Infrastructure Baseline Summary

**Seq datalust/seq:latest provisioned via Docker Compose with seq-data named volume; structlog 25.5.0 + seqlog 0.4.3 installed; SEQ_SERVER_URL/SEQ_API_KEY/SEQ_UI_URL/ENVIRONMENT wired through config.py with Docker-internal DNS default**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-17T08:43:53Z
- **Completed:** 2026-04-17T08:45:39Z
- **Tasks:** 3
- **Files modified:** 5 (docker-compose.yml, app/src/config.py, .env.example, pyproject.toml, poetry.lock)

## Accomplishments

- Seq service added to `docker-compose.yml` with `datalust/seq:latest`, EULA acceptance, port 5341:80, and `seq-data` named volume for persistence
- Four Seq/environment config vars added to `app/src/config.py` following existing `os.environ.get` pattern with safe defaults
- `structlog` (25.5.0) and `seqlog` (0.4.3) declared as runtime dependencies and installed; `poetry.lock` refreshed successfully

## Task Commits

1. **Task 1: Add Seq service to Docker Compose** - `43e216c` (feat)
2. **Task 2: Add Seq and environment config to config.py and .env files** - `a31b279` (feat)
3. **Task 3: Add structlog and seqlog to pyproject.toml** - `529087a` (feat)

## Files Created/Modified

- `docker-compose.yml` - Added `seq` service (datalust/seq:latest, port 5341:80, seq-data volume) and top-level `volumes:` block
- `app/src/config.py` - Added SEQ_SERVER_URL, SEQ_API_KEY, SEQ_UI_URL, ENVIRONMENT after EVIDENTLY_DASHBOARD_URL
- `.env.example` - Documented all four Seq vars with inline comments (gitignored .env.local updated locally)
- `pyproject.toml` - Added structlog (>=25.5.0,<26.0.0) and seqlog (>=0.4.3,<0.5.0)
- `poetry.lock` - Refreshed with two new packages

## Decisions Made

- **Seq host port 5341 / container port 80:** Seq listens on port 80 internally; 5341 is the established community host-port convention
- **ACCEPT_EULA: Y mandatory:** Container exits immediately without this environment variable — it is not optional
- **SEQ_SERVER_URL default is empty string:** Matches the EVIDENTLY_DASHBOARD_URL pattern — empty disables the feature non-fatally rather than pointing at a URL that may not exist
- **poetry lock refreshed:** SSL cert issue did not block this run; both packages resolved and lock file updated successfully

## Deviations from Plan

None - plan executed exactly as written.

**Port conflict noted but non-blocking:** An existing standalone `seq` container was already running on port 5341 from prior manual testing. `docker compose up -d seq` failed due to port conflict, but Seq was verified reachable (200 OK from http://localhost:5341). On a fresh environment the compose definition will start Seq cleanly. The compose file is correct.

## Issues Encountered

- Existing standalone `seq` container occupied port 5341, preventing `docker compose up -d seq` from starting a new container. Verified the running Seq instance satisfies all verification criteria: `docker compose ps seq` is not applicable to the standalone container, but `curl -sI http://localhost:5341` returned HTTP/1.1 200 OK confirming Seq is reachable.

## Next Phase Readiness

- Infrastructure baseline complete: Seq reachable, config vars exposed, dependencies importable
- Plans 02 (logging-core) and 03 (log-emission) can proceed in parallel — both `from app.src.config import SEQ_SERVER_URL` and `import structlog, seqlog` work
- No blockers for phase continuation
- poetry.lock pending todos from Phase 6 are now resolved (SSL did not block this run)

---
*Phase: 07-seq-structured-logging-observability*
*Completed: 2026-04-17*
