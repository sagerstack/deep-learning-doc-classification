---
phase: 05-demo-application-classification-page
plan: 02
subsystem: ui
tags: [fastapi, jinja2, tailwind, htmx, alpine, md3]

requires:
  - phase: 05-demo-application-classification-page
    provides: research and plan for demo app architecture

provides:
  - FastAPI application skeleton with lifespan pattern
  - Stitch-derived Jinja2 base template with MD3 design tokens
  - App config with RVL-CDIP labels and model checkpoint paths
  - Route stubs (GET /, POST /classify)
  - Test conftest with FastAPI TestClient fixture

affects: [05-03-model-loading, 05-04-inference-service, 05-05-classification-routes]

tech-stack:
  added: [fastapi, uvicorn, jinja2, jinja2-fragments, python-multipart, pytest, htmx, alpine.js, tailwind-cdn]
  patterns: [fastapi-lifespan, jinja2-blocks-for-htmx-fragments, md3-design-tokens]

key-files:
  created:
    - app/src/main.py
    - app/src/config.py
    - app/src/routes/classify.py
    - app/src/templates/base.html
    - app/src/static/css/app.css
    - app/tests/conftest.py
  modified:
    - pyproject.toml
    - poetry.lock

key-decisions:
  - "Starlette 1.0 TemplateResponse uses keyword args (request=, name=) not positional"
  - "Tailwind CDN with inline config for MD3 color tokens rather than build step"
  - "HTMX form with hx-post for classify, targeting #results div"

patterns-established:
  - "MD3 design tokens in Tailwind config matching Stitch prototype"
  - "Jinja2 block structure: upload_zone, results_section, cnn_features, graph_construction, model_predictions, detailed_analysis"
  - "FastAPI route pattern: Jinja2Templates with request= keyword arg for Starlette 1.0+"

duration: 4min
completed: 2026-04-07
---

# Phase 05 Plan 02: FastAPI App Scaffolding Summary

**FastAPI app with Stitch-derived MD3 template, HTMX upload form, route stubs, and 6-model config at localhost:8000**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-07T07:34:47Z
- **Completed:** 2026-04-07T07:38:11Z
- **Tasks:** 2
- **Files modified:** 15

## Accomplishments
- Installed web stack (FastAPI, uvicorn, Jinja2, HTMX, Alpine.js) with working poetry lock
- Created full app directory structure with routes, services, templates, static dirs
- Converted Stitch HTML prototype into Jinja2 base template preserving all MD3 design tokens
- App config defines all 6 model checkpoint paths and 16 RVL-CDIP class labels

## Task Commits

1. **Task 1: Install web dependencies and create app directory structure** - `3a4800e` (chore)
2. **Task 2: Create app config, main.py, route stubs, base template, and test conftest** - `30c511d` (feat)

## Files Created/Modified
- `app/src/main.py` - FastAPI app factory with lifespan, CORS, static mount, router
- `app/src/config.py` - RVL-CDIP labels, model checkpoint paths, directory paths
- `app/src/routes/classify.py` - GET / (template render) and POST /classify (501 stub)
- `app/src/templates/base.html` - Stitch-derived MD3 template with HTMX upload form
- `app/src/static/css/app.css` - Custom font/glass styles from Stitch prototype
- `app/tests/conftest.py` - pytest fixture with FastAPI TestClient
- `pyproject.toml` - Added fastapi, uvicorn, jinja2, jinja2-fragments, python-multipart, pytest

## Decisions Made
- Starlette 1.0 changed TemplateResponse API to keyword arguments -- adapted route code accordingly
- Tailwind CSS served from CDN with inline config (no build step needed for academic demo)
- HTMX 2.0 and Alpine.js 3.x loaded from CDN for progressive enhancement

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Starlette 1.0 TemplateResponse API change**
- **Found during:** Task 2 (server verification)
- **Issue:** Starlette 1.0 changed TemplateResponse to use keyword args; positional dict caused "unhashable type: dict" error
- **Fix:** Changed to `TemplateResponse(request=request, name="base.html")`
- **Files modified:** app/src/routes/classify.py
- **Verification:** GET / returns 200 with correct HTML
- **Committed in:** 30c511d

**2. [Rule 3 - Blocking] Installed missing pytest dev dependency**
- **Found during:** Task 2 (conftest verification)
- **Issue:** pytest not installed, conftest import failed
- **Fix:** Added pytest via `poetry add --group dev pytest`
- **Files modified:** pyproject.toml, poetry.lock
- **Verification:** `from app.tests.conftest import *` succeeds
- **Committed in:** 30c511d

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for app to function. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- App skeleton ready for model loading service (plan 03)
- Template block structure ready for results content (plans 04-05)
- Config has all model paths defined; checkpoint files must exist at those paths

---
*Phase: 05-demo-application-classification-page*
*Completed: 2026-04-07*
