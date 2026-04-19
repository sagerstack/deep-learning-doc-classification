"""Monitoring routes: redirect to Evidently dashboard and capture label feedback."""

import logging
import sqlite3

import structlog
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from app.src.config import EVIDENTLY_DASHBOARD_URL, MONITORING_DB_PATH

logger = logging.getLogger(__name__)
log = structlog.get_logger()

router = APIRouter()

_NOT_CONFIGURED_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Model Performance</title>
<style>
  body { font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: #f8f9fa; color: #191c1d; }
  .card { background: #fff; border: 1px solid #e1e3e4; border-radius: 8px; padding: 2rem 3rem; max-width: 480px; text-align: center; }
  h1 { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem; }
  p { font-size: 0.875rem; color: #454652; }
  code { background: #f3f4f5; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; }
</style>
</head>
<body>
<div class="card">
  <h1>Model Performance Dashboard</h1>
  <p>The Evidently dashboard is not configured yet.</p>
  <p>Set <code>EVIDENTLY_DASHBOARD_URL</code> in your <code>.env.local</code> to enable this redirect.</p>
</div>
</body>
</html>
"""

_CONFIRM_CORRECT_HTML = """
<div class="flex items-center gap-1.5 text-green-600">
    <span class="material-symbols-outlined text-green-600 text-lg" style="font-variation-settings: 'FILL' 1;">check_circle</span>
    <span class="font-label text-xs font-bold uppercase tracking-wider">Recorded</span>
</div>
"""

_CONFIRM_INCORRECT_HTML = """
<div class="flex items-center gap-1.5 text-slate-400">
    <span class="material-symbols-outlined text-slate-400 text-lg" style="font-variation-settings: 'FILL' 1;">cancel</span>
    <span class="font-label text-xs font-bold uppercase tracking-wider">Noted</span>
</div>
"""


@router.get("/model-performance")
async def model_performance():
    """Redirect to the Evidently dashboard, or show a fallback if not configured."""
    if EVIDENTLY_DASHBOARD_URL:
        return RedirectResponse(url=EVIDENTLY_DASHBOARD_URL)
    return HTMLResponse(content=_NOT_CONFIGURED_HTML, status_code=200)


@router.post("/label")
async def label_prediction(
    request_id: str = Form(...),
    model_id: str = Form(...),
    correct: str = Form(...),
    sample_set: str = Form(default=""),
    true_label: str = Form(default=""),
    predicted_class: str = Form(default=""),
):
    """Capture thumbs-up/down feedback and write target label to inference_events.

    On correct=true: sets target = predicted_label for the matching row.
    On correct=false: no DB write (target remains NULL).
    Emits a label.feedback Seq event so the accuracy dashboard reflects overrides.
    """
    if not request_id.strip() or not model_id.strip():
        return HTMLResponse(
            content='<span class="font-label text-xs text-red-400">Invalid input</span>',
            status_code=400,
        )

    is_correct = correct.lower() in ("true", "1", "yes")

    if is_correct:
        try:
            with sqlite3.connect(str(MONITORING_DB_PATH)) as conn:
                conn.execute(
                    "UPDATE inference_events SET target = predicted_label "
                    "WHERE request_id = ? AND model_id = ?",
                    (request_id, model_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Label write failed (non-fatal): %s", exc)

    log.info(
        "label.feedback",
        event_type="label.feedback",
        request_id=request_id,
        model_id=model_id,
        sample_set=sample_set or None,
        true_label=true_label or None,
        predicted_class=predicted_class or None,
        is_correct=is_correct,
        auto_correct_int=1 if is_correct else 0,
    )

    confirmation = _CONFIRM_CORRECT_HTML if is_correct else _CONFIRM_INCORRECT_HTML
    return HTMLResponse(content=confirmation, status_code=200)
