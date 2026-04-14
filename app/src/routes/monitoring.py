"""Monitoring route: redirect to the configured Evidently dashboard URL."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

from app.src.config import EVIDENTLY_DASHBOARD_URL

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


@router.get("/model-performance")
async def model_performance():
    """Redirect to the Evidently dashboard, or show a fallback if not configured."""
    if EVIDENTLY_DASHBOARD_URL:
        return RedirectResponse(url=EVIDENTLY_DASHBOARD_URL)
    return HTMLResponse(content=_NOT_CONFIGURED_HTML, status_code=200)
