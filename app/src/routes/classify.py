from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.src.config import TEMPLATES_DIR

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="base.html",
    )


@router.post("/classify")
async def classify(request: Request):
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented — inference added in plan 05"},
    )
