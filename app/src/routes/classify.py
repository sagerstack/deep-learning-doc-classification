import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import structlog

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from jinja2_fragments.fastapi import Jinja2Blocks
from PIL import Image

from app.src.config import MONITORING_DB_PATH, RVL_CDIP_LABELS, SAMPLES_DIR, SEQ_UI_URL, TEMPLATES_DIR
from app.src.monitoring.schema import build_inference_events
from app.src.monitoring.store import log_inference_events
from app.src.services.inference import run_inference_pipeline
from app.src.services.multimodal_gat import LAYOUT_CLASS_COLORS, LAYOUT_CLASS_NAMES
from app.src.services.visualization import (
    generate_16class_bar_chart,
    generate_activation_heatmap,
    generate_boc_density_html,
    generate_document_graph_plotly,
    generate_graph_overlay,
    generate_graph_svg,
    generate_node_importance_html,
    generate_original_image_base64,
    generate_probability_bars_html,
    generate_text_density_html,
)

logger = logging.getLogger(__name__)
log = structlog.get_logger()

router = APIRouter()
templates = Jinja2Blocks(directory=str(TEMPLATES_DIR))

NAV_ITEMS = (
    {"key": "demo", "label": "Demo", "href": "/"},
    {"key": "models", "label": "Models", "href": "/models"},
    {"key": "experiments", "label": "Experiments", "href": "/experiments"},
    {"key": "drift-monitoring", "label": "Drift Monitoring", "href": "/model-performance", "target": "_blank"},
    {"key": "observability", "label": "Observability", "href": SEQ_UI_URL, "target": "_blank"},
)

SAMPLE_CATEGORIES = (
    {"key": "in-dist", "name": "In-Dist"},
    {"key": "oo-dist", "name": "OO-Dist"},
    {"key": "oo-dom", "name": "OO-Dom"},
)


def _build_navigation(active_key: str) -> list[dict]:
    return [
        {
            **item,
            "is_active": item["key"] == active_key,
        }
        for item in NAV_ITEMS
    ]


def _load_sample_categories() -> list[dict]:
    categories = []
    for category in SAMPLE_CATEGORIES:
        folder_path = SAMPLES_DIR / category["key"]
        if not folder_path.exists():
            continue
        samples = []
        for path in sorted(folder_path.iterdir()):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff"):
                label = path.stem.replace("_", " ").title()
                samples.append({
                    "filename": f"{category['key']}/{path.name}",
                    "label": label,
                })
        if samples:
            categories.append({
                "key": category["key"],
                "name": category["name"],
                "samples": samples,
            })
    return categories


def _build_base_context(request: Request) -> dict:
    return {
        "request": request,
        "nav_items": _build_navigation("demo"),
        "sample_categories": _load_sample_categories(),
    }


def _load_image_from_upload(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def _load_image_from_sample(sample_name: str) -> Image.Image:
    sample_path = SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_name}")
    return Image.open(sample_path).convert("RGB")


def _infer_sample_set(sample_name: str | None) -> str:
    if not sample_name:
        return "upload"
    category = sample_name.split("/", 1)[0] if "/" in sample_name else ""
    if category == "in-dist":
        return "in_dist"
    if category == "oo-dist":
        return "oo_dist"
    if category == "oo-dom":
        return "oo_dom"
    return "upload"


def _build_result_context(
    request: Request,
    image: Image.Image,
    *,
    sample_type: str = "upload",
    sample_name: str = "upload",
    request_id: str,
) -> dict:
    registry = request.app.state.registry
    device = request.app.state.device

    pipeline = run_inference_pipeline(image, registry, device)

    # --- Monitoring: persist one row per model prediction ---
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        events = build_inference_events(
            request_id=request_id,
            timestamp=timestamp,
            sample_type=sample_type,
            sample_name=sample_name,
            image_width=image.width,
            image_height=image.height,
            image_mode=image.mode,
            results=pipeline.results,
            total_time_ms=pipeline.total_time_ms,
            feature_time_ms=pipeline.feature_extraction_time_ms,
            graph_time_ms=pipeline.graph_construction_time_ms,
            ocr_available=pipeline.boc_features is not None,
            text_density_available=pipeline.text_density is not None,
        )
        log_inference_events(events, MONITORING_DB_PATH)
    except Exception as exc:
        logger.warning("Monitoring log failed (non-fatal): %s", exc)

    original_b64 = generate_original_image_base64(image)
    heatmap_b64 = generate_activation_heatmap(pipeline.layer4_features, image)
    text_density_html = generate_text_density_html(pipeline.text_density)
    boc_density_html = generate_boc_density_html(pipeline.boc_features)
    grid_svg = generate_graph_svg(pipeline.grid_edge_index)
    knn_svg = generate_graph_svg(pipeline.knn_edge_index, edge_color="#24389c", node_color="#24389c")
    grid_overlay_b64 = generate_graph_overlay(
        image, pipeline.grid_edge_index, title="Spatial Grid (k=8)", edge_color="#2563eb",
    )
    knn_overlay_b64 = generate_graph_overlay(
        image, pipeline.knn_edge_index, title="Feature k-NN (k=8)", edge_color="#dc2626",
    )

    # GAT multimodal graph visualization
    gat_doc_graph_html = None
    if pipeline.gat_boxes:
        gat_doc_graph_html = generate_document_graph_plotly(
            image,
            pipeline.gat_boxes,
            pipeline.gat_edge_index,
            pipeline.gat_ocr_texts,
            LAYOUT_CLASS_COLORS,
            LAYOUT_CLASS_NAMES,
        )

    gat_regions = [
        {
            "class_name": LAYOUT_CLASS_NAMES.get(cls_id, f"class-{cls_id}"),
            "color": LAYOUT_CLASS_COLORS.get(cls_id, "#aaaaaa"),
            "ocr_text": text,
        }
        for cls_id, text in zip(pipeline.gat_layout_classes, pipeline.gat_ocr_texts)
    ]

    gat_legend = sorted(
        {cls_id: (cls_id, LAYOUT_CLASS_NAMES[cls_id], LAYOUT_CLASS_COLORS[cls_id])
         for cls_id in pipeline.gat_layout_classes
         if cls_id in LAYOUT_CLASS_NAMES}.values()
    )

    best_result = pipeline.results[0] if pipeline.results else None
    top3_bars_html = ""
    if best_result:
        top3_bars_html = generate_probability_bars_html(
            best_result.probabilities, RVL_CDIP_LABELS, top_n=3
        )

    model_details = []
    for r in pipeline.results:
        bar_chart_b64 = generate_16class_bar_chart(
            r.probabilities, RVL_CDIP_LABELS, r.predicted_index
        )
        prob_bars_html = generate_probability_bars_html(
            r.probabilities, RVL_CDIP_LABELS, top_n=3
        )
        node_importance = generate_node_importance_html(None)
        model_details.append({
            "result": r,
            "bar_chart_b64": bar_chart_b64,
            "probability_bars_html": prob_bars_html,
            "node_importance_html": node_importance,
        })

    return {
        **_build_base_context(request),
        "has_results": True,
        "request_id": request_id,
        "original_image_b64": original_b64,
        "heatmap_b64": heatmap_b64,
        "text_density_html": text_density_html,
        "boc_density_html": boc_density_html,
        "grid_svg": grid_svg,
        "knn_svg": knn_svg,
        "grid_overlay_b64": grid_overlay_b64,
        "knn_overlay_b64": knn_overlay_b64,
        "graph_stats": pipeline.graph_stats,
        "results": pipeline.results,
        "labels": RVL_CDIP_LABELS,
        "model_details": model_details,
        "top3_bars_html": top3_bars_html,
        "total_time_ms": pipeline.total_time_ms,
        "feature_time_ms": pipeline.feature_extraction_time_ms,
        "graph_time_ms": pipeline.graph_construction_time_ms,
        # GAT multimodal
        "gat_doc_graph_html": gat_doc_graph_html,
        "gat_boxes": pipeline.gat_boxes,
        "gat_regions": gat_regions,
        "gat_legend": gat_legend,
    }


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context={
            **_build_base_context(request),
            "has_results": False,
        },
    )


@router.post("/classify")
async def classify(
    request: Request,
    file: UploadFile | None = None,
    sample: str | None = Form(default=None),
):
    # Retrieve middleware request_id from contextvars (bound by LoggingMiddleware)
    middleware_ctx = structlog.contextvars.get_contextvars()
    request_id = middleware_ctx.get("request_id")
    if not request_id:
        import uuid as _uuid
        request_id = str(_uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)

    # Determine action fields
    if sample:
        action_type = "sample_click"
        sample_label = sample
        sample_set = _infer_sample_set(sample)
    elif file and file.filename:
        action_type = "file_upload"
        sample_label = file.filename
        sample_set = "upload"
    else:
        action_type = "unknown"
        sample_label = None
        sample_set = None

    # Bind action fields into contextvars (inherited by all downstream log events)
    structlog.contextvars.bind_contextvars(
        action_type=action_type,
        sample_name=sample_label,
        sample_set=sample_set,
    )

    # Emit request.received (image dimensions not yet known)
    log.info(
        "request.received",
        event_type="request.received",
        action_type=action_type,
        sample_name=sample_label,
        sample_set=sample_set,
    )

    if sample:
        try:
            image = _load_image_from_sample(sample)
        except FileNotFoundError:
            log.error(
                "request.failed",
                event_type="request.failed",
                error=f"Sample not found: {sample}",
                reason="sample_not_found",
            )
            return HTMLResponse(
                content='<div class="text-error font-label text-sm p-4">Sample not found. Please select a valid sample.</div>',
                status_code=404,
            )
        structlog.contextvars.bind_contextvars(
            image_width=image.width,
            image_height=image.height,
        )
        try:
            context = _build_result_context(
                request, image, sample_type="sample", sample_name=sample, request_id=request_id
            )
        except Exception as exc:
            log.error(
                "request.failed",
                event_type="request.failed",
                error=str(exc),
                exc_info=True,
            )
            raise
    elif file and file.filename:
        image = _load_image_from_upload(file)
        structlog.contextvars.bind_contextvars(
            image_width=image.width,
            image_height=image.height,
        )
        try:
            context = _build_result_context(
                request, image, sample_type="upload", sample_name=file.filename or "upload", request_id=request_id
            )
        except Exception as exc:
            log.error(
                "request.failed",
                event_type="request.failed",
                error=str(exc),
                exc_info=True,
            )
            raise
    else:
        log.error(
            "request.failed",
            event_type="request.failed",
            error="No image provided",
            reason="no_image",
        )
        return HTMLResponse(
            content='<div class="text-error font-label text-sm p-4">No image provided. Upload a file or select a sample.</div>',
            status_code=400,
        )

    best = context.get("results", [None])[0] if context.get("results") else None
    log.info(
        "request.completed",
        event_type="request.completed",
        total_latency_ms=context.get("total_time_ms", 0.0),
        model_count=len(context.get("results", [])),
        best_model_id=(best.model_name if best else None),
        best_predicted_class=(best.predicted_class if best else None),
        best_confidence=(best.confidence if best else None),
    )

    is_htmx = request.headers.get("HX-Request") == "true"

    if is_htmx:
        return templates.TemplateResponse(
            request=request,
            name="base.html",
            context=context,
            block_name="results_section",
        )

    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context=context,
    )
