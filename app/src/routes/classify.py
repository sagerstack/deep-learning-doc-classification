import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from jinja2_fragments.fastapi import Jinja2Blocks
from PIL import Image
import io

from app.src.config import RVL_CDIP_LABELS, SAMPLES_DIR, TEMPLATES_DIR
from app.src.services.inference import run_inference_pipeline
from app.src.services.visualization import (
    generate_16class_bar_chart,
    generate_activation_heatmap,
    generate_graph_overlay,
    generate_graph_svg,
    generate_node_importance_html,
    generate_original_image_base64,
    generate_probability_bars_html,
    generate_text_density_html,
)

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Blocks(directory=str(TEMPLATES_DIR))

SAMPLE_CATEGORIES = {
    "in-dist": "In-Distribution (RVL-CDIP)",
    "ood": "Out-of-Distribution (RVL-CDIP-N)",
}


def _load_sample_categories() -> list[dict]:
    categories = []
    for folder, display_name in SAMPLE_CATEGORIES.items():
        folder_path = SAMPLES_DIR / folder
        if not folder_path.exists():
            continue
        samples = []
        for path in sorted(folder_path.iterdir()):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff"):
                label = path.stem.replace("_", " ").title()
                samples.append({
                    "filename": f"{folder}/{path.name}",
                    "label": label,
                })
        if samples:
            categories.append({
                "key": folder,
                "name": display_name,
                "samples": samples,
            })
    return categories


def _load_image_from_upload(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def _load_image_from_sample(sample_name: str) -> Image.Image:
    sample_path = SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_name}")
    return Image.open(sample_path).convert("RGB")


def _build_result_context(request: Request, image: Image.Image) -> dict:
    registry = request.app.state.registry
    device = request.app.state.device

    pipeline = run_inference_pipeline(image, registry, device)

    original_b64 = generate_original_image_base64(image)
    heatmap_b64 = generate_activation_heatmap(pipeline.layer4_features, image)
    text_density_html = generate_text_density_html(pipeline.text_density)
    grid_svg = generate_graph_svg(pipeline.grid_edge_index)
    knn_svg = generate_graph_svg(pipeline.knn_edge_index, edge_color="#24389c", node_color="#24389c")
    grid_overlay_b64 = generate_graph_overlay(
        image, pipeline.grid_edge_index, title="Spatial Grid (k=8)", edge_color="#2563eb",
    )
    knn_overlay_b64 = generate_graph_overlay(
        image, pipeline.knn_edge_index, title="Feature k-NN (k=8)", edge_color="#dc2626",
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
        "request": request,
        "has_results": True,
        "original_image_b64": original_b64,
        "heatmap_b64": heatmap_b64,
        "text_density_html": text_density_html,
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
        "sample_categories": _load_sample_categories(),
    }


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context={
            "request": request,
            "has_results": False,
            "sample_categories": _load_sample_categories(),
        },
    )


@router.post("/classify")
async def classify(
    request: Request,
    file: UploadFile | None = None,
    sample: str | None = Form(default=None),
):
    if sample:
        try:
            image = _load_image_from_sample(sample)
        except FileNotFoundError:
            return HTMLResponse(
                content='<div class="text-error font-label text-sm p-4">Sample not found. Please select a valid sample.</div>',
                status_code=404,
            )
    elif file and file.filename:
        image = _load_image_from_upload(file)
    else:
        return HTMLResponse(
            content='<div class="text-error font-label text-sm p-4">No image provided. Upload a file or select a sample.</div>',
            status_code=400,
        )

    context = _build_result_context(request, image)

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
