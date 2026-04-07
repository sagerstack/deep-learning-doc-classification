"""Single-image inference pipeline: image -> features -> graphs -> multi-model predictions."""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights

from app.src.config import RVL_CDIP_LABELS
from src.graph import (
    add_positional_encoding_2d,
    build_grid_edge_index,
    feature_map_to_graph_feature_knn,
)


@dataclass
class InferenceResult:
    model_name: str
    display_name: str
    model_type: str
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: list[float]
    inference_time_ms: float


@dataclass
class PipelineResult:
    results: list[InferenceResult] = field(default_factory=list)
    layer4_features: Optional[torch.Tensor] = None  # [2048, 7, 7]
    text_density: Optional[torch.Tensor] = None  # [7, 7]
    grid_edge_index: Optional[torch.Tensor] = None
    knn_edge_index: Optional[torch.Tensor] = None
    graph_stats: dict = field(default_factory=dict)
    total_time_ms: float = 0.0
    feature_extraction_time_ms: float = 0.0
    graph_construction_time_ms: float = 0.0


def _build_knn_edges(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Build feature-space k-NN edge index from node features [N, C]."""
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = x_norm @ x_norm.T
    sim.fill_diagonal_(-float("inf"))
    _, topk_indices = sim.topk(k, dim=1)

    sources = torch.arange(x.shape[0]).unsqueeze(1).expand_as(topk_indices).flatten()
    targets = topk_indices.flatten()
    return torch.stack([sources, targets], dim=0)


def _compute_boc_from_image(image: Image.Image) -> Optional[torch.Tensor]:
    """Try to compute BoC features from an image using Tesseract. Returns None if unavailable."""
    try:
        from src.ocr_features import compute_boc_features, run_tesseract_single

        img_rgb = image.convert("RGB") if image.mode != "RGB" else image
        img_w, img_h = img_rgb.size
        words = run_tesseract_single(img_rgb)
        boc = compute_boc_features(words, img_w, img_h)
        return boc  # [49, 70]
    except (ImportError, OSError, RuntimeError):
        return None


def _compute_text_density(image: Image.Image, device: torch.device) -> Optional[torch.Tensor]:
    """Try to extract text density heatmap using doctr. Returns None if unavailable."""
    try:
        from src.text_features import create_text_detector, extract_text_density

        detector = create_text_detector(type("Config", (), {"device": device})())
        density = extract_text_density(image, detector, device)
        return density  # [7, 7]
    except (ImportError, OSError, RuntimeError):
        return None


def _run_single_model(
    spec,
    model: torch.nn.Module,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    global_feat: torch.Tensor,
    boc_features: Optional[torch.Tensor],
    device: torch.device,
) -> InferenceResult:
    """Run a single GNN model and return an InferenceResult."""
    t0 = time.perf_counter()

    x = node_features.to(device)
    ei = edge_index.to(device)
    b = batch.to(device)
    gf = global_feat.unsqueeze(0).to(device)  # [1, 2048]

    if spec.graph_type == "gated_boc" and boc_features is not None:
        x_boc = boc_features.to(device)
        logits = model(x, ei, b, gf, x_boc)
    else:
        logits = model(x, ei, b, gf)

    probs = F.softmax(logits, dim=1).squeeze(0)
    confidence, pred_idx = probs.max(dim=0)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return InferenceResult(
        model_name=spec.name,
        display_name=spec.display_name,
        model_type=spec.model_type,
        predicted_class=RVL_CDIP_LABELS[pred_idx.item()],
        predicted_index=pred_idx.item(),
        confidence=confidence.item(),
        probabilities=probs.tolist(),
        inference_time_ms=elapsed_ms,
    )


@torch.no_grad()
def run_inference_pipeline(
    image: Image.Image,
    registry: dict,
    device: torch.device,
) -> PipelineResult:
    """Run the full inference pipeline on a single image.

    Args:
        image: PIL Image (any mode/size)
        registry: Dict from load_all_models()
        device: torch.device for inference

    Returns:
        PipelineResult with predictions from all loaded models, features, and timing.
    """
    pipeline_start = time.perf_counter()
    result = PipelineResult()

    feature_extractor = registry["feature_extractor"]
    resnet_classifier = registry["resnet50_classifier"]

    # Preprocess image
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    img_rgb = image.convert("RGB") if image.mode != "RGB" else image
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Feature extraction
    t_feat = time.perf_counter()
    feat_out = feature_extractor(input_tensor)
    layer4 = feat_out["layer4"].squeeze(0)  # [2048, 7, 7]
    avgpool = feat_out["avgpool"].squeeze(-1).squeeze(-1).squeeze(0)  # [2048]
    result.layer4_features = layer4.cpu()
    result.feature_extraction_time_ms = (time.perf_counter() - t_feat) * 1000

    # Graph construction
    t_graph = time.perf_counter()

    # Grid edges (for visualization)
    grid_edge_index = build_grid_edge_index(7, 7, k=8)
    result.grid_edge_index = grid_edge_index

    # Node features: reshape layer4 to [49, 2048]
    node_features_raw = layer4.reshape(2048, 49).T  # [49, 2048]

    # Feature-kNN edges
    knn_edge_index = _build_knn_edges(node_features_raw, k=8)
    result.knn_edge_index = knn_edge_index.cpu()

    # Node features with PE for models that need it
    node_features_pe = add_positional_encoding_2d(node_features_raw)  # [49, 2050]

    result.graph_construction_time_ms = (time.perf_counter() - t_graph) * 1000

    # Graph stats
    result.graph_stats = {
        "num_nodes": 49,
        "grid_edges": grid_edge_index.shape[1],
        "knn_edges": knn_edge_index.shape[1],
        "avg_degree": knn_edge_index.shape[1] / 49,
    }

    # Optionally compute BoC features
    boc_features = _compute_boc_from_image(image)

    # Optionally compute text density (for visualization only)
    result.text_density = _compute_text_density(image, device)

    # Batch vector for single graph (all nodes belong to graph 0)
    batch_vec = torch.zeros(49, dtype=torch.long)

    # --- CNN Baseline ---
    t_cnn = time.perf_counter()
    cnn_logits = resnet_classifier(input_tensor)
    cnn_probs = F.softmax(cnn_logits, dim=1).squeeze(0)
    cnn_conf, cnn_pred = cnn_probs.max(dim=0)
    cnn_ms = (time.perf_counter() - t_cnn) * 1000

    result.results.append(
        InferenceResult(
            model_name="cnn_baseline",
            display_name="CNN Baseline (ResNet-50)",
            model_type="CNN",
            predicted_class=RVL_CDIP_LABELS[cnn_pred.item()],
            predicted_index=cnn_pred.item(),
            confidence=cnn_conf.item(),
            probabilities=cnn_probs.tolist(),
            inference_time_ms=cnn_ms,
        )
    )

    # --- GNN Models ---
    for spec_name, (spec, model) in registry["models"].items():
        # Skip OCR-dependent models if BoC not available
        if spec.needs_boc and boc_features is None:
            result.results.append(
                InferenceResult(
                    model_name=spec.name,
                    display_name=spec.display_name,
                    model_type=spec.model_type,
                    predicted_class="(OCR unavailable)",
                    predicted_index=-1,
                    confidence=0.0,
                    probabilities=[0.0] * len(RVL_CDIP_LABELS),
                    inference_time_ms=0.0,
                )
            )
            continue

        # Choose node features based on model spec
        # Start with raw or PE features
        if spec.needs_pe:
            node_feat = node_features_pe  # [49, 2050]
        else:
            node_feat = node_features_raw  # [49, 2048]

        # Append BoC for models that concatenate it into node features (exp25)
        if spec.name == "boc_graphsage" and boc_features is not None:
            node_feat = torch.cat([node_feat, boc_features.to(node_feat.device)], dim=1)

        # Choose edge index based on graph type
        if spec.graph_type in ("feature_knn", "gated_boc"):
            edge_idx = knn_edge_index
        else:
            edge_idx = grid_edge_index

        inf_result = _run_single_model(
            spec=spec,
            model=model,
            node_features=node_feat,
            edge_index=edge_idx,
            batch=batch_vec,
            global_feat=avgpool,
            boc_features=boc_features,
            device=device,
        )
        result.results.append(inf_result)

    # Sort by confidence descending
    result.results.sort(key=lambda r: r.confidence, reverse=True)
    result.total_time_ms = (time.perf_counter() - pipeline_start) * 1000

    return result
