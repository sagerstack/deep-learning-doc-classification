"""Single-image inference pipeline: image -> features -> graphs -> multi-model predictions."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights
from torchvision import transforms as T

from app.src.config import RVL_CDIP_LABELS
from app.src.graph import add_positional_encoding_2d, build_grid_edge_index

log = structlog.get_logger()


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
    boc_features: Optional[torch.Tensor] = None  # [49, 70]
    grid_edge_index: Optional[torch.Tensor] = None
    knn_edge_index: Optional[torch.Tensor] = None
    graph_stats: dict = field(default_factory=dict)
    total_time_ms: float = 0.0
    feature_extraction_time_ms: float = 0.0
    graph_construction_time_ms: float = 0.0
    # Multimodal GAT graph metadata
    gat_boxes: list = field(default_factory=list)          # [(x, y, w, h, class_id), ...]
    gat_layout_classes: list = field(default_factory=list) # [class_id, ...]
    gat_edge_index: Optional[torch.Tensor] = None
    gat_ocr_texts: list = field(default_factory=list)      # OCR text per region node
    gat_img_w: int = 0
    gat_img_h: int = 0


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


def _compute_text_density(image: Image.Image, detector, device: torch.device) -> Optional[torch.Tensor]:
    """Run text density extraction using a pre-loaded doctr detector.

    Args:
        detector: doctr DetectionPredictor loaded at startup, or None if unavailable.
    """
    if detector is None:
        return None
    try:
        from src.text_features import extract_text_density
        return extract_text_density(image, detector, device)
    except Exception:
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
    elif spec.needs_global_feat:
        logits = model(x, ei, b, gf)
    else:
        logits = model(x, ei, b)

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

    # Preprocess image — must match training: Grayscale(3) before Normalize
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
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

    log.info(
        "graph.built",
        event_type="graph.built",
        graph_latency_ms=result.graph_construction_time_ms,
        feature_extraction_ms=result.feature_extraction_time_ms,
        num_nodes=result.graph_stats.get("num_nodes", 0),
        knn_edges=result.graph_stats.get("knn_edges", 0),
        grid_edges=result.graph_stats.get("grid_edges", 0),
    )

    # Optionally compute BoC features
    boc_features = _compute_boc_from_image(image)
    result.boc_features = boc_features

    # Optionally compute text density using pre-loaded detector (for visualization only)
    text_detector = registry.get("text_detector")
    result.text_density = _compute_text_density(image, text_detector, device)

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

    # --- Multimodal GAT ---
    multimodalGatModel = registry.get("multimodal_gat_model")
    if multimodalGatModel is not None:
        try:
            from app.src.services.multimodal_gat import runMultimodalGatInference

            probs, graphMeta, elapsedMs = runMultimodalGatInference(
                image, multimodalGatModel, feature_extractor, device
            )
            confidence, predIdx = probs.max(dim=0)
            result.results.append(
                InferenceResult(
                    model_name="multimodal_gat",
                    display_name="Multimodal GAT (YOLO + OCR)",
                    model_type="GNN+OCR",
                    predicted_class=RVL_CDIP_LABELS[predIdx.item()],
                    predicted_index=predIdx.item(),
                    confidence=confidence.item(),
                    probabilities=probs.tolist(),
                    inference_time_ms=elapsedMs,
                )
            )
            result.gat_boxes = graphMeta["boxes"]
            result.gat_layout_classes = graphMeta["layout_classes"]
            result.gat_edge_index = graphMeta["edge_index"]
            result.gat_ocr_texts = graphMeta["ocr_texts"]
            result.gat_img_w = graphMeta["img_w"]
            result.gat_img_h = graphMeta["img_h"]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Multimodal GAT inference failed: %s", exc)

    # Sort by confidence descending
    result.results.sort(key=lambda r: r.confidence, reverse=True)
    result.total_time_ms = (time.perf_counter() - pipeline_start) * 1000

    return result
