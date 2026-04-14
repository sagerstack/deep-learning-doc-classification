"""Model registry: ModelSpec definitions and checkpoint loading for all 6 models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from app.src.config import MODEL_DIR
from app.src.model import (
    AttentionPoolFusionSAGE,
    DocumentGAT,
    FusionGAT,
    FusionGraphSAGE,
    GatedBoCGraphSAGE,
    HybridGraphSAGE,
)


@dataclass
class ModelSpec:
    name: str
    display_name: str
    model_type: str  # "CNN" | "GNN" | "GNN+OCR"
    model_class: Optional[Type[nn.Module]]
    constructor_kwargs: dict = field(default_factory=dict)
    checkpoint_path: str = ""
    graph_type: str = "none"  # "none" | "feature_knn" | "gated_boc"
    needs_global_feat: bool = False
    needs_pe: bool = False
    needs_boc: bool = False
    needs_text_density: bool = False


MODEL_SPECS = [
    ModelSpec(
        name="cnn_baseline",
        display_name="CNN Baseline (ResNet-50)",
        model_type="CNN",
        model_class=None,
        checkpoint_path="exp14b_finetuned_resnet50_cnn.pt",
        graph_type="none",
    ),
    ModelSpec(
        name="fusion_graphsage",
        display_name="Fusion GraphSAGE",
        model_type="GNN",
        model_class=FusionGraphSAGE,
        constructor_kwargs={"in_channels": 2048},
        checkpoint_path="exp16_fusion_featknn_graphsage.pt",
        graph_type="feature_knn",
        needs_global_feat=True,
    ),
    ModelSpec(
        name="fusion_gat",
        display_name="Fusion GAT",
        model_type="GNN",
        model_class=FusionGAT,
        constructor_kwargs={"in_channels": 2048},
        checkpoint_path="exp23_gat_fusion.pt",
        graph_type="feature_knn",
        needs_global_feat=True,
    ),
    ModelSpec(
        name="boc_graphsage",
        display_name="BoC GraphSAGE",
        model_type="GNN+OCR",
        model_class=HybridGraphSAGE,
        constructor_kwargs={"node_dim": 2120},
        checkpoint_path="exp25_boc_sage.pt",
        graph_type="feature_knn",
        needs_global_feat=True,
        needs_pe=True,
        needs_boc=True,
    ),
    ModelSpec(
        name="gated_boc_graphsage",
        display_name="Gated BoC GraphSAGE",
        model_type="GNN+OCR",
        model_class=GatedBoCGraphSAGE,
        constructor_kwargs={"cnn_dim": 2050, "boc_dim": 70, "proj_dim": 16},
        checkpoint_path="exp26_gated_boc.pt",
        graph_type="gated_boc",
        needs_global_feat=True,
        needs_pe=True,
        needs_boc=True,
    ),
    ModelSpec(
        name="attn_pool_graphsage",
        display_name="Attention Pool GraphSAGE",
        model_type="GNN",
        model_class=AttentionPoolFusionSAGE,
        constructor_kwargs={},
        checkpoint_path="exp27_attn_pool.pt",
        graph_type="feature_knn",
        needs_global_feat=True,
        needs_pe=True,
    ),
]


def _unwrap_resnet_checkpoint(checkpoint) -> dict:
    """Normalize checkpoint formats to a plain ResNet-50 state dict."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_all_models(device: torch.device) -> dict:
    """Load all model checkpoints and return a structured registry dict.

    Returns:
        {
            "feature_extractor": torchvision feature extractor (layer4 + avgpool),
            "resnet50_classifier": full ResNet-50 with fc head (for CNN baseline),
            "models": {spec.name: (spec, loaded_model) for each GNN},
        }
    """
    # Load CNN baseline checkpoint (ResNet-50 with fc head)
    cnn_checkpoint_path = MODEL_DIR / "exp14b_finetuned_resnet50_cnn.pt"
    checkpoint = torch.load(cnn_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_resnet_checkpoint(checkpoint)

    num_classes = state_dict["fc.weight"].shape[0]

    # Build full ResNet-50 classifier (CNN baseline)
    resnet = resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet.load_state_dict(state_dict)
    resnet.eval()
    resnet = resnet.to(device)

    # Build feature extractor from the same fine-tuned weights
    feature_extractor = create_feature_extractor(
        resnet,
        return_nodes={"layer4": "layer4", "avgpool": "avgpool"},
    )
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor = feature_extractor.to(device)

    # Load GNN models
    gnn_models = {}
    for spec in MODEL_SPECS:
        if spec.model_class is None:
            continue  # CNN baseline handled above

        ckpt_path = MODEL_DIR / spec.checkpoint_path
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found for {spec.display_name}: {ckpt_path}")
            continue

        model = spec.model_class(**spec.constructor_kwargs)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        model = model.to(device)

        gnn_models[spec.name] = (spec, model)

    # Load multimodal GAT
    multimodal_gat_model = None
    multimodal_gat_path = MODEL_DIR / "best_gat_multimodal_k8_L2.pt"
    if multimodal_gat_path.exists():
        multimodal_gat_model = DocumentGAT(
            in_dim=2447,
            hidden_dim=256,
            out_dim=num_classes,
            heads=4,
            num_layers=2,
            dropout=0.2,
            pooling="mean",
        )
        ckpt = torch.load(multimodal_gat_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        multimodal_gat_model.load_state_dict(state_dict)
        multimodal_gat_model.eval()
        multimodal_gat_model = multimodal_gat_model.to(device)
        print(f"Loaded DocumentGAT (multimodal) from {multimodal_gat_path.name}")
    else:
        print(f"Warning: multimodal GAT checkpoint not found at {multimodal_gat_path}")

    print(f"Loaded CNN baseline + {len(gnn_models)} GNN models on {device}")

    # Pre-load doctr text density detector once at startup.
    # Loading it per-request causes a hang (model weights fetched/loaded on every call).
    text_detector = None
    try:
        from src.text_features import create_text_detector
        text_detector = create_text_detector(type("Config", (), {"device": device})())
        print("Loaded doctr text density detector")
    except Exception as exc:
        print(f"Text density detector unavailable (will skip): {exc}")

    return {
        "feature_extractor": feature_extractor,
        "resnet50_classifier": resnet,
        "models": gnn_models,
        "multimodal_gat_model": multimodal_gat_model,
        "text_detector": text_detector,
    }
